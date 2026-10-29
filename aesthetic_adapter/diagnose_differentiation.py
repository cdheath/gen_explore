"""Diagnostic for the SECOND problem the gate-LR fix exposed: after training with the
gate isolated in its own high-LR group (see diagnose_gate.py and
train_kontext_gatefix.log), the adapter clearly perturbs generation now -- but visually,
all 4 aesthetic registers and all 4 explore concepts converge to the SAME generic
"bright, saturated, prototypical park photo" look at scale=2.0, rather than each looking
distinct. Two opposite explore concepts (cosmic-terror vs serene-harmony) produced
near-identical output. That's consistent with: the gate (a single scalar per block)
learned "when to push" quickly under its 100x LR, but the K/V weights + projectors (the
~718M-param machinery that has to learn "push *how*, differently per input vector") were
left at the original lr=1e-4 and, after only 5,000 steps, apparently haven't learned
content-dependent behavior yet -- just one generic helpful direction.

This script tests that directly and cheaply, without waiting for another multi-hour real
run and without trusting rendered-image comparisons (expensive, and confounded by 20
sampling steps' worth of drift). The core measurement: pick two training images whose own
aesthetic vectors are maximally far apart (real corpus vectors, not eval-time register
centroids), fix ONE noisy-latent/timestep/target setup, and compare the model's predicted
velocity when conditioned on vector A vs vector B on that *identical* input. If the
adapter is still content-independent, that prediction gap should be small and not grow
with training. If raising the K/V/projector LR lets it become content-dependent, the gap
should grow measurably above its (untrained-adapter) starting point.

Trains with base_lr raised (default 1e-3, 10x the original run's 1e-4) while keeping the
already-confirmed-working gate_lr=1e-2 unchanged, on a modestly larger overfit set
(default 16 images, vs diagnose_gate.py's 6) so there's enough real corpus diversity to
find a genuinely-distant probe pair.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .attach import FluxAestheticAdapter, attach
from .train_kontext import (
    GUIDANCE_SCALE, _collate, build_dataset, load_kontext_pipe_for_training, parse_args as train_parse_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _most_distant_pair(examples) -> tuple[int, int, float]:
    """Returns (i, j, cosine) for the pair of examples with the lowest aesthetic-vector
    cosine similarity in the overfit set -- the most content-distant probe pair
    available, so the differentiation test isn't diluted by picking two similar images."""
    vecs = np.stack([np.asarray(e.aesthetic_vector, dtype=np.float32) for e in examples])
    norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = norms @ norms.T
    np.fill_diagonal(sims, 2.0)  # exclude self-pairs from the argmin
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    return int(i), int(j), float(sims[i, j])


def run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This diagnostic requires CUDA (4-bit quantization needs a GPU).")
    dtype = torch.bfloat16

    dataset_args = train_parse_args([])
    dataset_args.npz = args.npz
    dataset_args.aesthetic_checkpoint = args.aesthetic_checkpoint
    dataset_args.explore_checkpoint = args.explore_checkpoint
    dataset_args.explore_concepts = args.explore_concepts
    dataset_args.explore_global_mean = args.explore_global_mean
    dataset_args.caption_cache = args.caption_cache
    dataset_args.captioner_gpu_gib = args.captioner_gpu_gib
    dataset_args.resolution = args.resolution
    dataset_args.limit_examples = args.limit_examples

    dataset = build_dataset(dataset_args)
    if len(dataset) < 4:
        raise RuntimeError("Need at least 4 training examples to pick a meaningful probe pair.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=0)

    probe_i, probe_j, probe_cos = _most_distant_pair(dataset.examples)
    logger.info(
        "Overfit set: %d examples. Probe pair: idx %d vs %d (aesthetic cosine=%.3f, most distant available)",
        len(dataset), probe_i, probe_j, probe_cos,
    )

    logger.info("Loading Kontext pipeline (quantized NF4, frozen, no cpu offload)...")
    pipe = load_kontext_pipe_for_training(device, dtype)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.eval()
    pipe.vae.eval()

    adapter = FluxAestheticAdapter().to(device=device, dtype=torch.float32)
    attach(pipe.transformer, adapter)

    gate_params = [g for p in adapter.processors for g in p.gate]
    gate_param_ids = {id(g) for g in gate_params}
    other_params = [p for p in adapter.parameters() if id(p) not in gate_param_ids]
    logger.info(
        "Param groups: %d gate scalars @ lr=%.4g, %d other params (%d numel) @ lr=%.4g",
        len(gate_params), args.gate_lr, len(other_params),
        sum(p.numel() for p in other_params), args.base_lr,
    )

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit([
        {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
        {"params": other_params, "lr": args.base_lr},
    ])

    scheduler = pipe.scheduler
    num_train_timesteps = scheduler.config.num_train_timesteps

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    num_channels_latents = pipe.transformer.config.in_channels // 4

    def load_example_tensors(idx: int):
        pixel_values, caption, aes_vec, exp_vec = dataset[idx]
        pixel_values = torch.from_numpy(pixel_values).unsqueeze(0).to(device=device, dtype=dtype)
        aes_vec = torch.from_numpy(np.asarray(aes_vec, dtype=np.float32)).to(device)
        exp_vec = torch.from_numpy(np.asarray(exp_vec, dtype=np.float32)).to(device)
        return pixel_values, caption, aes_vec, exp_vec

    # Fixed probe setup, built once: example i's own real target image + a FIXED noise
    # draw + FIXED mid-range timestep, so the only thing that ever changes across the two
    # probe forward passes is which conditioning vector is fed in.
    canvas_img = Image.new("RGB", (args.resolution, args.resolution), "white")
    canvas_arr = (np.asarray(canvas_img, dtype=np.float32) / 127.5) - 1.0
    canvas_arr = canvas_arr.transpose(2, 0, 1)
    probe_canvas = torch.from_numpy(canvas_arr[None]).to(device=device, dtype=dtype)

    with torch.no_grad():
        probe_pixels, probe_caption, probe_vec_a_aes, probe_vec_a_exp = load_example_tensors(probe_i)
        _, _, probe_vec_b_aes, probe_vec_b_exp = load_example_tensors(probe_j)

        ref_latents_raw = pipe._encode_vae_image(image=probe_canvas, generator=None)
        ref_h, ref_w = ref_latents_raw.shape[2:]
        probe_image_latents = pipe._pack_latents(ref_latents_raw, 1, num_channels_latents, ref_h, ref_w)
        probe_image_ids = pipe._prepare_latent_image_ids(1, ref_h // 2, ref_w // 2, device, dtype)
        probe_image_ids[..., 0] = 1

        target_latents_raw = pipe._encode_vae_image(image=probe_pixels, generator=None)
        tgt_h, tgt_w = target_latents_raw.shape[2:]
        probe_target_latents = pipe._pack_latents(target_latents_raw, 1, num_channels_latents, tgt_h, tgt_w)
        probe_latent_ids = pipe._prepare_latent_image_ids(1, tgt_h // 2, tgt_w // 2, device, dtype)
        probe_img_ids = torch.cat([probe_latent_ids, probe_image_ids], dim=0)

        probe_prompt_embeds, probe_pooled, probe_text_ids = pipe.encode_prompt(
            prompt=[probe_caption], device=device,
        )

        gen = torch.Generator(device=device).manual_seed(args.probe_seed)
        probe_noise = torch.randn(probe_target_latents.shape, generator=gen, device=device, dtype=probe_target_latents.dtype)
        mid_idx = num_train_timesteps // 2
        probe_sigma = scheduler.sigmas[mid_idx].to(device=device, dtype=dtype)
        probe_timestep = scheduler.timesteps[mid_idx].to(device=device).unsqueeze(0)
        probe_noisy_latents = (1.0 - probe_sigma) * probe_target_latents + probe_sigma * probe_noise
        probe_hidden_states = torch.cat([probe_noisy_latents, probe_image_latents], dim=1)
        probe_guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32)

    def probe_pred(aes_vec: torch.Tensor, exp_vec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = pipe.transformer(
                hidden_states=probe_hidden_states,
                timestep=probe_timestep / 1000,
                guidance=probe_guidance,
                pooled_projections=probe_pooled,
                encoder_hidden_states=probe_prompt_embeds,
                txt_ids=probe_text_ids,
                img_ids=probe_img_ids,
                joint_attention_kwargs={
                    "ip_adapter_image_embeds": [aes_vec.view(1, 1, -1), exp_vec.view(1, 1, -1)],
                },
                return_dict=False,
            )[0]
            return pred[:, : probe_target_latents.size(1)].float()

    # Baseline (untrained adapter, gate~0) cross-conditioning gap -- what "no
    # differentiation yet" looks like, to compare growth against.
    pred_a0 = probe_pred(probe_vec_a_aes, probe_vec_a_exp)
    pred_b0 = probe_pred(probe_vec_b_aes, probe_vec_b_exp)
    baseline_gap = F.mse_loss(pred_a0, pred_b0).item()
    logger.info("Baseline (untrained) cross-conditioning MSE gap: %.6f", baseline_gap)

    blank_canvas = None
    step = 0
    running, running_n = 0.0, 0

    while step < args.steps:
        for pixel_values, captions, aes_vecs, exp_vecs in loader:
            if step >= args.steps:
                break

            pixel_values = pixel_values.to(device=device, dtype=dtype)
            aes_vecs = aes_vecs.to(device=device, dtype=torch.float32)
            exp_vecs = exp_vecs.to(device=device, dtype=torch.float32)
            bsz = pixel_values.shape[0]

            if blank_canvas is None or blank_canvas.shape[0] != bsz:
                blank_canvas = probe_canvas.expand(bsz, -1, -1, -1).contiguous()

            with torch.no_grad():
                ref_latents_raw = pipe._encode_vae_image(image=blank_canvas, generator=None)
                ref_h, ref_w = ref_latents_raw.shape[2:]
                image_latents = pipe._pack_latents(ref_latents_raw, bsz, num_channels_latents, ref_h, ref_w)
                image_ids = pipe._prepare_latent_image_ids(bsz, ref_h // 2, ref_w // 2, device, dtype)
                image_ids[..., 0] = 1

                target_latents_raw = pipe._encode_vae_image(image=pixel_values, generator=None)
                tgt_h, tgt_w = target_latents_raw.shape[2:]
                latents = pipe._pack_latents(target_latents_raw, bsz, num_channels_latents, tgt_h, tgt_w)
                latent_ids = pipe._prepare_latent_image_ids(bsz, tgt_h // 2, tgt_w // 2, device, dtype)

                img_ids = torch.cat([latent_ids, image_ids], dim=0)

                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=list(captions), device=device,
                )

            aes_ref = aes_vecs.view(bsz, 1, -1)
            exp_ref = exp_vecs.view(bsz, 1, -1)

            noise = torch.randn_like(latents)
            u = torch.sigmoid(torch.randn(bsz))
            indices = (u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
            sigmas = scheduler.sigmas[indices].to(device=device, dtype=dtype).view(-1, 1, 1)
            timesteps = scheduler.timesteps[indices].to(device=device)

            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target_velocity = noise - latents

            hidden_states = torch.cat([noisy_latents, image_latents], dim=1)
            guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32).expand(bsz)

            model_pred = pipe.transformer(
                hidden_states=hidden_states,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs={"ip_adapter_image_embeds": [aes_ref, exp_ref]},
                return_dict=False,
            )[0]
            model_pred = model_pred[:, : latents.size(1)]

            loss = F.mse_loss(model_pred.float(), target_velocity.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
            optimizer.step()

            running += loss.item()
            running_n += 1
            step += 1

            cross_gap = None
            if step % args.probe_every == 0 or step == args.steps:
                pred_a = probe_pred(probe_vec_a_aes, probe_vec_a_exp)
                pred_b = probe_pred(probe_vec_b_aes, probe_vec_b_exp)
                cross_gap = F.mse_loss(pred_a, pred_b).item()

            if step % args.log_every == 0 or step == args.steps:
                avg = running / max(running_n, 1)
                mean_gate = float(torch.stack([g.detach().abs() for g in gate_params]).mean())
                msg = f"step {step}/{args.steps}  loss={avg:.4f}  mean|gate|={mean_gate:.4f}"
                if cross_gap is not None:
                    # baseline_gap is always exactly 0 at zero-init (gate=0 makes any
                    # conditioning a no-op) -- not a useful denominator. What matters is
                    # whether cross_gap grows to a magnitude comparable to the training
                    # loss itself (real, exploitable content-dependence) vs staying
                    # orders of magnitude smaller (negligible/noise-level).
                    msg += f"  cross_cond_gap={cross_gap:.6f}  (loss={avg:.4f}, gap/loss={cross_gap / max(avg, 1e-9):.4f})"
                logger.info(msg)
                with out_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": avg, "mean_abs_gate": mean_gate,
                        "cross_cond_gap": cross_gap, "baseline_gap": baseline_gap,
                    }) + "\n")
                running, running_n = 0.0, 0

    logger.info("Diagnostic done. Log: %s", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None)
    p.add_argument("--aesthetic-checkpoint", default=None)
    p.add_argument("--explore-checkpoint", default=None)
    p.add_argument("--explore-concepts", default=None)
    p.add_argument("--explore-global-mean", default=None)
    p.add_argument("--caption-cache", default=None)
    p.add_argument("--captioner-gpu-gib", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit-examples", type=int, default=16)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--gate-lr", type=float, default=1e-2)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--output", default="aesthetic_adapter/diagnose_differentiation.jsonl")
    args = p.parse_args()
    defaults = train_parse_args([])
    for field in ("npz", "aesthetic_checkpoint", "explore_checkpoint", "explore_concepts",
                  "explore_global_mean", "caption_cache"):
        if getattr(args, field) is None:
            setattr(args, field, getattr(defaults, field))
    return args


if __name__ == "__main__":
    run(parse_args())
