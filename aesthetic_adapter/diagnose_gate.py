"""Diagnostic for why the Kontext-stage adapter's learned gates plateaued near their
zero-init value (~0.013 mean magnitude) after 5,000 real-run training steps while loss
stayed flat (see aesthetic_adapter/train_kontext.log: gate rises 0.002->0.014 in the
first 500 steps, then drifts down to 0.013 and stays there through step 5000; loss
0.4369 at step 500 vs 0.4413 at step 5000).

Two competing hypotheses, and this script's job is to tell them apart empirically rather
than guess:

  (a) LR/gradient starvation: `clip_grad_norm_(adapter.parameters(), max_norm=1.0)` in
      train_kontext.py clips ALL adapter params (the ~716M to_k_ip/to_v_ip weights AND
      the 38 scalar gates) together by one shared norm. If the K/V gradients are large,
      the shared clip could be scaling the gate's already-tiny gradient down further,
      and/or a uniform lr=1e-4 across parameter types of very different scale could be
      wrong for a scalar gate specifically.

  (b) Genuine ceiling: the conditioning tokens (256-d vector -> 4 tokens via a freshly
      initialized small MLP) just don't carry information useful for reducing the
      flow-matching MSE loss as currently wired, so Adam correctly found that ~0.013 is
      about as much of this contribution as helps (or that more actively hurts), and
      training longer would not change that.

Method: overfit hard on a handful of images (small enough to memorize), with:
  - No CFG dropout, no dual-ref noise (isolates the single-reference signal cleanly).
  - The gate parameters in their OWN optimizer param group at a much higher LR than the
    K/V/projector weights (tests hypothesis (a) directly -- if this was gradient/LR
    starvation, an isolated high-LR gate should move a lot more than it did in the real
    run).
  - Every `--ablate-every` steps, an extra no-grad forward pass on the SAME batch with
    the gate values temporarily zeroed, to measure whether the adapter's current gate is
    actually reducing loss below a matched no-adapter baseline (tests hypothesis (b) --
    if real_loss stays pinned to ablation_loss throughout, the conditioning isn't
    providing exploitable signal regardless of how far the gate itself moves).

Reuses train_kontext.py's pipeline loading / dataset building so this exercises the exact
same forward path as the real run, just on a tiny overfit set with different optimizer
settings.
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


def run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This diagnostic requires CUDA (4-bit quantization needs a GPU).")
    dtype = torch.bfloat16

    dataset_args = train_parse_args([])  # defaults, then override the ones we need
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
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 training examples.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=0)

    logger.info("Overfit set: %d examples", len(dataset))
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

    blank_canvas = None
    step = 0
    running, running_n = 0.0, 0

    def zero_gates():
        saved = [g.detach().clone() for g in gate_params]
        with torch.no_grad():
            for g in gate_params:
                g.zero_()
        return saved

    def restore_gates(saved):
        with torch.no_grad():
            for g, s in zip(gate_params, saved):
                g.copy_(s)

    while step < args.steps:
        for pixel_values, captions, aes_vecs, exp_vecs in loader:
            if step >= args.steps:
                break

            pixel_values = pixel_values.to(device=device, dtype=dtype)
            aes_vecs = aes_vecs.to(device=device, dtype=torch.float32)
            exp_vecs = exp_vecs.to(device=device, dtype=torch.float32)
            bsz = pixel_values.shape[0]

            if blank_canvas is None or blank_canvas.shape[0] != bsz:
                canvas_img = Image.new("RGB", (args.resolution, args.resolution), "white")
                canvas_arr = (np.asarray(canvas_img, dtype=np.float32) / 127.5) - 1.0
                canvas_arr = canvas_arr.transpose(2, 0, 1)
                blank_canvas = torch.from_numpy(np.stack([canvas_arr] * bsz)).to(device=device, dtype=dtype)

            num_channels_latents = pipe.transformer.config.in_channels // 4

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

            # Always single-reference, real vectors -- no CFG dropout, no dual-ref, to
            # isolate the cleanest possible signal for this overfit test.
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

            def forward_once():
                pred = pipe.transformer(
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
                return pred[:, : latents.size(1)]

            model_pred = forward_once()
            loss = F.mse_loss(model_pred.float(), target_velocity.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
            optimizer.step()

            running += loss.item()
            running_n += 1
            step += 1

            ablation_loss = None
            if step % args.ablate_every == 0 or step == args.steps:
                with torch.no_grad():
                    saved = zero_gates()
                    ablated_pred = forward_once()
                    ablation_loss = F.mse_loss(ablated_pred.float(), target_velocity.float()).item()
                    restore_gates(saved)

            if step % args.log_every == 0 or step == args.steps:
                avg = running / max(running_n, 1)
                mean_gate = float(torch.stack([g.detach().abs() for g in gate_params]).mean())
                max_gate = float(torch.stack([g.detach().abs() for g in gate_params]).max())
                msg = f"step {step}/{args.steps}  loss={avg:.4f}  mean|gate|={mean_gate:.4f}  max|gate|={max_gate:.4f}"
                if ablation_loss is not None:
                    msg += f"  ablation(gate=0)_loss={ablation_loss:.4f}  gap={ablation_loss - loss.item():+.4f}"
                logger.info(msg)
                with out_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": avg, "mean_abs_gate": mean_gate, "max_abs_gate": max_gate,
                        "ablation_loss": ablation_loss,
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
    p.add_argument("--limit-examples", type=int, default=6)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--gate-lr", type=float, default=1e-2)
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ablate-every", type=int, default=20)
    p.add_argument("--output", default="aesthetic_adapter/diagnose_gate.jsonl")
    args = p.parse_args()
    defaults = train_parse_args([])
    for field in ("npz", "aesthetic_checkpoint", "explore_checkpoint", "explore_concepts",
                  "explore_global_mean", "caption_cache"):
        if getattr(args, field) is None:
            setattr(args, field, getattr(defaults, field))
    return args


if __name__ == "__main__":
    run(parse_args())
