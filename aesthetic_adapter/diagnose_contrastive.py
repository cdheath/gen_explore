"""Step 1 of the differentiation fix (see notes/plan): the LR-only fix
(diagnose_differentiation.py) showed raising the K/V/projector LR 10x does not make the
adapter content-dependent -- the cross-conditioning prediction gap stayed flat (~0.1-2%
of the loss) across 500 steps with no growth trend. Root cause: the training objective
("denoise this one real image, using its own vector as a hint") never directly penalizes
giving the same answer regardless of which vector is fed. A generic, content-independent
nudge already reduces average reconstruction loss, so nothing pushes the network past
that toward genuine per-vector differentiation.

This script adds the missing pressure directly: a pairwise contrastive term. Each step,
alongside the normal (target image i, its own vector) reconstruction pass, a second
forward pass runs on the *same* noisy hidden_states/timestep/target but with a different
example's vector (the one least similar to vec_i, sampled from a small random pool each
step -- mirrors diagnose_differentiation.py's `_most_distant_pair`, done online instead of
once). A hinge loss then pushes the two predictions apart up to a margin:

    contrastive = relu(margin - MSE(pred_i, pred_j))
    loss = recon_loss + contrastive_weight * contrastive

The hinge means the network is never rewarded for *unbounded* divergence between
differently-conditioned predictions (which could otherwise incentivize increasingly
extreme/incoherent output purely to maximize a difference) -- only for reaching some
minimum separation.

Cost: this doubles the transformer forward+backward per step (VAE/text encode and the
probe setup stay shared and single). Expect roughly 2x wall-clock vs the gate-fix run's
per-step cost, and higher peak memory (two forward activation graphs alive before
backward) -- watch GPU memory here before ever running this against the real corpus.

Reuses the exact same fixed-probe-pair cross_cond_gap measurement as
diagnose_differentiation.py so results are directly comparable: the pass/fail bar is
whether that gap now shows a real growth trend, not just a nonzero value.

v2 update: the first real contrastive run (train_kontext_contrastive.log) fixed
aesthetic-axis differentiation convincingly (elegiac/uncanny/idyllic/sublime now visually
distinct) but two *opposite* explore concepts (cosmic-terror vs serene-harmony) still
rendered as nearly the same image. Root cause: `_most_distant_in_pool`'s negative
selection maximized aesthetic-vector distance only, so the paired negative's explore
vector had no guaranteed separation at all. Fixed to maximize the *minimum* of the two
per-axis distances, forcing genuine separation on whichever axis is the bottleneck. This
script tracks TWO cross_cond_gap probes (one per axis) so both can be verified.

v3 update: v2's real corpus retrain DID fix explore-axis differentiation (concepts now
visually distinct), but the stricter joint "minimum of both distances" constraint
measurably diluted aesthetic-axis differentiation compared to v1 -- registers that were
dramatically distinct under v1's aesthetic-only selection became nearly identical again.
That trade-off wasn't visible in v2's own diagnostic run (which only measured each axis's
gap growing in isolation, never whether one came at the other's expense) -- a real gap in
this diagnostic's coverage, now addressed differently: rather than a single joint
selection rule, `_most_distant_in_pool` now takes an `axis` argument and the caller
alternates which single axis is maximized each step (50/50 aesthetic vs explore), so each
axis gets full-strength, undiluted contrastive pressure on its own steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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


def _most_distant_pair(examples, vector_attr: str = "aesthetic_vector") -> tuple[int, int, float]:
    """Same as diagnose_differentiation.py's helper, generalized to either axis -- used
    once per axis, to pick the fixed probe pair for that axis's cross_cond_gap
    measurement."""
    vecs = np.stack([np.asarray(getattr(e, vector_attr), dtype=np.float32) for e in examples])
    norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = norms @ norms.T
    np.fill_diagonal(sims, 2.0)
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    return int(i), int(j), float(sims[i, j])


def _most_distant_in_pool(vec: np.ndarray, examples, pool_idx: list[int], axis: str) -> int:
    """Per-step negative selection: among a small random pool of candidate examples,
    returns the index of the one whose `axis` vector ("aesthetic" or "explore") is least
    similar to `vec`. See module docstring's "v3 update": the caller alternates axis per
    step rather than requiring joint separation on both, which diluted aesthetic-axis
    strength in v2's real retrain."""
    v = vec / (np.linalg.norm(vec) + 1e-9)
    best_idx, best_sim = pool_idx[0], 2.0
    for idx in pool_idx:
        cand = np.asarray(getattr(examples[idx], f"{axis}_vector"), dtype=np.float32)
        cand = cand / (np.linalg.norm(cand) + 1e-9)
        sim = float(np.dot(v, cand))
        if sim < best_sim:
            best_sim, best_idx = sim, idx
    return best_idx


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
    if len(dataset) < max(4, args.neg_pool_size + 1):
        raise RuntimeError("Need enough training examples to pick a meaningful probe pair and negative pool.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=0)

    aes_probe_i, aes_probe_j, aes_probe_cos = _most_distant_pair(dataset.examples, "aesthetic_vector")
    exp_probe_i, exp_probe_j, exp_probe_cos = _most_distant_pair(dataset.examples, "explore_vector")
    logger.info(
        "Overfit set: %d examples. Aesthetic probe pair: idx %d vs %d (cosine=%.3f). "
        "Explore probe pair: idx %d vs %d (cosine=%.3f).",
        len(dataset), aes_probe_i, aes_probe_j, aes_probe_cos, exp_probe_i, exp_probe_j, exp_probe_cos,
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
        "Param groups: %d gate scalars @ lr=%.4g, %d other params (%d numel) @ lr=%.4g. "
        "Contrastive margin=%.4g weight=%.4g",
        len(gate_params), args.gate_lr, len(other_params),
        sum(p.numel() for p in other_params), args.base_lr, args.margin, args.contrastive_weight,
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

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    num_channels_latents = pipe.transformer.config.in_channels // 4

    def load_example_tensors(idx: int):
        pixel_values, caption, aes_vec, exp_vec = dataset[idx]
        pixel_values = torch.from_numpy(pixel_values).unsqueeze(0).to(device=device, dtype=dtype)
        aes_vec = torch.from_numpy(np.asarray(aes_vec, dtype=np.float32)).to(device)
        exp_vec = torch.from_numpy(np.asarray(exp_vec, dtype=np.float32)).to(device)
        return pixel_values, caption, aes_vec, exp_vec

    canvas_img = Image.new("RGB", (args.resolution, args.resolution), "white")
    canvas_arr = (np.asarray(canvas_img, dtype=np.float32) / 127.5) - 1.0
    canvas_arr = canvas_arr.transpose(2, 0, 1)
    probe_canvas = torch.from_numpy(canvas_arr[None]).to(device=device, dtype=dtype)

    def build_probe(idx_i: int, idx_j: int, seed: int):
        """Fixed probe context for one axis: example idx_i's own real target image + a
        FIXED noise draw + FIXED mid-range timestep, so the only thing that changes across
        the two probe forward passes (idx_i's vs idx_j's vectors) is the conditioning."""
        with torch.no_grad():
            pixels, caption, vec_i_aes, vec_i_exp = load_example_tensors(idx_i)
            _, _, vec_j_aes, vec_j_exp = load_example_tensors(idx_j)

            ref_latents_raw = pipe._encode_vae_image(image=probe_canvas, generator=None)
            ref_h, ref_w = ref_latents_raw.shape[2:]
            image_latents = pipe._pack_latents(ref_latents_raw, 1, num_channels_latents, ref_h, ref_w)
            image_ids = pipe._prepare_latent_image_ids(1, ref_h // 2, ref_w // 2, device, dtype)
            image_ids[..., 0] = 1

            target_latents_raw = pipe._encode_vae_image(image=pixels, generator=None)
            tgt_h, tgt_w = target_latents_raw.shape[2:]
            target_latents = pipe._pack_latents(target_latents_raw, 1, num_channels_latents, tgt_h, tgt_w)
            latent_ids = pipe._prepare_latent_image_ids(1, tgt_h // 2, tgt_w // 2, device, dtype)
            img_ids = torch.cat([latent_ids, image_ids], dim=0)

            prompt_embeds, pooled, text_ids = pipe.encode_prompt(prompt=[caption], device=device)

            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(target_latents.shape, generator=gen, device=device, dtype=target_latents.dtype)
            mid_idx = num_train_timesteps // 2
            sigma = scheduler.sigmas[mid_idx].to(device=device, dtype=dtype)
            timestep = scheduler.timesteps[mid_idx].to(device=device).unsqueeze(0)
            noisy_latents = (1.0 - sigma) * target_latents + sigma * noise
            hidden_states = torch.cat([noisy_latents, image_latents], dim=1)
            guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32)

        return {
            "hidden_states": hidden_states, "timestep": timestep, "guidance": guidance,
            "pooled": pooled, "prompt_embeds": prompt_embeds, "text_ids": text_ids, "img_ids": img_ids,
            "target_size": target_latents.size(1),
            "vec_i": (vec_i_aes, vec_i_exp), "vec_j": (vec_j_aes, vec_j_exp),
        }

    def probe_pred(probe: dict, aes_vec: torch.Tensor, exp_vec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = pipe.transformer(
                hidden_states=probe["hidden_states"],
                timestep=probe["timestep"] / 1000,
                guidance=probe["guidance"],
                pooled_projections=probe["pooled"],
                encoder_hidden_states=probe["prompt_embeds"],
                txt_ids=probe["text_ids"],
                img_ids=probe["img_ids"],
                joint_attention_kwargs={
                    "ip_adapter_image_embeds": [aes_vec.view(1, 1, -1), exp_vec.view(1, 1, -1)],
                },
                return_dict=False,
            )[0]
            return pred[:, : probe["target_size"]].float()

    aes_probe = build_probe(aes_probe_i, aes_probe_j, args.probe_seed)
    exp_probe = build_probe(exp_probe_i, exp_probe_j, args.probe_seed + 1)

    def cross_gap_for(probe: dict) -> float:
        vec_i_aes, vec_i_exp = probe["vec_i"]
        vec_j_aes, vec_j_exp = probe["vec_j"]
        pred_i = probe_pred(probe, vec_i_aes, vec_i_exp)
        pred_j = probe_pred(probe, vec_j_aes, vec_j_exp)
        return F.mse_loss(pred_i, pred_j).item()

    baseline_gap_aes = cross_gap_for(aes_probe)
    baseline_gap_exp = cross_gap_for(exp_probe)
    logger.info(
        "Baseline (untrained) cross-conditioning MSE gap: aesthetic=%.6f  explore=%.6f",
        baseline_gap_aes, baseline_gap_exp,
    )

    blank_canvas = None
    step = 0
    running_recon, running_contrastive, running_n = 0.0, 0.0, 0

    while step < args.steps:
        for batch_idx, (pixel_values, captions, aes_vecs, exp_vecs) in enumerate(loader):
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

            recon_loss = F.mse_loss(model_pred.float(), target_velocity.float())

            # Contrastive term: same hidden_states/timestep/target, a different vector.
            # bsz is always 1 here (matches the rest of this training family): draw a
            # small random pool of OTHER dataset indices and pick the one least similar on
            # ONE axis, alternated 50/50 per step (see _most_distant_in_pool's docstring).
            contrastive_axis = random.choice(("aesthetic", "explore"))
            cur_vec_np = (aes_vecs if contrastive_axis == "aesthetic" else exp_vecs)[0].detach().cpu().numpy()
            pool = random.sample(range(len(dataset.examples)), min(args.neg_pool_size, len(dataset.examples)))
            neg_idx = _most_distant_in_pool(cur_vec_np, dataset.examples, pool, contrastive_axis)
            neg_aes = torch.from_numpy(
                np.asarray(dataset.examples[neg_idx].aesthetic_vector, dtype=np.float32)
            ).to(device).view(1, 1, -1)
            neg_exp = torch.from_numpy(
                np.asarray(dataset.examples[neg_idx].explore_vector, dtype=np.float32)
            ).to(device).view(1, 1, -1)

            neg_pred = pipe.transformer(
                hidden_states=hidden_states,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs={"ip_adapter_image_embeds": [neg_aes, neg_exp]},
                return_dict=False,
            )[0]
            neg_pred = neg_pred[:, : latents.size(1)]

            pred_gap = F.mse_loss(model_pred.float(), neg_pred.float())
            contrastive_loss = F.relu(args.margin - pred_gap)

            loss = recon_loss + args.contrastive_weight * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
            optimizer.step()

            running_recon += recon_loss.item()
            running_contrastive += contrastive_loss.item()
            running_n += 1
            step += 1

            cross_gap_aes, cross_gap_exp = None, None
            if step % args.probe_every == 0 or step == args.steps:
                cross_gap_aes = cross_gap_for(aes_probe)
                cross_gap_exp = cross_gap_for(exp_probe)

            if step % args.log_every == 0 or step == args.steps:
                avg_recon = running_recon / max(running_n, 1)
                avg_contrastive = running_contrastive / max(running_n, 1)
                mean_gate = float(torch.stack([g.detach().abs() for g in gate_params]).mean())
                msg = (
                    f"step {step}/{args.steps}  recon={avg_recon:.4f}  contrastive={avg_contrastive:.4f}  "
                    f"mean|gate|={mean_gate:.4f}"
                )
                if cross_gap_aes is not None:
                    msg += (
                        f"  cross_gap_aes={cross_gap_aes:.6f}  cross_gap_exp={cross_gap_exp:.6f}  "
                        f"(aes/recon={cross_gap_aes / max(avg_recon, 1e-9):.4f}, "
                        f"exp/recon={cross_gap_exp / max(avg_recon, 1e-9):.4f})"
                    )
                logger.info(msg)
                with out_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "recon_loss": avg_recon, "contrastive_loss": avg_contrastive,
                        "mean_abs_gate": mean_gate, "cross_gap_aes": cross_gap_aes, "cross_gap_exp": cross_gap_exp,
                        "baseline_gap_aes": baseline_gap_aes, "baseline_gap_exp": baseline_gap_exp,
                    }) + "\n")
                running_recon, running_contrastive, running_n = 0.0, 0.0, 0

    if device == "cuda":
        logger.info(
            "Final GPU peak alloc=%.2f GiB  peak reserved=%.2f GiB",
            torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9,
        )
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
    p.add_argument("--margin", type=float, default=0.5,
                    help="Hinge margin for the contrastive term (in MSE units between "
                         "differently-conditioned predictions on the same input).")
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--neg-pool-size", type=int, default=4,
                    help="How many random candidates to consider each step when picking "
                         "the most-distant negative vector.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--output", default="aesthetic_adapter/diagnose_contrastive.jsonl")
    args = p.parse_args()
    defaults = train_parse_args([])
    for field in ("npz", "aesthetic_checkpoint", "explore_checkpoint", "explore_concepts",
                  "explore_global_mean", "caption_cache"):
        if getattr(args, field) is None:
            setattr(args, field, getattr(defaults, field))
    return args


if __name__ == "__main__":
    run(parse_args())
