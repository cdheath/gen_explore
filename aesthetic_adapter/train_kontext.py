"""Trains the Kontext-stage Aesthetic+Explore adapter.

Kontext is an image-*editing* model -- even "fresh" generation goes through a blank
white canvas as the `image=` input (see gen_explore/src/gen_explore/stages.py's
`generate_rough`). Its transformer call is correspondingly different from a plain
txt2img flow-matching loop (what the SD3.5 adapter trained against): the noisy target
latent and the *reference image's* latent get concatenated along the token sequence
before the transformer call, with position ids distinguishing them (frame index 0 vs 1),
and the transformer's output gets sliced back down to just the target portion afterward.
Verified against the real `FluxKontextPipeline.__call__`/`prepare_latents` source this
session (not guessed) -- this script calls that pipeline's own private helper methods
(`_encode_vae_image`, `_pack_latents`, `_prepare_latent_image_ids`, `encode_prompt`)
directly rather than re-deriving that mechanics by hand.

Loading mirrors `gen_explore.models.ModelBundle`'s quantization config (NF4 4-bit,
required per that module's own docstring for this to fit in 24GB VRAM at all) but
deliberately skips `enable_model_cpu_offload()` -- that's an inference-time memory trick
that complicates backward-pass bookkeeping and isn't needed here: the frozen 4-bit base
is already compact (~6GB), leaving real headroom for the adapter's own
activations/gradients/optimizer state without offloading anything. See this file's
`--memory-check` flag and notes/aesthetic-cross-attention-spec.md's FLUX plan section for
why this is a hard gate before any real run, not an afterthought, on this specific host
(31GB system RAM, confirmed the hard way this session).

Usage:
    # Architecture-only smoke test already covered by model.py/attach.py directly.
    # Real-model memory + correctness smoke test (REQUIRED before any real run):
    uv run python -m aesthetic_adapter.train_kontext --memory-check --limit-examples 4 --max-steps 2

    # Real run, once the memory check passes:
    uv run python -m aesthetic_adapter.train_kontext --max-steps <TBD from smoke test>
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import T5EncoderModel

from aesthetic_encoder import AestheticEncoder
from explore_encoder import ExploreEncoder

from .attach import FluxAestheticAdapter, attach
from .dataset import (
    DEFAULT_CAPTION_CACHE, DEFAULT_CHECKPOINT, DEFAULT_EXPLORE_CHECKPOINT, DEFAULT_EXPLORE_CONCEPTS,
    DEFAULT_EXPLORE_GLOBAL_MEAN,
    DEFAULT_NPZ, KontextAdapterDataset, build_examples, image_paths_from_npz,
)

EMBEDDING_RESEARCH = Path(__file__).resolve().parents[2] / "embedding_research"
import sys  # noqa: E402
sys.path.insert(0, str(EMBEDDING_RESEARCH))
from aesthetic_adapter_probe.caption import caption_files  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
GUIDANCE_SCALE = 3.5  # matches gen_explore/src/gen_explore/config.py's KONTEXT_PARAMS


def _most_distant_in_pool(vec: np.ndarray, examples, pool_idx: list[int], axis: str) -> int:
    """Per-step negative selection for the contrastive term: among a small random pool of
    candidate examples, returns the index of the one whose `axis` vector ("aesthetic" or
    "explore") is least similar to `vec`.

    v1 history: the original version always maximized aesthetic distance, which taught
    the network to differentiate aesthetically while giving no guarantee the paired
    example's explore vector differed at all -- confirmed as the root cause of
    near-identical output for opposite explore concepts (cosmic-terror vs
    serene-harmony). v2 fixed that by maximizing the *minimum* of both per-axis
    distances jointly -- but that stricter joint constraint measurably diluted peak
    aesthetic-axis differentiation on the real corpus retrain (registers that were
    dramatically distinct under v1 became nearly identical again under v2), a trade-off
    the cheap 16-image diagnostic didn't surface since it only measured each axis's gap
    growing in isolation, not whether one came at the other's expense.

    v3 (this version): alternate which single axis is maximized each step (see the
    caller, which passes `axis` from a per-step random choice) instead of requiring both
    simultaneously. Each axis gets full-strength, undiluted contrastive pressure on its
    own steps rather than always compromising."""
    v = vec / (np.linalg.norm(vec) + 1e-9)
    best_idx, best_sim = pool_idx[0], 2.0
    for idx in pool_idx:
        cand = np.asarray(getattr(examples[idx], f"{axis}_vector"), dtype=np.float32)
        cand = cand / (np.linalg.norm(cand) + 1e-9)
        sim = float(np.dot(v, cand))
        if sim < best_sim:
            best_sim, best_idx = sim, idx
    return best_idx


def load_kontext_pipe_for_training(device: str, dtype: torch.dtype) -> FluxKontextPipeline:
    quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
    )
    text_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        KONTEXT_REPO, subfolder="transformer", quantization_config=quant_config, torch_dtype=dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        KONTEXT_REPO, subfolder="text_encoder_2", quantization_config=text_quant_config, torch_dtype=dtype,
    )
    pipe = FluxKontextPipeline.from_pretrained(
        KONTEXT_REPO, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype,
    )
    # Deliberately no enable_model_cpu_offload() -- see module docstring.
    pipe.to(device)
    return pipe


def build_dataset(args) -> KontextAdapterDataset:
    paths = image_paths_from_npz(args.npz)
    if args.limit_examples:
        paths = paths[: args.limit_examples]
    logger.info("Bootstrap corpus: %d image-modality files (limit=%s)", len(paths), args.limit_examples)

    captions = caption_files(paths, Path(args.caption_cache), max_gpu_gib=args.captioner_gpu_gib)
    logger.info("Captions ready for %d/%d files", sum(1 for c in captions.values() if c), len(paths))

    aesthetic_encoder = AestheticEncoder(checkpoint=args.aesthetic_checkpoint, centroids_npz=args.npz)
    explore_encoder = ExploreEncoder(
        checkpoint=args.explore_checkpoint, concepts_npz=args.explore_concepts,
        global_mean_npy=args.explore_global_mean,
    )
    examples = build_examples(paths, captions, aesthetic_encoder, explore_encoder)
    logger.info("Training examples after caption/encode join: %d", len(examples))

    return KontextAdapterDataset(examples, resolution=args.resolution)


def _collate(batch):
    pixel_values, captions, aes_vecs, exp_vecs = zip(*batch)
    return (
        torch.from_numpy(np.stack(pixel_values)),
        list(captions),
        torch.from_numpy(np.stack(aes_vecs)),
        torch.from_numpy(np.stack(exp_vecs)),
    )


def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This training script requires CUDA (4-bit quantization needs a GPU).")
    dtype = torch.bfloat16

    dataset = build_dataset(args)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 training examples.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=2)

    logger.info("Loading %s (quantized NF4, frozen, %s, no cpu offload)...", KONTEXT_REPO, dtype)
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
    n_params = sum(p.numel() for p in adapter.parameters())
    logger.info("FluxAestheticAdapter trainable parameters: %d", n_params)

    scheduler = pipe.scheduler
    num_train_timesteps = scheduler.config.num_train_timesteps

    # 8-bit AdamW (bitsandbytes, already a dependency for model quantization): the
    # smoke test's first real attempt OOM'd by only 36MiB, specifically inside
    # optimizer.step()'s state allocation -- forward+backward through the frozen 4-bit
    # base + adapter fit fine, but standard AdamW's two fp32 moment buffers for ~750M+
    # adapter parameters (19 blocks x 2 sources x (to_k_ip+to_v_ip) at hidden_size=3072,
    # plus projectors) pushed it just over. 8-bit optimizer state cuts that ~4x.
    #
    # Gate params get their own higher-LR group, clipped separately from the K/V/projector
    # weights: a real 5,000-step run with everything in one group + one shared
    # clip_grad_norm_ left the 38 scalar gates stuck near their zero-init value
    # (mean|gate| plateaued at ~0.013 by step 500 and never moved again, loss flat too) --
    # diagnosed via aesthetic_adapter/diagnose_gate.py, which isolated the gate params at
    # lr=1e-2 (vs 1e-4 for everything else) and confirmed the conditioning DOES reduce loss
    # once the gate can actually move (20/20 logged steps showed a positive loss gap vs a
    # gate-forced-to-zero ablation on the same batch; mean|gate| reached 0.26 in 400 steps,
    # 20x further than the real run got in 5,000). The 716M K/V weights' gradient norm was
    # dominating the shared clip and the shared per-parameter-type lr=1e-4 was simply too
    # small a step size for a scalar starting at 0 -- not a sign the conditioning is
    # useless, a sign the gate specifically was starved.
    import bitsandbytes as bnb
    gate_params = [g for p in adapter.processors for g in p.gate]
    gate_param_ids = {id(g) for g in gate_params}
    other_params = [p for p in adapter.parameters() if id(p) not in gate_param_ids]
    optimizer = bnb.optim.AdamW8bit([
        {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
        {"params": other_params, "lr": args.lr},
    ])

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    def save_checkpoint(step: int, loss: float) -> None:
        torch.save({
            "step": step, "loss": loss,
            "adapter_config": {"n_tokens": adapter.aesthetic_projector.n_tokens,
                                "hidden_size": adapter.aesthetic_projector.out_width,
                                "num_blocks": len(adapter.processors)},
            "adapter_state": adapter.state_dict(),
        }, save_dir / "best.pt")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    blank_canvas = None  # built once resolution is known, per-batch below

    best_loss = float("inf")
    step = 0
    running, running_contrastive, running_n = 0.0, 0.0, 0

    while step < args.max_steps:
        for pixel_values, captions, aes_vecs, exp_vecs in loader:
            if step >= args.max_steps:
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
                # Reference (blank canvas) latents -- frame index 1, per Kontext's own convention.
                ref_latents_raw = pipe._encode_vae_image(image=blank_canvas, generator=None)
                ref_h, ref_w = ref_latents_raw.shape[2:]
                image_latents = pipe._pack_latents(ref_latents_raw, bsz, num_channels_latents, ref_h, ref_w)
                image_ids = pipe._prepare_latent_image_ids(bsz, ref_h // 2, ref_w // 2, device, dtype)
                image_ids[..., 0] = 1

                # Target (real image) latents -- frame index 0.
                target_latents_raw = pipe._encode_vae_image(image=pixel_values, generator=None)
                tgt_h, tgt_w = target_latents_raw.shape[2:]
                latents = pipe._pack_latents(target_latents_raw, bsz, num_channels_latents, tgt_h, tgt_w)
                latent_ids = pipe._prepare_latent_image_ids(bsz, tgt_h // 2, tgt_w // 2, device, dtype)

                img_ids = torch.cat([latent_ids, image_ids], dim=0)

                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=list(captions), device=device,
                )

            dropped = random.random() < args.cfg_dropout_p
            dual_ref = (not dropped) and random.random() < args.dual_ref_p
            if dropped:
                aes_ref = torch.zeros(bsz, 1, 256, device=device, dtype=torch.float32)
                exp_ref = torch.zeros(bsz, 1, 256, device=device, dtype=torch.float32)
            elif dual_ref:
                other = random.choice(dataset.examples)
                other_aes = torch.from_numpy(np.asarray(other.aesthetic_vector, dtype=np.float32)).to(device)
                other_exp = torch.from_numpy(np.asarray(other.explore_vector, dtype=np.float32)).to(device)
                aes_ref = torch.stack([aes_vecs[0], other_aes]).unsqueeze(0)
                exp_ref = torch.stack([exp_vecs[0], other_exp]).unsqueeze(0)
            else:
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

            # Contrastive term (see diagnose_contrastive.py for how this was validated):
            # without it, the training objective never penalizes the network for giving
            # the same prediction regardless of which vector it's conditioned on -- a
            # generic content-independent nudge already reduces average reconstruction
            # loss, so nothing pushes toward genuine per-vector differentiation. Only
            # applied on clean single-reference steps (not dropped/dual_ref, which have
            # their own separate purposes) -- same recipe the diagnostic validated: 500
            # steps showed cross_cond_gap consistently 0.4-1.8 (vs a flat ~0.001-0.01
            # without this term), confirming the network can learn real content-dependence
            # once something in the objective actually rewards it.
            contrastive_loss = torch.zeros((), device=device)
            contrastive_axis = None
            if not dropped and not dual_ref and args.contrastive_weight > 0:
                contrastive_axis = random.choice(("aesthetic", "explore"))
                cur_vec_np = (aes_vecs if contrastive_axis == "aesthetic" else exp_vecs)[0].detach().cpu().numpy()
                pool = random.sample(
                    range(len(dataset.examples)), min(args.neg_pool_size, len(dataset.examples)),
                )
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
                contrastive_loss = F.relu(args.contrastive_margin - pred_gap)

            loss = recon_loss + args.contrastive_weight * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            # Clipped separately from other_params -- a shared clip let the (far larger)
            # K/V gradient norm dominate and scale the gate's own gradient down with it.
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
            optimizer.step()

            running += recon_loss.item()
            running_contrastive += contrastive_loss.item()
            running_n += 1
            step += 1

            if args.memory_check and device == "cuda":
                logger.info(
                    "step %d  loss=%.4f  recon=%.4f  contrastive=%.4f  GPU peak alloc=%.2f GiB  GPU peak reserved=%.2f GiB",
                    step, loss.item(), recon_loss.item(), contrastive_loss.item(),
                    torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9,
                )

            if step % args.log_every == 0 or step == args.max_steps:
                avg = running / max(running_n, 1)
                avg_contrastive = running_contrastive / max(running_n, 1)
                mean_gate = float(torch.stack([
                    g.detach().abs() for p in adapter.processors for g in p.gate
                ]).mean())
                logger.info(
                    "step %d/%d  recon=%.4f  contrastive=%.4f  mean|gate|=%.4f  mode=%s  contrastive_axis=%s",
                    step, args.max_steps, avg, avg_contrastive, mean_gate,
                    "dropped" if dropped else ("dual" if dual_ref else "single"),
                    contrastive_axis,
                )
                with metrics_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": avg, "contrastive_loss": avg_contrastive, "mean_abs_gate": mean_gate,
                    }) + "\n")
                if avg < best_loss:
                    best_loss = avg
                    save_checkpoint(step, avg)
                running, running_contrastive, running_n = 0.0, 0.0, 0

            if step >= args.max_steps:
                break

    if device == "cuda":
        logger.info(
            "Final GPU peak alloc=%.2f GiB  peak reserved=%.2f GiB",
            torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9,
        )
    logger.info("Done. Best loss %.4f. Checkpoint: %s", best_loss, save_dir / "best.pt")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--aesthetic-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--explore-checkpoint", default=DEFAULT_EXPLORE_CHECKPOINT)
    p.add_argument("--explore-concepts", default=DEFAULT_EXPLORE_CONCEPTS)
    p.add_argument("--explore-global-mean", default=DEFAULT_EXPLORE_GLOBAL_MEAN)
    p.add_argument("--caption-cache", default=DEFAULT_CAPTION_CACHE)
    p.add_argument("--captioner-gpu-gib", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-lr", type=float, default=1e-2,
                    help="Separate, higher LR for the 38 per-block gate scalars -- see the "
                         "optimizer setup comment above for why these need their own group.")
    p.add_argument("--contrastive-margin", type=float, default=0.5,
                    help="Hinge margin (MSE units) for the contrastive term -- see "
                         "diagnose_contrastive.py for how this was validated.")
    p.add_argument("--contrastive-weight", type=float, default=1.0,
                    help="Weight on the contrastive term; 0 disables it (recovers the "
                         "gate-fix-only training loop).")
    p.add_argument("--neg-pool-size", type=int, default=4,
                    help="How many random candidates to consider each single-reference "
                         "step when picking the most-distant negative vector.")
    p.add_argument("--max-steps", type=int, default=75000)
    p.add_argument("--limit-examples", type=int, default=None)
    p.add_argument("--cfg-dropout-p", type=float, default=0.1)
    p.add_argument("--dual-ref-p", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--run-name", default="kontext_aesthetic_adapter")
    p.add_argument("--memory-check", action="store_true", help="Log GPU memory every step (smoke-test mode).")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
