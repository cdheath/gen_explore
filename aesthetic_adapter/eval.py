"""Eval for the Kontext-stage Aesthetic+Explore adapter: a register/concept x gate_scale
x prompt grid (one source at a time, the other held at scale=0) plus a dual-source blend
row -- does combining an Aesthetic register and an Explore concept produce a genuine
combined effect, or does one source dominate/interfere with the other? Mirrors
embedding_research/aesthetic_adapter/eval.py's shape (grid + blend row, two-phase
generate-then-score for VRAM), adapted for two independently-scaled native sources
instead of one.

Usage:
    uv run python -m aesthetic_adapter.eval --checkpoint aesthetic_adapter/checkpoints/kontext_aesthetic_adapter/best.pt
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from aesthetic_encoder import AestheticEncoder
from explore_encoder import ExploreEncoder

from .attach import FluxAestheticAdapter, attach
from .dataset import (
    DEFAULT_CHECKPOINT, DEFAULT_EXPLORE_CHECKPOINT, DEFAULT_EXPLORE_CONCEPTS,
    DEFAULT_EXPLORE_GLOBAL_MEAN, DEFAULT_NPZ,
)
from .train_kontext import GUIDANCE_SCALE, load_kontext_pipe_for_training

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

EVAL_PROMPTS = ["a person standing in a park", "a house on a hill"]
EVAL_REGISTERS = ["elegiac", "uncanny", "idyllic", "sublime"]
EVAL_CONCEPTS = ["serene-harmony", "ominous-foreboding", "dramatic-tension", "cosmic-terror"]
EVAL_SCALES = [0.0, 1.0, 2.0]
BLEND_REGISTER = "elegiac"
BLEND_CONCEPT = "ominous-foreboding"


def load_adapter(checkpoint_path: str, device: str) -> FluxAestheticAdapter:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = FluxAestheticAdapter(**ckpt["adapter_config"])
    adapter.load_state_dict(ckpt["adapter_state"])
    return adapter.eval().to(device=device, dtype=torch.float32)


def _set_gate_scale(adapter: FluxAestheticAdapter, base_gates: list[tuple], aes_scale: float, exp_scale: float) -> None:
    """base_gates[i] = (aesthetic_gate, explore_gate) tensors for block i, captured at
    the as-trained values before any scaling. Independent per-source scaling -- the
    whole point of this adapter's design."""
    with torch.no_grad():
        for proc, (base_aes, base_exp) in zip(adapter.processors, base_gates):
            proc.gate[0].copy_(base_aes * aes_scale)
            proc.gate[1].copy_(base_exp * exp_scale)


def _blank_canvas(resolution: int) -> Image.Image:
    return Image.new("RGB", (resolution, resolution), "white")


def _generate(pipe, aes_vec: torch.Tensor, exp_vec: torch.Tensor, prompt: str, steps: int,
              resolution: int, seed: int, device) -> Image.Image:
    gen = torch.Generator(device=device).manual_seed(seed)
    # joint_attention_kwargs directly (not the pipeline's ip_adapter_image_embeds
    # convenience param) -- same mechanism already validated in train_kontext.py's
    # training loop, avoiding any surprises from prepare_ip_adapter_image_embeds'
    # own reshaping logic on a path this session hasn't exercised.
    return pipe(
        image=_blank_canvas(resolution),
        prompt=prompt,
        joint_attention_kwargs={"ip_adapter_image_embeds": [aes_vec, exp_vec]},
        num_inference_steps=steps, guidance_scale=GUIDANCE_SCALE,
        height=resolution, width=resolution, generator=gen,
    ).images[0]


def _zero_ref(device) -> torch.Tensor:
    return torch.zeros(1, 1, 256, device=device, dtype=torch.float32)


def run_grid(args, pipe, adapter, base_gates, device) -> list[dict]:
    aesthetic_encoder = AestheticEncoder(checkpoint=args.aesthetic_checkpoint, centroids_npz=args.npz)
    explore_encoder = ExploreEncoder(
        checkpoint=args.explore_checkpoint, concepts_npz=args.explore_concepts,
        global_mean_npy=args.explore_global_mean,
    )

    aes_centroids = {r: aesthetic_encoder.center(aesthetic_encoder.centroids[r]) for r in args.registers}
    exp_centroids = {}
    for c in args.concepts:
        idx = explore_encoder.concept_names.index(c)
        exp_centroids[c] = explore_encoder.center(explore_encoder.concept_centroids[idx])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Aesthetic-only: explore source held at scale 0 throughout.
    for register in args.registers:
        target = torch.tensor(aes_centroids[register], dtype=torch.float32, device=device).view(1, 1, -1)
        for prompt in args.prompts:
            for scale in args.scales:
                _set_gate_scale(adapter, base_gates, scale, 0.0)
                image = _generate(pipe, target, _zero_ref(device), prompt, args.steps, args.resolution, args.seed, device)
                tag = f"aesthetic_{register}_{scale}_{abs(hash(prompt)) % 10000}"
                path = out_dir / f"{tag}.png"
                image.save(path)
                rows.append({"source": "aesthetic", "label": register, "prompt": prompt, "scale": scale,
                             "aes_target": aes_centroids[register], "image": str(path)})
                print(f"generated aesthetic={register:16s} scale={scale:.1f}  {prompt!r} -> {path.name}")

    # Explore-only: aesthetic source held at scale 0 throughout.
    for concept in args.concepts:
        target = torch.tensor(exp_centroids[concept], dtype=torch.float32, device=device).view(1, 1, -1)
        for prompt in args.prompts:
            for scale in args.scales:
                _set_gate_scale(adapter, base_gates, 0.0, scale)
                image = _generate(pipe, _zero_ref(device), target, prompt, args.steps, args.resolution, args.seed, device)
                tag = f"explore_{concept}_{scale}_{abs(hash(prompt)) % 10000}"
                path = out_dir / f"{tag}.png"
                image.save(path)
                rows.append({"source": "explore", "label": concept, "prompt": prompt, "scale": scale,
                             "exp_target": exp_centroids[concept], "image": str(path)})
                print(f"generated explore={concept:16s} scale={scale:.1f}  {prompt!r} -> {path.name}")

    _set_gate_scale(adapter, base_gates, 1.0, 1.0)
    return rows, aes_centroids, exp_centroids


def run_blend(args, pipe, adapter, base_gates, aes_centroids, exp_centroids, device) -> list[dict]:
    """Blend row: an Aesthetic register and an Explore concept, alone and together, same
    prompt/seed -- does combining them produce a genuinely combined effect?"""
    out_dir = Path(args.output_dir)
    aes_target = torch.tensor(aes_centroids[args.blend_register], dtype=torch.float32, device=device).view(1, 1, -1)
    exp_target = torch.tensor(exp_centroids[args.blend_concept], dtype=torch.float32, device=device).view(1, 1, -1)
    prompt = args.blend_prompt
    rows = []

    conditions = [
        (f"{args.blend_register}_alone", 1.0, 0.0),
        (f"{args.blend_concept}_alone", 0.0, 1.0),
        ("both_together", 1.0, 1.0),
        ("neither", 0.0, 0.0),
    ]
    for name, aes_scale, exp_scale in conditions:
        _set_gate_scale(adapter, base_gates, aes_scale, exp_scale)
        image = _generate(pipe, aes_target, exp_target, prompt, args.steps, args.resolution, args.seed, device)
        path = out_dir / f"blend_{name}.png"
        image.save(path)
        rows.append({
            "kind": "blend", "label": name, "prompt": prompt,
            "aes_target": aes_centroids[args.blend_register], "exp_target": exp_centroids[args.blend_concept],
            "image": str(path),
        })
        print(f"generated blend row: {name} -> {path.name}")

    _set_gate_scale(adapter, base_gates, 1.0, 1.0)
    return rows


def score(args, rows: list[dict], device) -> None:
    aesthetic_encoder = AestheticEncoder(checkpoint=args.aesthetic_checkpoint, centroids_npz=args.npz)
    explore_encoder = ExploreEncoder(
        checkpoint=args.explore_checkpoint, concepts_npz=args.explore_concepts,
        global_mean_npy=args.explore_global_mean,
    )
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    for row in rows:
        image = Image.open(row["image"])
        aes_result = aesthetic_encoder.encode(row["image"], "image")
        aes_vector = aesthetic_encoder.center(np.asarray(aes_result["vector"], dtype=np.float32))
        exp_result = explore_encoder.encode(row["image"], "image")
        exp_vector = explore_encoder.center(np.asarray(exp_result["vector"], dtype=np.float32))

        if "aes_target" in row:
            row["cos_aes"] = float(aes_vector @ np.asarray(row["aes_target"], dtype=np.float32))
        if "exp_target" in row:
            row["cos_exp"] = float(exp_vector @ np.asarray(row["exp_target"], dtype=np.float32))

        clip_inputs = clip_processor(text=[row["prompt"]], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            row["clip_score"] = clip_model(**clip_inputs).logits_per_image.item()

        for k in ("aes_target", "exp_target"):
            row.pop(k, None)


def run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    pipe = load_kontext_pipe_for_training(device, dtype)
    adapter = load_adapter(args.checkpoint, device)
    attach(pipe.transformer, adapter)
    base_gates = [(p.gate[0].detach().clone(), p.gate[1].detach().clone()) for p in adapter.processors]
    print("mean|learned aesthetic gate|:", float(torch.stack([g[0].abs() for g in base_gates]).mean()))
    print("mean|learned explore gate|:  ", float(torch.stack([g[1].abs() for g in base_gates]).mean()))

    grid_rows, aes_centroids, exp_centroids = run_grid(args, pipe, adapter, base_gates, device)
    blend_rows = run_blend(args, pipe, adapter, base_gates, aes_centroids, exp_centroids, device)
    rows = grid_rows + blend_rows

    del pipe, adapter
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    score(args, rows, device)

    out_dir = Path(args.output_dir)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    print("\n--- mean cosine to target, by source and scale ---")
    for source in ("aesthetic", "explore"):
        for scale in args.scales:
            vals = [r["cos_aes" if source == "aesthetic" else "cos_exp"]
                    for r in rows if r.get("source") == source and r["scale"] == scale]
            if vals:
                print(f"  {source:10s} scale={scale:.1f}  mean cos={np.mean(vals):+.4f}")

    print("\n--- mean CLIP text-alignment, by source and scale ---")
    for source in ("aesthetic", "explore"):
        for scale in args.scales:
            vals = [r["clip_score"] for r in rows if r.get("source") == source and r["scale"] == scale]
            if vals:
                print(f"  {source:10s} scale={scale:.1f}  mean clip={np.mean(vals):6.2f}")

    print("\n--- blend row (read the images in", out_dir, ") ---")
    for r in rows:
        if r.get("kind") == "blend":
            print(f"  {r['label']:20s} cos_aes={r.get('cos_aes', float('nan')):+.3f}  "
                  f"cos_exp={r.get('cos_exp', float('nan')):+.3f}  clip={r['clip_score']:6.2f}  {r['image']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aesthetic-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--explore-checkpoint", default=DEFAULT_EXPLORE_CHECKPOINT)
    p.add_argument("--explore-concepts", default=DEFAULT_EXPLORE_CONCEPTS)
    p.add_argument("--explore-global-mean", default=DEFAULT_EXPLORE_GLOBAL_MEAN)
    p.add_argument("--output-dir", default="aesthetic_adapter/eval_output")
    p.add_argument("--prompts", nargs="+", default=EVAL_PROMPTS)
    p.add_argument("--scales", nargs="+", type=float, default=EVAL_SCALES)
    p.add_argument("--registers", nargs="+", default=EVAL_REGISTERS)
    p.add_argument("--concepts", nargs="+", default=EVAL_CONCEPTS)
    p.add_argument("--blend-register", default=BLEND_REGISTER)
    p.add_argument("--blend-concept", default=BLEND_CONCEPT)
    p.add_argument("--blend-prompt", default="a person standing in a park")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)  # matches gen_explore's KONTEXT_PARAMS
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
