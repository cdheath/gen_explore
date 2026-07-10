#!/usr/bin/env python3
"""CLI: generate background/character elements as N superimposed candidate
line-interpretations (vectorized to SVG), and write a layer manifest.

Generation is batched by stage (all Kontext calls across every element, then
all restyle calls) rather than interleaved per element -- see models.py's
docstring for why: swapping FLUX pipelines mid-run costs tens of seconds.
"""

import argparse
import time
from pathlib import Path

from gen_explore import config, stages
from gen_explore.manifest import Element, LayerManifest
from gen_explore.models import ModelBundle
from gen_explore.svg_export import save_candidate_svg
from gen_explore.vectorize import image_to_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", required=True, help="Prompt for the background element")
    parser.add_argument("--character", action="append", default=[], help="Prompt for a character (repeatable)")
    parser.add_argument("--fx", action="append", default=[], help="Prompt for an fx element (repeatable, optional)")
    parser.add_argument("--out", required=True, help="Output directory for manifest.json and layers/")
    parser.add_argument("--width", type=int, default=config.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=config.DEFAULT_HEIGHT)
    parser.add_argument("--candidates", type=int, default=config.NUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str]] = [("background", "background", args.background)]
    jobs += [(f"character_{i + 1}", "character", p) for i, p in enumerate(args.character)]
    jobs += [(f"fx_{i + 1}", "fx", p) for i, p in enumerate(args.fx)]

    bundle = ModelBundle()

    # Stage A: one Kontext generation per element -- "character" roles use
    # the neutral character_subject style (locking a recognizable
    # subject/pose before the construction-sketch language below fights it);
    # background/fx roles generate directly in their final style, since
    # candidate 0 for them is that raw generation itself, not a restyle.
    base_images = {}
    for element_id, role, prompt in jobs:
        subject_role = "character_subject" if role == "character" else role
        print(f"[{element_id}] base generation...")
        base_images[element_id] = stages.generate_rough(
            bundle, prompt, role=subject_role, width=args.width, height=args.height, seed=args.seed
        )

    # Stage B: restyle candidates -- "character" elements restyle every
    # candidate (none of them is the raw, un-styled subject); background/fx
    # elements keep their base generation as candidate 0 and restyle the rest.
    manifest = LayerManifest(width=args.width, height=args.height)
    for z_index, (element_id, role, prompt) in enumerate(jobs):
        base_image = base_images[element_id]
        candidates = [] if role == "character" else [base_image]
        start = 0 if role == "character" else 1
        for i in range(start, args.candidates):
            variant_seed = None if args.seed is None else args.seed + 1 + i
            print(f"[{element_id}] restyle candidate {i}...")
            candidates.append(stages.generate_variant(bundle, base_image, prompt, role=role, seed=variant_seed))

        candidate_paths = []
        for i, img in enumerate(candidates):
            img.save(out_dir / f"layers/{element_id}_candidate_{i}_raw.png")
            t0 = time.time()
            candidate_paths.append(image_to_paths(img))
            print(f"[{element_id}] candidate {i} vectorized in {time.time() - t0:.1f}s", flush=True)

        svg_rel_path = f"layers/{element_id}.svg"
        save_candidate_svg(candidate_paths, args.width, args.height, out_dir / svg_rel_path)

        candidate_ids = [f"candidate_{i}" for i in range(len(candidates))]
        manifest.elements.append(
            Element(id=element_id, role=role, z_index=z_index, svg=svg_rel_path, candidate_ids=candidate_ids)
        )

    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
