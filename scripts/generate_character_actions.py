#!/usr/bin/env python3
"""CLI: generate one character across multiple action poses, keeping its
design consistent via a FLUX Kontext identity anchor. Each action is its own
file with N superimposed construction-sketch candidates (aligned restyle
variants of that one pose), like a single character element in
generate_scene.py -- but as a pose library, not a composed scene.

Generation is batched by stage (all Kontext calls, then all restyle calls)
rather than interleaved -- see models.py's docstring for why: swapping FLUX
pipelines mid-run costs tens of seconds each time.
"""

import argparse
import re
import time
from pathlib import Path

from gen_explore import config, stages
from gen_explore.manifest import Element, LayerManifest
from gen_explore.models import ModelBundle
from gen_explore.svg_export import save_candidate_svg
from gen_explore.vectorize import image_to_paths


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", required=True, help="Base character description")
    parser.add_argument("--action", action="append", required=True, help="Action/pose prompt (repeatable, 1+)")
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

    bundle = ModelBundle()

    # Stage A: everything that needs Kontext -- identity anchor, then one
    # posed subject per action, all while Kontext stays loaded.
    print("Generating identity anchor...")
    identity_image = stages.generate_identity_anchor(bundle, args.character, args.width, args.height, seed=args.seed)
    identity_image.save(layers_dir / "identity_anchor.png")

    action_ids = [f"action_{slugify(a)}" for a in args.action]
    posed_subjects = {}
    for z_index, (action_id, action_prompt) in enumerate(zip(action_ids, args.action)):
        print(f"[{action_id}] posed subject...")
        posed_seed = None if args.seed is None else args.seed + 1 + z_index * 100
        posed_subject = stages.generate_posed_subject(
            bundle, args.character, action_prompt, identity_image, args.width, args.height, seed=posed_seed
        )
        posed_subject.save(layers_dir / f"{action_id}_posed_subject_raw.png")
        posed_subjects[action_id] = (posed_subject, posed_seed)

    # Stage B: everything that needs the restyle ControlNet -- N aligned
    # candidates per action, all while that pipeline stays loaded.
    manifest = LayerManifest(width=args.width, height=args.height)
    for z_index, action_id in enumerate(action_ids):
        posed_subject, posed_seed = posed_subjects[action_id]
        candidate_paths = []
        for i in range(args.candidates):
            variant_seed = None if posed_seed is None else posed_seed + 1 + i
            print(f"[{action_id}] restyle candidate {i}...")
            candidate = stages.generate_variant(
                bundle, posed_subject, args.character, role="character", seed=variant_seed
            )
            candidate.save(layers_dir / f"{action_id}_candidate_{i}_raw.png")
            t0 = time.time()
            candidate_paths.append(image_to_paths(candidate))
            print(f"[{action_id}] candidate {i} vectorized in {time.time() - t0:.1f}s", flush=True)

        svg_rel_path = f"layers/{action_id}.svg"
        save_candidate_svg(candidate_paths, args.width, args.height, out_dir / svg_rel_path)

        candidate_ids = [f"candidate_{i}" for i in range(len(candidate_paths))]
        manifest.elements.append(
            Element(id=action_id, role="character", z_index=z_index, svg=svg_rel_path, candidate_ids=candidate_ids)
        )

    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
