"""Public library API: create_character() and create_background(). Each is a
standalone call -- it builds its own models, generates everything, and fully
unloads them (see ModelBundle.unload_all()) before returning, so nothing is
left resident on GPU/CPU between calls. This trades repeated model-load cost
for call isolation, e.g. for embedding in a process/service that shouldn't
carry FLUX resident in memory between uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from gen_explore import config, stages
from gen_explore.manifest import Element, LayerManifest
from gen_explore.models import ModelBundle
from gen_explore.svg_export import build_candidate_svg
from gen_explore.vectorize import image_to_paths

ReferenceImage = Image.Image | str | Path


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _load_reference_image(image: ReferenceImage, width: int, height: int) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return image.convert("RGB").resize((width, height))


@dataclass
class ActionResult:
    action: str
    posed_subject: Image.Image
    candidates: list[Image.Image]
    svg: str


@dataclass
class CharacterResult:
    identity_image: Image.Image
    actions: dict[str, ActionResult] = field(default_factory=dict)
    manifest: LayerManifest | None = None


@dataclass
class BackgroundResult:
    candidates: list[Image.Image]
    svg: str
    manifest: LayerManifest | None = None


def _write_character_output(result: CharacterResult, output_dir: Path) -> None:
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    result.identity_image.save(layers_dir / "identity_anchor.png")
    for action_id, action_result in result.actions.items():
        action_result.posed_subject.save(layers_dir / f"{action_id}_posed_subject_raw.png")
        for i, candidate in enumerate(action_result.candidates):
            candidate.save(layers_dir / f"{action_id}_candidate_{i}_raw.png")
        (layers_dir / f"{action_id}.svg").write_text(action_result.svg)
    if result.manifest is not None:
        result.manifest.save(output_dir / "manifest.json")


def _write_background_output(result: BackgroundResult, output_dir: Path) -> None:
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    for i, candidate in enumerate(result.candidates):
        candidate.save(layers_dir / f"background_candidate_{i}_raw.png")
    (layers_dir / "background.svg").write_text(result.svg)
    if result.manifest is not None:
        result.manifest.save(output_dir / "manifest.json")


def create_character(
    character_description: str,
    actions: list[str],
    identity_image: ReferenceImage | None = None,
    style_image: ReferenceImage | None = None,
    style_mode: Literal["kontext", "ip_adapter"] = "kontext",
    candidates: int = config.NUM_CANDIDATES,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> CharacterResult:
    """Generates one character across multiple action poses, keeping its
    design consistent -- either from a text description, or starting from a
    user-supplied `identity_image` (e.g. an existing character you want drawn
    doing something new) instead of generating one. `identity_image`/
    `style_image` accept a PIL Image, or a path to one.

    When given, `identity_image`/`style_image` are used directly as the
    identity anchor -- no separate generation call for the anchor itself;
    each action's pose is still a real new Kontext generation conditioned on
    that anchor.

    `style_image` conditions the same way `identity_image` does by default
    (`style_mode="kontext"`), and the two are mutually exclusive in that mode
    -- there's only one Kontext image-conditioning slot and no verified way to
    chain two content-conditioning passes yet. `style_mode="ip_adapter"` is
    experimental and would let both be supplied together (see
    ip_adapter_flux.py) but isn't wired in yet.
    """
    if identity_image is not None and style_image is not None and style_mode == "kontext":
        raise ValueError(
            "identity_image and style_image can't both be set with style_mode='kontext' -- "
            "there's only one Kontext conditioning slot in this mode. Use style_mode='ip_adapter' "
            "to combine them, or supply just one."
        )
    if style_mode == "ip_adapter":
        raise NotImplementedError(
            "style_mode='ip_adapter' is experimental and not yet wired in -- see ip_adapter_flux.py"
        )

    bundle = ModelBundle()
    try:
        if identity_image is not None:
            identity = _load_reference_image(identity_image, width, height)
        elif style_image is not None:
            identity = _load_reference_image(style_image, width, height)
        else:
            identity = stages.generate_identity_anchor(bundle, character_description, width, height, seed=seed)

        action_ids = [f"action_{_slugify(a)}" for a in actions]

        # Stage A: everything needing Kontext (one posed subject per action).
        posed_subjects: dict[str, Image.Image] = {}
        for z_index, (action_id, action_prompt) in enumerate(zip(action_ids, actions)):
            posed_seed = None if seed is None else seed + 1 + z_index * 100
            posed_subjects[action_id] = stages.generate_posed_subject(
                bundle, character_description, action_prompt, identity, width, height, seed=posed_seed
            )

        # Stage B: everything needing the restyle ControlNet (N candidates/action).
        manifest = LayerManifest(width=width, height=height)
        action_results: dict[str, ActionResult] = {}
        for z_index, action_id in enumerate(action_ids):
            posed_subject = posed_subjects[action_id]
            candidate_images: list[Image.Image] = []
            candidate_paths: list[list[str]] = []
            for i in range(candidates):
                variant_seed = None if seed is None else seed + 1 + z_index * 100 + 1 + i
                candidate_image = stages.generate_variant(
                    bundle, posed_subject, character_description, role="character", seed=variant_seed
                )
                candidate_images.append(candidate_image)
                candidate_paths.append(image_to_paths(candidate_image))

            svg_content = build_candidate_svg(candidate_paths, width, height)
            action_results[action_id] = ActionResult(
                action=actions[z_index], posed_subject=posed_subject, candidates=candidate_images, svg=svg_content
            )
            candidate_ids = [f"candidate_{i}" for i in range(len(candidate_images))]
            manifest.elements.append(
                Element(
                    id=action_id,
                    role="character",
                    z_index=z_index,
                    svg=f"layers/{action_id}.svg",
                    candidate_ids=candidate_ids,
                )
            )

        result = CharacterResult(identity_image=identity, actions=action_results, manifest=manifest)
        if output_dir is not None:
            _write_character_output(result, Path(output_dir))
        return result
    finally:
        bundle.unload_all()


def create_background(
    description: str,
    style_image: ReferenceImage | None = None,
    style_mode: Literal["kontext", "ip_adapter"] = "kontext",
    candidates: int = config.NUM_CANDIDATES,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> BackgroundResult:
    """Generates a background as N superimposed candidate line-interpretations.
    `style_image` (a PIL Image, or a path to one), when given, is used
    directly as the base (candidate 0) instead of generating one -- the
    remaining N-1 candidates are real restyle passes conditioned on it."""
    if style_mode == "ip_adapter":
        raise NotImplementedError(
            "style_mode='ip_adapter' is experimental and not yet wired in -- see ip_adapter_flux.py"
        )

    bundle = ModelBundle()
    try:
        if style_image is not None:
            base_image = _load_reference_image(style_image, width, height)
        else:
            base_image = stages.generate_rough(
                bundle, description, role="background", width=width, height=height, seed=seed
            )

        candidate_images = [base_image]
        for i in range(1, candidates):
            variant_seed = None if seed is None else seed + 1 + i
            candidate_images.append(
                stages.generate_variant(bundle, base_image, description, role="background", seed=variant_seed)
            )

        candidate_paths = [image_to_paths(img) for img in candidate_images]
        svg_content = build_candidate_svg(candidate_paths, width, height)

        manifest = LayerManifest(width=width, height=height)
        candidate_ids = [f"candidate_{i}" for i in range(len(candidate_images))]
        manifest.elements.append(
            Element(
                id="background", role="background", z_index=0, svg="layers/background.svg", candidate_ids=candidate_ids
            )
        )

        result = BackgroundResult(candidates=candidate_images, svg=svg_content, manifest=manifest)
        if output_dir is not None:
            _write_background_output(result, Path(output_dir))
        return result
    finally:
        bundle.unload_all()
