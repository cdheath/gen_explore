"""Candidate generation using FLUX: Kontext for identity-preserving pose
generation (including the very first "no real reference yet" rough/anchor,
via a blank white canvas -- Kontext always expects an `image=`), and
FLUX.1-dev + ControlNet-Union (soft-edge) for restyling into
construction-sketch linework while preserving structure. This replaces the
SD1.5 + IP-Adapter approach entirely -- see config.py's module docstring for
why."""

import torch
from PIL import Image

from gen_explore import config
from gen_explore.models import ModelBundle


def _full_prompt(subject_prompt: str, role: str) -> str:
    suffix = config.ROLE_STYLE_SUFFIX.get(role, config.SKETCH_STYLE_SUFFIX)
    return f"{subject_prompt}, {suffix}"


def _generator(bundle: ModelBundle, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=bundle.device).manual_seed(seed)


def _blank_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def generate_rough(
    bundle: ModelBundle,
    prompt: str,
    role: str = "background",
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    seed: int | None = None,
) -> Image.Image:
    """A fresh generation with no real reference yet."""
    result = bundle.kontext(
        image=_blank_canvas(width, height),
        prompt=_full_prompt(prompt, role),
        generator=_generator(bundle, seed),
        **config.KONTEXT_PARAMS,
    )
    return result.images[0]


def generate_identity_anchor(
    bundle: ModelBundle,
    character_prompt: str,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    seed: int | None = None,
) -> Image.Image:
    """A single reference image of a character, reused as the identity anchor
    for every action pose."""
    return generate_rough(bundle, character_prompt, role="character_subject", width=width, height=height, seed=seed)


def generate_posed_subject(
    bundle: ModelBundle,
    character_prompt: str,
    action_prompt: str,
    identity_image: Image.Image,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    seed: int | None = None,
) -> Image.Image:
    """A new pose/action for the same character, conditioned directly on the
    identity anchor via Kontext's image-editing mechanism."""
    result = bundle.kontext(
        image=identity_image,
        prompt=_full_prompt(f"the same character, now {action_prompt}", "character_subject"),
        generator=_generator(bundle, seed),
        **config.KONTEXT_PARAMS,
    )
    return result.images[0]


def generate_variant(
    bundle: ModelBundle,
    reference_image: Image.Image,
    prompt: str,
    role: str = "background",
    seed: int | None = None,
) -> Image.Image:
    width, height = reference_image.size
    control_image = bundle.hed_detector(
        reference_image, detect_resolution=max(width, height), image_resolution=max(width, height)
    ).resize((width, height))
    restyle_params = config.ROLE_RESTYLE_PARAMS.get(role, config.ROLE_RESTYLE_PARAMS["background"])
    result = bundle.controlnet_pipe(
        prompt=_full_prompt(prompt, role),
        control_image=control_image,
        width=width,
        height=height,
        generator=_generator(bundle, seed),
        **restyle_params,
    )
    return result.images[0]
