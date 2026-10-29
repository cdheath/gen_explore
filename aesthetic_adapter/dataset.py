"""Training examples for the Kontext-stage adapter: reuses the same 2,673-image
bootstrap corpus and caption cache already built for the SD3.5 work
(embedding_research/aesthetic_adapter_probe/), adding the Explore vector that corpus
never needed before (aesthetic_adapter_probe only ever computed Aesthetic vectors).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from aesthetic_encoder import AestheticEncoder
from explore_encoder import ExploreEncoder

EMBEDDING_RESEARCH = Path(__file__).resolve().parents[2] / "embedding_research"
sys.path.insert(0, str(EMBEDDING_RESEARCH))
from aesthetic_adapter_probe.dataset import image_paths_from_npz  # noqa: E402

# NOTE: aesthetic_adapter_probe's own DEFAULT_NPZ/DEFAULT_CHECKPOINT are bare relative
# strings (meant to be run from embedding_research's own cwd) -- deliberately not
# reused directly here, since this package runs from gen_explore's cwd instead.
DEFAULT_NPZ = str(EMBEDDING_RESEARCH / "projected_embeddings.npz")
DEFAULT_CHECKPOINT = str(EMBEDDING_RESEARCH / "checkpoints" / "vjepa_aesthetic" / "best.pt")
DEFAULT_CAPTION_CACHE = str(EMBEDDING_RESEARCH / "aesthetic_adapter_probe" / "captions.jsonl")
DEFAULT_EXPLORE_CHECKPOINT = str(EMBEDDING_RESEARCH / "checkpoints" / "explore" / "full_v2" / "best.pt")
DEFAULT_EXPLORE_CONCEPTS = str(EMBEDDING_RESEARCH / "checkpoints" / "explore" / "full_v2" / "explore_concepts.npz")
# Precomputed from a 300-image live-encoded sample of this same corpus (see
# embedding_research/explore_encoder/encoder.py's center() docstring) -- confirmed this
# session that Explore's raw vectors share the same dominant-corpus-direction pathology
# Aesthetic's did (a 40-image check found global mean norm ~0.93; the full 300-image
# estimate used here is ~0.50 -- the smaller sample was noisy, not wrong in direction).
DEFAULT_EXPLORE_GLOBAL_MEAN = str(EMBEDDING_RESEARCH / "checkpoints" / "explore" / "full_v2" / "global_mean.npy")


@dataclass
class Example:
    path: Path
    caption: str
    aesthetic_vector: np.ndarray  # (256,) float32, centered unit-norm
    explore_vector: np.ndarray    # (256,) float32, centered unit-norm


def build_examples(
    paths: list[Path],
    captions: dict[str, str],
    aesthetic_encoder: AestheticEncoder,
    explore_encoder: ExploreEncoder,
) -> list[Example]:
    """Skips any path missing a caption or that fails to load/encode."""
    examples = []
    for p in paths:
        caption = captions.get(str(p))
        if not caption or not p.exists():
            continue
        try:
            aes_result = aesthetic_encoder.encode(p, "image")
            aes_vector = aesthetic_encoder.center(np.asarray(aes_result["vector"], dtype=np.float32))
            exp_result = explore_encoder.encode(p, "image")
            exp_vector = explore_encoder.center(np.asarray(exp_result["vector"], dtype=np.float32))
        except Exception:
            continue
        examples.append(Example(path=p, caption=caption, aesthetic_vector=aes_vector, explore_vector=exp_vector))
    return examples


class KontextAdapterDataset(Dataset):
    """Yields (pixel_values in [-1, 1], caption, aesthetic_vector, explore_vector)."""

    def __init__(self, examples: list[Example], resolution: int = 512):
        self.examples = examples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        image = Image.open(ex.path).convert("RGB").resize(
            (self.resolution, self.resolution), Image.BICUBIC,
        )
        pixel_values = (np.asarray(image, dtype=np.float32) / 127.5) - 1.0  # [-1, 1]
        pixel_values = pixel_values.transpose(2, 0, 1)  # HWC -> CHW
        return pixel_values, ex.caption, ex.aesthetic_vector, ex.explore_vector
