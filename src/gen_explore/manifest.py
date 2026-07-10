"""The layer manifest is the source of truth generation writes and every
exporter (PSD now, a future web canvas later) reads from — none of them
should need to know how the images were produced."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Transform:
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0


@dataclass
class Element:
    id: str
    role: str  # "background" | "character" | "fx"
    z_index: int
    svg: str  # path (relative to the manifest's directory) to the multi-candidate SVG
    candidate_ids: list[str]
    transform: Transform = field(default_factory=Transform)


@dataclass
class LayerManifest:
    width: int
    height: int
    elements: list[Element] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        data = {
            "canvas": {"width": self.width, "height": self.height},
            "elements": [
                {
                    "id": e.id,
                    "role": e.role,
                    "z_index": e.z_index,
                    "transform": asdict(e.transform),
                    "svg": e.svg,
                    "candidate_ids": e.candidate_ids,
                }
                for e in self.elements
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LayerManifest":
        data = json.loads(Path(path).read_text())
        elements = [
            Element(
                id=e["id"],
                role=e["role"],
                z_index=e["z_index"],
                svg=e["svg"],
                candidate_ids=e["candidate_ids"],
                transform=Transform(**e["transform"]),
            )
            for e in data["elements"]
        ]
        return cls(width=data["canvas"]["width"], height=data["canvas"]["height"], elements=elements)
