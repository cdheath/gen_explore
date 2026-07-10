"""Combines each candidate's traced paths into one SVG per element: one <g>
per candidate, distinct stroke color, superimposed so a human can toggle
candidate visibility and delete individual paths to pick the correct lines."""

from pathlib import Path

from gen_explore import config


def build_candidate_svg(candidate_paths: list[list[str]], width: int, height: int) -> str:
    groups = []
    for i, paths in enumerate(candidate_paths):
        color = config.CANDIDATE_COLORS[i % len(config.CANDIDATE_COLORS)]
        path_tags = "\n".join(f'    <path id="c{i}_p{j}" d="{d}"/>' for j, d in enumerate(paths))
        groups.append(
            f'  <g id="candidate_{i}" stroke="{color}" fill="none" stroke-width="1.5" '
            f'stroke-linecap="round" opacity="0.85">\n{path_tags}\n  </g>'
        )
    body = "\n".join(groups)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n{body}\n</svg>\n'
    )


def save_candidate_svg(candidate_paths: list[list[str]], width: int, height: int, path: str | Path) -> None:
    Path(path).write_text(build_candidate_svg(candidate_paths, width, height))
