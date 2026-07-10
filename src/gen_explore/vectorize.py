"""Converts a grayscale ink-on-white raster sketch into a list of SVG path
strings by tracing skeleton centerlines, rather than filled-contour outlines
(potrace/vtracer-style), so each stroke becomes its own separately editable
path -- including through crossings, which are joined as continuations rather
than left as four dangling stubs.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize

MIN_OBJECT_SIZE = 20
SPUR_LENGTH = 10
FILL_AREA_PERIMETER_RATIO = 6.0
SIMPLIFY_EPSILON = 1.2
DIRECTION_WINDOW = 5
# Long traced strokes get split into pieces of at most this many raw
# (pre-simplification) points, so each becomes its own independent SVG path
# -- selectable and deletable with a single click in a vector editor, rather
# than requiring node-range selection within one long path.
CHUNK_LENGTH = 60

_STRUCT8 = np.ones((3, 3), dtype=int)
_NEIGHBOR_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _threshold_ink(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.float32)
    ink = 255.0 - gray
    blurred = cv2.GaussianBlur(ink, (0, 0), sigmaX=0.8)
    thresh = threshold_otsu(blurred)
    mask = blurred > thresh
    return remove_small_objects(mask, min_size=MIN_OBJECT_SIZE)


def _split_fill_regions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separates thin linework from shading/fill blobs by area/perimeter
    ratio. Returns (line_mask, fill_mask): fill regions are excluded from
    centerline tracing (skeletonizing a solid shaded blob just produces
    clutter) but their outline is kept separately via `_fill_region_paths` so
    a heavily-shaded sketch (e.g. cross-hatched form shading) doesn't lose its
    silhouette entirely -- only the internal texture."""
    labeled, num = ndimage.label(mask, structure=_STRUCT8)
    line_mask = np.zeros_like(mask)
    fill_mask = np.zeros_like(mask)
    for label_id, slc in enumerate(ndimage.find_objects(labeled), start=1):
        if slc is None:
            continue
        component = labeled[slc] == label_id
        area = int(component.sum())
        eroded = ndimage.binary_erosion(component)
        perimeter = max(int((component & ~eroded).sum()), 1)
        target = fill_mask if area / perimeter > FILL_AREA_PERIMETER_RATIO else line_mask
        target[slc] |= component
    return line_mask, fill_mask


def _chunk_points(points: list, max_length: int) -> list[list]:
    """Splits a point sequence into pieces of at most `max_length` points.
    Consecutive chunks share their boundary point so there's no visual gap
    between them once each is drawn as its own path."""
    if len(points) <= max_length:
        return [points]
    chunks = []
    step = max_length - 1
    for start in range(0, len(points) - 1, step):
        chunk = points[start : start + max_length]
        if len(chunk) >= 2:
            chunks.append(chunk)
    return chunks


def _fill_region_paths(fill_mask: np.ndarray) -> list[str]:
    """Traces the outer silhouette of shading/fill blobs dropped from
    centerline tracing, so their overall shape is preserved even though their
    internal texture is not -- chunked the same as regular strokes."""
    contours, _ = cv2.findContours(fill_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = [tuple(p[0]) for p in contour]
        points.append(points[0])  # close the loop's seam before chunking
        for chunk in _chunk_points(points, CHUNK_LENGTH):
            simplified = cv2.approxPolyDP(
                np.array(chunk, dtype=np.float32).reshape(-1, 1, 2), epsilon=SIMPLIFY_EPSILON, closed=False
            ).reshape(-1, 2)
            if len(simplified) < 2:
                continue
            commands = [f"M{simplified[0][0]:.1f},{simplified[0][1]:.1f}"]
            commands += [f"L{px:.1f},{py:.1f}" for px, py in simplified[1:]]
            paths.append(" ".join(commands))
    return paths


def _neighbor_coords(y: int, x: int, skeleton: np.ndarray) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    coords = []
    for dy, dx in _NEIGHBOR_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
            coords.append((ny, nx))
    return coords


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    counts = ndimage.convolve(skeleton.astype(np.uint8), _STRUCT8, mode="constant")
    return (counts - skeleton.astype(np.uint8)) * skeleton


def _prune_spurs(skeleton: np.ndarray, iterations: int = 3) -> np.ndarray:
    skeleton = skeleton.copy()
    for _ in range(iterations):
        counts = _neighbor_count(skeleton)
        endpoints = [tuple(p) for p in np.argwhere(counts == 1)]
        removed_any = False
        for start in endpoints:
            if not skeleton[start]:
                continue
            chain = [start]
            prev, cur = None, start
            reached_junction = False
            while True:
                neighbors = [n for n in _neighbor_coords(*cur, skeleton) if n != prev]
                if len(neighbors) != 1:
                    reached_junction = len(neighbors) > 1
                    break
                nxt = neighbors[0]
                if counts[nxt] >= 3:
                    reached_junction = True
                    break
                chain.append(nxt)
                prev, cur = cur, nxt
                if len(chain) > SPUR_LENGTH:
                    break
            if reached_junction and len(chain) <= SPUR_LENGTH:
                for p in chain:
                    skeleton[p] = False
                removed_any = True
        if not removed_any:
            break
    return skeleton


def _order_arc(pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Orders an arc's unordered pixel set into a walked sequence. Arcs are
    open chains (<=2 neighbors per pixel within the arc) by construction,
    except for isolated closed loops with no free end."""
    degree = {}
    for p in pixels:
        y, x = p
        degree[p] = sum(1 for dy, dx in _NEIGHBOR_OFFSETS if (y + dy, x + dx) in pixels)

    ends = [p for p, d in degree.items() if d <= 1]
    start = ends[0] if ends else next(iter(pixels))

    ordered = [start]
    visited = {start}
    prev, cur = None, start
    while True:
        neighbors = [
            (cur[0] + dy, cur[1] + dx)
            for dy, dx in _NEIGHBOR_OFFSETS
            if (cur[0] + dy, cur[1] + dx) in pixels and (cur[0] + dy, cur[1] + dx) != prev
        ]
        neighbors = [n for n in neighbors if n not in visited]
        if not neighbors:
            break
        nxt = neighbors[0]
        ordered.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    return ordered


def _direction_at_end(arc: list[tuple[int, int]], at_start: bool) -> np.ndarray:
    window = arc[: DIRECTION_WINDOW] if at_start else arc[-DIRECTION_WINDOW:][::-1]
    tip = np.array(window[0], dtype=np.float32)
    tail = np.array(window[-1], dtype=np.float32)
    vec = tip - tail
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


def _labeled_pixel_lists(labeled: np.ndarray, num_labels: int) -> list[list[tuple[int, int]]]:
    """Pixel coordinates per label, 1-indexed (index 0 unused). Uses per-label
    bounding-box slices (`ndimage.find_objects`) so this is O(pixels), not
    O(pixels * num_labels) -- calling `np.argwhere(labeled == i)` in a loop
    rescans the whole array per label, which is catastrophic once there are
    thousands of labels (a busy, heavily-hatched sketch easily produces that
    many arcs/junction clusters)."""
    result: list[list[tuple[int, int]]] = [[] for _ in range(num_labels + 1)]
    for label_id, slc in enumerate(ndimage.find_objects(labeled), start=1):
        if slc is None:
            continue
        local_coords = np.argwhere(labeled[slc] == label_id)
        offset = np.array([s.start for s in slc])
        result[label_id] = [tuple(c) for c in (local_coords + offset)]
    return result


def _trace_paths(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    counts = _neighbor_count(skeleton)
    junction_mask = counts >= 3
    arc_only = skeleton & ~junction_mask

    labeled_arcs, num_arcs = ndimage.label(arc_only, structure=_STRUCT8)
    arc_pixel_lists = _labeled_pixel_lists(labeled_arcs, num_arcs)
    arc_pixel_sets: list[set | None] = [None] + [set(arc_pixel_lists[i]) for i in range(1, num_arcs + 1)]
    ordered_arcs: list[list | None] = [None] + [_order_arc(arc_pixel_sets[i]) for i in range(1, num_arcs + 1)]

    arc_id_at = {}
    for i in range(1, num_arcs + 1):
        for pixel in arc_pixel_sets[i]:
            arc_id_at[pixel] = i

    # connection[(arc_id, 'start'|'end')] = (other_arc_id, other_end, junction_yx)
    connection: dict[tuple[int, str], tuple[int, str, tuple[int, int]]] = {}

    # A crossing skeletonizes to a small *cluster* of junction pixels, not one
    # point -- treat each connected cluster as a single logical junction node.
    labeled_junctions, num_junctions = ndimage.label(junction_mask, structure=_STRUCT8)
    junction_pixel_lists = _labeled_pixel_lists(labeled_junctions, num_junctions)

    for cluster_id in range(1, num_junctions + 1):
        cluster_pixels = junction_pixel_lists[cluster_id]
        centroid = (
            int(round(np.mean([p[0] for p in cluster_pixels]))),
            int(round(np.mean([p[1] for p in cluster_pixels]))),
        )
        cluster_set = set(cluster_pixels)
        touches = []  # (arc_id, end_name, direction_vector)
        seen_arc_ends: set[tuple[int, str]] = set()
        for jy, jx in cluster_pixels:
            for ny, nx in _neighbor_coords(jy, jx, skeleton):
                if (ny, nx) in cluster_set:
                    continue
                aid = arc_id_at.get((ny, nx))
                if aid is None:
                    continue
                arc = ordered_arcs[aid]
                if arc[0] == (ny, nx) and (aid, "start") not in seen_arc_ends:
                    seen_arc_ends.add((aid, "start"))
                    touches.append((aid, "start", _direction_at_end(arc, True)))
                elif arc[-1] == (ny, nx) and (aid, "end") not in seen_arc_ends:
                    seen_arc_ends.add((aid, "end"))
                    touches.append((aid, "end", _direction_at_end(arc, False)))
        jy, jx = centroid
        remaining = touches[:]
        while len(remaining) >= 2:
            best_pair, best_score = None, None
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    score = float(np.dot(remaining[i][2], remaining[j][2]))
                    if best_score is None or score < best_score:
                        best_score, best_pair = score, (i, j)
            i, j = best_pair
            a_id, a_end, _ = remaining[i]
            b_id, b_end, _ = remaining[j]
            connection[(a_id, a_end)] = (b_id, b_end, (jy, jx))
            connection[(b_id, b_end)] = (a_id, a_end, (jy, jx))
            remaining = [t for k, t in enumerate(remaining) if k not in (i, j)]

    visited_arcs = set()
    final_paths: list[list[tuple[int, int]]] = []

    def walk_from(start_arc: int, start_end: str) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        cur_arc, cur_end = start_arc, start_end
        # `connection` links are never consumed, so a chain of arcs that closes
        # into a loop purely through junction crossings (common in hatching)
        # would otherwise revisit the same links forever. Stop as soon as we're
        # about to re-enter a (arc, end) state already seen in this walk.
        visited_this_walk: set[tuple[int, str]] = set()
        while (cur_arc, cur_end) not in visited_this_walk:
            visited_this_walk.add((cur_arc, cur_end))
            arc = ordered_arcs[cur_arc]
            ordered_pixels = arc if cur_end == "start" else list(reversed(arc))
            points.extend(ordered_pixels[1:] if points else ordered_pixels)
            visited_arcs.add(cur_arc)
            other_end = "end" if cur_end == "start" else "start"
            key = (cur_arc, other_end)
            if key in connection:
                next_arc, next_end, junction_yx = connection[key]
                points.append(junction_yx)
                cur_arc, cur_end = next_arc, next_end
            else:
                break
        return points

    for i in range(1, num_arcs + 1):
        if i in visited_arcs:
            continue
        free_ends = [end for end in ("start", "end") if (i, end) not in connection]
        if not free_ends:
            continue  # part of a closed loop entirely built of connected arcs; handled below
        final_paths.append(walk_from(i, free_ends[0]))

    # Any arcs left unvisited at this point are pure cycles (every end connected) -- walk them starting anywhere.
    for i in range(1, num_arcs + 1):
        if i in visited_arcs:
            continue
        final_paths.append(walk_from(i, "start"))

    return final_paths


def _simplify_to_svg_path(points: list[tuple[int, int]]) -> str | None:
    if len(points) < 2:
        return None
    contour = np.array([[x, y] for y, x in points], dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon=SIMPLIFY_EPSILON, closed=False).reshape(-1, 2)
    if len(simplified) < 2:
        return None
    commands = [f"M{simplified[0][0]:.1f},{simplified[0][1]:.1f}"]
    commands += [f"L{px:.1f},{py:.1f}" for px, py in simplified[1:]]
    return " ".join(commands)


def image_to_paths(image: Image.Image) -> list[str]:
    """Traces a grayscale ink-on-white sketch into a list of SVG path `d`
    strings, one per distinct stroke (crossings become through-paths, not
    four dangling stubs)."""
    mask = _threshold_ink(image)
    line_mask, fill_mask = _split_fill_regions(mask)
    skeleton = skeletonize(line_mask)
    skeleton = _prune_spurs(skeleton)
    chains = _trace_paths(skeleton)
    chunks = [chunk for chain in chains for chunk in _chunk_points(chain, CHUNK_LENGTH)]
    paths = [_simplify_to_svg_path(chunk) for chunk in chunks]
    paths = [p for p in paths if p]
    paths += _fill_region_paths(fill_mask)
    return paths
