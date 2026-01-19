import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


POINTS_HEADER_1 = "# 3D point list with one line of data per point:\n"
POINTS_HEADER_2 = "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
IMAGES_HEADER_1 = "# Image list with two lines of data per image:\n"
IMAGES_HEADER_2 = "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
IMAGES_HEADER_3 = "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
CAMERAS_HEADER_1 = "# Camera list with one line of data per camera:\n"
CAMERAS_HEADER_2 = "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"


@dataclass(frozen=True)
class BlockSpec:
    name: str
    bounds: Tuple[float, float, float, float, float, float]  # xmin, xmax, ymin, ymax, zmin, zmax


@dataclass
class BlockState:
    spec: BlockSpec
    out_dir: str
    points_file_path: str
    points_header_offset: int
    point_count: int
    image_point_counts: Dict[int, int]
    point_ids: Optional[Set[int]]
    selected_images: Set[int]
    cameras_used: Set[int]


def _normalize_bounds(bounds: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    if len(bounds) != 6:
        raise ValueError(f"Bounds must have 6 values, got {len(bounds)}")
    xmin, ymin, zmin, xmax, ymax, zmax = map(float, bounds)
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if zmin > zmax:
        zmin, zmax = zmax, zmin
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def _point_in_bounds(x: float, y: float, z: float, bounds: Tuple[float, float, float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax)


def _load_blocks(path: str) -> List[BlockSpec]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".js"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "blocks" in data:
            data = data["blocks"]
        blocks: List[BlockSpec] = []
        if not isinstance(data, list):
            raise ValueError("JSON blocks file must be a list or contain a 'blocks' list")
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                name = item.get("name") or f"block_{idx}"
                bounds = item.get("bounds") or item.get("bbox") or item.get("box")
            else:
                name = f"block_{idx}"
                bounds = item
            if bounds is None:
                raise ValueError(f"Block {idx} missing bounds")
            blocks.append(BlockSpec(name=name, bounds=_normalize_bounds(bounds)))
        return blocks

    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 6:
                name = f"block_{idx}"
                bounds = parts
            elif len(parts) == 7:
                name = parts[0]
                bounds = parts[1:]
            else:
                raise ValueError(f"Invalid bounds line: {raw.strip()}")
            blocks.append(BlockSpec(name=name, bounds=_normalize_bounds(bounds)))
    return blocks


def _write_points_header(f, count: int, width: int = 12) -> int:
    f.write(POINTS_HEADER_1)
    f.write(POINTS_HEADER_2)
    count_line = f"# Number of points: {count:{width}d}\n"
    f.write(count_line)
    return len(POINTS_HEADER_1.encode("utf-8")) + len(POINTS_HEADER_2.encode("utf-8"))


def _write_images_header(f, count: int) -> None:
    f.write(IMAGES_HEADER_1)
    f.write(IMAGES_HEADER_2)
    f.write(IMAGES_HEADER_3)
    f.write(f"# Number of images: {count}\n")


def _write_cameras_header(f, count: int) -> None:
    f.write(CAMERAS_HEADER_1)
    f.write(CAMERAS_HEADER_2)
    f.write(f"# Number of cameras: {count}\n")


def _update_points_header(points_file_path: str, offset: int, count: int, width: int = 12) -> None:
    line = f"# Number of points: {count:{width}d}\n"
    with open(points_file_path, "r+b") as f:
        f.seek(offset)
        f.write(line.encode("utf-8"))


def _iter_points3d_lines(points_path: str) -> Iterable[str]:
    with open(points_path, "r", encoding="utf-8", newline="\n") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            stripped = line.strip()
            if not stripped:
                continue
            yield line


def _iter_images_pairs(images_path: str) -> Iterable[Tuple[str, str]]:
    with open(images_path, "r", encoding="utf-8", newline="\n") as f:
        while True:
            line1 = f.readline()
            if not line1:
                break
            if not line1.strip() or line1.startswith("#"):
                continue
            line2 = f.readline()
            if line2 is None:
                line2 = ""
            yield line1, line2


def _filter_points2d_line(line: str, allowed_point_ids: Set[int]) -> str:
    tokens = line.split()
    if not tokens:
        return "\n"
    # tokens: x y point3d_id repeated
    for idx in range(2, len(tokens), 3):
        try:
            pid = int(tokens[idx])
        except ValueError:
            continue
        if pid != -1 and pid not in allowed_point_ids:
            tokens[idx] = "-1"
    return " ".join(tokens) + "\n"


def partition_colmap(
    colmap_dir: str,
    blocks_path: str,
    output_dir: str,
    min_points_per_image: int = 1,
    filter_points2d: bool = False,
    progress_every: int = 1_000_000,
) -> None:
    cameras_path = os.path.join(colmap_dir, "Cameras.txt")
    images_path = os.path.join(colmap_dir, "Images.txt")
    points_path = os.path.join(colmap_dir, "Points3D.txt")

    blocks = _load_blocks(blocks_path)
    if not blocks:
        raise ValueError("No blocks loaded")

    os.makedirs(output_dir, exist_ok=True)

    states: List[BlockState] = []
    points_files = []
    for block in blocks:
        out_dir = os.path.join(output_dir, block.name)
        os.makedirs(out_dir, exist_ok=True)
        points_file_path = os.path.join(out_dir, "Points3D.txt")
        f = open(points_file_path, "w", encoding="utf-8", newline="\n")
        header_offset = _write_points_header(f, 0)
        points_files.append(f)
        states.append(
            BlockState(
                spec=block,
                out_dir=out_dir,
                points_file_path=points_file_path,
                points_header_offset=header_offset,
                point_count=0,
                image_point_counts=defaultdict(int),
                point_ids=set() if filter_points2d else None,
                selected_images=set(),
                cameras_used=set(),
            )
        )

    processed = 0
    for line in _iter_points3d_lines(points_path):
        tokens = line.split()
        if len(tokens) < 8:
            continue
        try:
            point_id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
        except ValueError:
            continue

        matched_indices: List[int] = []
        for idx, state in enumerate(states):
            if _point_in_bounds(x, y, z, state.spec.bounds):
                matched_indices.append(idx)

        if not matched_indices:
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(f"[points] processed={processed}")
            continue

        track_tokens = tokens[8:]
        track_image_ids: List[int] = []
        if track_tokens:
            for i in range(0, len(track_tokens), 2):
                try:
                    track_image_ids.append(int(track_tokens[i]))
                except ValueError:
                    continue

        for idx in matched_indices:
            states[idx].point_count += 1
            points_files[idx].write(line)
            if states[idx].point_ids is not None:
                states[idx].point_ids.add(point_id)
            if track_image_ids:
                img_counts = states[idx].image_point_counts
                for image_id in track_image_ids:
                    img_counts[image_id] += 1

        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"[points] processed={processed}")

    for f in points_files:
        f.close()

    for state in states:
        _update_points_header(state.points_file_path, state.points_header_offset, state.point_count)

    image_to_blocks: Dict[int, List[int]] = defaultdict(list)
    for idx, state in enumerate(states):
        for image_id, count in state.image_point_counts.items():
            if count >= min_points_per_image:
                state.selected_images.add(image_id)
                image_to_blocks[image_id].append(idx)

    images_files = []
    for state in states:
        images_file_path = os.path.join(state.out_dir, "Images.txt")
        f = open(images_file_path, "w", encoding="utf-8", newline="\n")
        _write_images_header(f, len(state.selected_images))
        images_files.append(f)

    for line1, line2 in _iter_images_pairs(images_path):
        tokens = line1.split()
        if len(tokens) < 9:
            continue
        try:
            image_id = int(tokens[0])
            camera_id = int(tokens[8])
        except ValueError:
            continue
        block_indices = image_to_blocks.get(image_id)
        if not block_indices:
            continue
        for idx in block_indices:
            state = states[idx]
            state.cameras_used.add(camera_id)
            images_files[idx].write(line1)
            if filter_points2d and state.point_ids is not None:
                images_files[idx].write(_filter_points2d_line(line2, state.point_ids))
            else:
                images_files[idx].write(line2)

    for f in images_files:
        f.close()

    camera_to_blocks: Dict[int, List[int]] = defaultdict(list)
    for idx, state in enumerate(states):
        for cam_id in state.cameras_used:
            camera_to_blocks[cam_id].append(idx)

    cameras_files = []
    for state in states:
        cameras_file_path = os.path.join(state.out_dir, "Cameras.txt")
        f = open(cameras_file_path, "w", encoding="utf-8", newline="\n")
        _write_cameras_header(f, len(state.cameras_used))
        cameras_files.append(f)

    with open(cameras_path, "r", encoding="utf-8", newline="\n") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            tokens = line.split()
            if not tokens:
                continue
            try:
                cam_id = int(tokens[0])
            except ValueError:
                continue
            block_indices = camera_to_blocks.get(cam_id)
            if not block_indices:
                continue
            for idx in block_indices:
                cameras_files[idx].write(line)

    for f in cameras_files:
        f.close()

    for state in states:
        with open(os.path.join(state.out_dir, "colmap.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("The files Cameras.txt, Images.txt and Points3D.txt are in the same folder.\n")

    print(f"Done. Blocks written: {len(states)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split COLMAP text model into multiple blocks.")
    parser.add_argument("--colmap-dir", required=True, help="Input COLMAP folder containing Cameras.txt/Images.txt/Points3D.txt")
    parser.add_argument("--blocks", required=True, help="Blocks definition file (.json or .txt)")
    parser.add_argument("--output-dir", required=True, help="Output root folder for blocks")
    parser.add_argument("--min-points-per-image", type=int, default=1, help="Min points in block to keep an image")
    parser.add_argument("--filter-points2d", action="store_true", help="Filter Points2D to only keep points in block (uses extra memory)")
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Print progress every N points (0 to disable)")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    partition_colmap(
        colmap_dir=args.colmap_dir,
        blocks_path=args.blocks,
        output_dir=args.output_dir,
        min_points_per_image=args.min_points_per_image,
        filter_points2d=args.filter_points2d,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
