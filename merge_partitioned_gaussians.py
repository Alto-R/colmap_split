import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np


def _default_path(*parts):
    return Path(__file__).resolve().parent.joinpath(*parts)


# ================= Configuration =================
config = {
    # Partition JSONs (part_*.json) from partition_colmap.py
    "parts_dir": _default_path("split_output"),

    # Root directory that contains part_0, part_1, ... (each trained separately)
    # Each part folder should contain a Gaussian point cloud PLY file.
    # Example (your structure):
    # split_colmap/part_0/output/*.ply
    # split_colmap/part_1/output/*.ply
    "gaussian_root": _default_path("split_colmap"),

    # Relative path to the PLY file inside each part folder.
    # Example for 3DGS: point_cloud/iteration_30000/point_cloud.ply
    # Supports glob patterns like "output/*.ply"
    "gaussian_relpath": os.path.join("output", "*.ply"),

    # Fallback candidates if gaussian_relpath is not found (checked in order)
    "gaussian_candidates": [
        os.path.join("output", "*.ply"),
        os.path.join("point_cloud", "iteration_30000", "point_cloud.ply"),
        os.path.join("point_cloud", "point_cloud.ply"),
        "point_cloud.ply",
    ],

    # Output merged PLY
    "output_ply": _default_path("merged_gaussians", "point_cloud.ply"),

    # Merge mode:
    # - "strict": keep only points inside strict bbox of each part
    # - "overlap": keep points inside overlap bbox (will duplicate overlaps)
    "mode": "strict",

    # If True (strict mode only), keep points that are outside all strict bboxes
    # but still inside this part's overlap bbox. This avoids holes if training drifted.
    "keep_orphans": True,

    # Chunk size for streaming (number of vertices per read)
    "chunk_size": 1_000_000,

    # Overwrite output if exists
    "overwrite": True,

    # Only print statistics without writing output
    "dry_run": False,
}
# ===============================================


PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _parse_ply_header(ply_path: Path):
    header_lines = []
    comments = []
    properties = []
    vertex_count = None
    ply_format = None
    in_vertex = False

    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {ply_path}")
            header_lines.append(line)
            line_str = line.decode("ascii", errors="ignore").strip()
            if line_str.startswith("format"):
                ply_format = line_str.split()[1]
            elif line_str.startswith("comment") or line_str.startswith("obj_info"):
                comments.append(line_str)
            elif line_str.startswith("element"):
                parts = line_str.split()
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False
            elif line_str.startswith("property") and in_vertex:
                parts = line_str.split()
                if len(parts) >= 3 and parts[1] != "list":
                    ptype = parts[1]
                    pname = parts[2]
                    properties.append((pname, ptype))
            elif line_str == "end_header":
                data_start = f.tell()
                break

    if ply_format != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {ply_format} ({ply_path})")
    if vertex_count is None or not properties:
        raise ValueError(f"Missing vertex element or properties: {ply_path}")

    dtype_fields = []
    for pname, ptype in properties:
        if ptype not in PLY_TYPE_TO_DTYPE:
            raise ValueError(f"Unsupported PLY property type '{ptype}' in {ply_path}")
        dtype_fields.append((pname, "<" + PLY_TYPE_TO_DTYPE[ptype]))
    dtype = np.dtype(dtype_fields)

    return {
        "vertex_count": vertex_count,
        "properties": properties,
        "dtype": dtype,
        "comments": comments,
        "data_start": data_start,
    }


def _build_ply_header(properties: List[Tuple[str, str]], vertex_count: int, comments: List[str]):
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments:
        lines.append(c)
    lines.append(f"element vertex {vertex_count}")
    for pname, ptype in properties:
        lines.append(f"property {ptype} {pname}")
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _load_parts(parts_dir: Path):
    parts = []
    for p in sorted(parts_dir.glob("part_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts.append((p.stem, data))
    return parts


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in ["*", "?", "["])


def _resolve_gaussian_path(part_name: str, root: Path, relpath: str, candidates: List[str]):
    part_dir = root / part_name
    if relpath:
        pattern = relpath.replace("\\", "/")
        if _has_glob(pattern):
            matches = sorted(part_dir.glob(pattern))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Multiple PLY matches for {part_name}: {matches}")
        else:
            p = part_dir / relpath
            if p.exists():
                return p
    for rel in candidates:
        pattern = rel.replace("\\", "/")
        if _has_glob(pattern):
            matches = sorted(part_dir.glob(pattern))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Multiple PLY matches for {part_name}: {matches}")
        else:
            p = part_dir / rel
            if p.exists():
                return p
    return None


def _in_bbox_mask(x, y, z, bmin, bmax):
    return (
        (x >= bmin[0]) & (x <= bmax[0]) &
        (y >= bmin[1]) & (y <= bmax[1]) &
        (z >= bmin[2]) & (z <= bmax[2])
    )


def _in_any_bbox_mask(x, y, z, bboxes: List[Tuple[np.ndarray, np.ndarray]]):
    if not bboxes:
        return np.zeros_like(x, dtype=bool)
    mask = np.zeros_like(x, dtype=bool)
    for bmin, bmax in bboxes:
        mask |= _in_bbox_mask(x, y, z, bmin, bmax)
    return mask


def _iter_ply_chunks(ply_path: Path, dtype: np.dtype, vertex_count: int, data_start: int, chunk_size: int):
    with open(ply_path, "rb") as f:
        f.seek(data_start)
        remaining = vertex_count
        while remaining > 0:
            count = min(chunk_size, remaining)
            chunk = np.fromfile(f, dtype=dtype, count=count)
            if chunk.size == 0:
                break
            yield chunk
            remaining -= chunk.size


def main():
    parts_dir = Path(config["parts_dir"])
    gaussian_root = Path(config["gaussian_root"])
    output_ply = Path(config["output_ply"])

    if not parts_dir.exists():
        raise FileNotFoundError(f"parts_dir not found: {parts_dir}")
    if not gaussian_root.exists():
        raise FileNotFoundError(f"gaussian_root not found: {gaussian_root}")

    parts = _load_parts(parts_dir)
    if not parts:
        raise ValueError("No part_*.json found.")

    strict_bboxes = []
    overlap_bboxes = []
    for _, pdata in parts:
        strict_bboxes.append((np.array(pdata["strict_min"]), np.array(pdata["strict_max"])))
        overlap_bboxes.append((np.array(pdata["overlap_min"]), np.array(pdata["overlap_max"])))

    mode = config["mode"]
    keep_orphans = bool(config.get("keep_orphans", False))
    chunk_size = int(config["chunk_size"])

    header_ref = None
    per_part_counts: Dict[str, int] = {}
    per_part_total: Dict[str, int] = {}

    print(f"[-] parts: {len(parts)}")
    print(f"[-] mode: {mode}, keep_orphans={keep_orphans}")

    # First pass: count kept vertices
    total_keep = 0
    for idx, (part_name, pdata) in enumerate(parts):
        ply_path = _resolve_gaussian_path(
            part_name,
            gaussian_root,
            config["gaussian_relpath"],
            config["gaussian_candidates"],
        )
        if ply_path is None:
            raise FileNotFoundError(f"PLY not found for {part_name} under {gaussian_root}")

        header = _parse_ply_header(ply_path)
        if header_ref is None:
            header_ref = header
        else:
            if header["properties"] != header_ref["properties"]:
                raise ValueError(f"PLY property mismatch in {ply_path}")

        strict_min = np.array(pdata["strict_min"])
        strict_max = np.array(pdata["strict_max"])
        overlap_min = np.array(pdata["overlap_min"])
        overlap_max = np.array(pdata["overlap_max"])

        kept = 0
        for chunk in _iter_ply_chunks(
            ply_path,
            header["dtype"],
            header["vertex_count"],
            header["data_start"],
            chunk_size,
        ):
            x = chunk["x"]
            y = chunk["y"]
            z = chunk["z"]

            if mode == "overlap":
                mask = _in_bbox_mask(x, y, z, overlap_min, overlap_max)
            else:
                mask = _in_bbox_mask(x, y, z, strict_min, strict_max)
                if keep_orphans:
                    in_any_strict = _in_any_bbox_mask(x, y, z, strict_bboxes)
                    in_overlap = _in_bbox_mask(x, y, z, overlap_min, overlap_max)
                    mask |= (~in_any_strict) & in_overlap

            kept += int(np.sum(mask))

        per_part_counts[part_name] = kept
        per_part_total[part_name] = header["vertex_count"]
        total_keep += kept
        print(f"[count] {part_name}: keep={kept} / total={header['vertex_count']}")

    print(f"[-] total_keep={total_keep}")
    if config["dry_run"]:
        print("[-] dry_run=True, skipping write.")
        return

    if output_ply.exists() and not config["overwrite"]:
        raise FileExistsError(f"Output exists: {output_ply}")
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    header_text = _build_ply_header(header_ref["properties"], total_keep, header_ref["comments"])
    with open(output_ply, "wb") as out:
        out.write(header_text.encode("ascii"))

        # Second pass: write filtered vertices
        for idx, (part_name, pdata) in enumerate(parts):
            ply_path = _resolve_gaussian_path(
                part_name,
                gaussian_root,
                config["gaussian_relpath"],
                config["gaussian_candidates"],
            )
            header = _parse_ply_header(ply_path)
            strict_min = np.array(pdata["strict_min"])
            strict_max = np.array(pdata["strict_max"])
            overlap_min = np.array(pdata["overlap_min"])
            overlap_max = np.array(pdata["overlap_max"])

            written = 0
            for chunk in _iter_ply_chunks(
                ply_path,
                header["dtype"],
                header["vertex_count"],
                header["data_start"],
                chunk_size,
            ):
                x = chunk["x"]
                y = chunk["y"]
                z = chunk["z"]

                if mode == "overlap":
                    mask = _in_bbox_mask(x, y, z, overlap_min, overlap_max)
                else:
                    mask = _in_bbox_mask(x, y, z, strict_min, strict_max)
                    if keep_orphans:
                        in_any_strict = _in_any_bbox_mask(x, y, z, strict_bboxes)
                        in_overlap = _in_bbox_mask(x, y, z, overlap_min, overlap_max)
                        mask |= (~in_any_strict) & in_overlap

                if np.any(mask):
                    chunk[mask].tofile(out)
                    written += int(np.sum(mask))

            print(f"[write] {part_name}: wrote={written}")

    print(f"[-] merged PLY saved: {output_ply}")


if __name__ == "__main__":
    main()
