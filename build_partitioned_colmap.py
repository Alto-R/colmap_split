import os
import json
import shutil
from pathlib import Path


def _default_path(*parts):
    return Path(__file__).resolve().parent.joinpath(*parts)


# ================= 配置区域 (Config) =================
config = {
    # 原始 COLMAP 文件夹（包含 Cameras.txt / Images.txt / Points3D.txt）
    "colmap_dir": _default_path("old_street_colmap"),

    # 原始图片目录
    "images_dir": _default_path("old_street_colmap", "images"),

    # 分块 JSON 目录（partition_colmap.py 输出）
    "parts_dir": _default_path("split_output"),

    # 输出根目录
    "output_root": _default_path("split_colmap"),

    # 输出结构：part_x/sparse/0 + part_x/images
    "sparse_relpath": os.path.join("sparse", "0"),
    "images_relpath": "images",

    # 点的最小 track 长度（过滤掉只有 1 次观测的点）
    "min_track_len": 2,

    # 点过滤模式: "overlap" | "strict" | "none"
    # 用于分块并行训练时，推荐使用 overlap 或 strict 来限制点数量
    "point_filter": "overlap",

    # 图片复制模式: "copy" | "hardlink" | "symlink"
    "image_mode": "copy",

    # 如果输出已存在，是否覆盖
    "overwrite": True,
    # 仅打印不写出
    "dry_run": False,
}
# ====================================================


def build_image_index(images_path):
    header_lines = []
    name_to_id = {}
    id_to_camera = {}

    with open(images_path, "r") as f:
        it = iter(f)
        for line in it:
            if line.startswith("#"):
                header_lines.append(line)
                continue
            if line.strip() == "":
                continue

            parts = line.split()
            if len(parts) < 9:
                continue
            image_id = int(parts[0])
            camera_id = int(parts[8])
            name = " ".join(parts[9:])
            name_to_id[name] = image_id
            id_to_camera[image_id] = camera_id

            # skip points line
            for pline in it:
                if pline.startswith("#") or pline.strip() == "":
                    continue
                break

    return header_lines, name_to_id, id_to_camera


def _in_bbox(xyz, bbox_min, bbox_max):
    return (
        bbox_min[0] <= xyz[0] <= bbox_max[0]
        and bbox_min[1] <= xyz[1] <= bbox_max[1]
        and bbox_min[2] <= xyz[2] <= bbox_max[2]
    )


def filter_points3d(
    points_path,
    out_path,
    selected_image_ids,
    min_track_len,
    bbox_min=None,
    bbox_max=None,
):
    kept_point_ids = set()
    kept_count = 0
    with open(points_path, "r") as f, open(out_path, "w") as out:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                out.write(line)
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            pid = int(parts[0])
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            base = parts[:8]
            track_parts = parts[8:]

            if bbox_min is not None and bbox_max is not None:
                if not _in_bbox(xyz, bbox_min, bbox_max):
                    continue

            new_track = []
            for i in range(0, len(track_parts), 2):
                if i + 1 >= len(track_parts):
                    break
                img_id = int(track_parts[i])
                pt_idx = int(track_parts[i + 1])
                if img_id in selected_image_ids:
                    new_track.append((img_id, pt_idx))

            if len(new_track) < min_track_len:
                continue

            kept_point_ids.add(pid)
            track_tokens = []
            for img_id, pt_idx in new_track:
                track_tokens.append(str(img_id))
                track_tokens.append(str(pt_idx))
            out.write(" ".join(base + track_tokens) + "\n")
            kept_count += 1

    return kept_point_ids, kept_count


def filter_images(images_path, out_path, selected_image_ids, kept_point_ids, header_lines):
    written = 0
    with open(images_path, "r") as f, open(out_path, "w") as out:
        for h in header_lines:
            out.write(h)

        it = iter(f)
        for line in it:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.split()
            if len(parts) < 9:
                continue
            image_id = int(parts[0])
            selected = image_id in selected_image_ids

            # get points line
            points_line = None
            for pline in it:
                if pline.startswith("#") or pline.strip() == "":
                    continue
                points_line = pline
                break

            if not selected:
                continue

            out.write(line)
            if points_line is None:
                out.write("\n")
                written += 1
                continue

            pparts = points_line.split()
            if len(pparts) % 3 != 0:
                # still try to process by triples
                pass

            out_tokens = []
            for i in range(0, len(pparts), 3):
                if i + 2 >= len(pparts):
                    break
                x_str = pparts[i]
                y_str = pparts[i + 1]
                pid = int(pparts[i + 2])
                if pid != -1 and pid not in kept_point_ids:
                    pid = -1
                out_tokens.extend([x_str, y_str, str(pid)])

            out.write(" ".join(out_tokens) + "\n")
            written += 1

    return written


def copy_images(image_names, src_dir, dst_dir, mode, overwrite, dry_run):
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    missing = 0

    for name in image_names:
        src = Path(src_dir) / name
        dst = Path(dst_dir) / name
        if not src.exists():
            missing += 1
            continue
        if dst.exists() and not overwrite:
            continue

        if dry_run:
            copied += 1
            continue

        try:
            if mode == "hardlink":
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            elif mode == "symlink":
                if dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)

        copied += 1

    return copied, missing


def load_partitions(parts_dir):
    parts = []
    for p in sorted(Path(parts_dir).glob("part_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts.append((p.stem, data))
    return parts


def main():
    colmap_dir = Path(config["colmap_dir"])
    images_dir = Path(config["images_dir"])
    parts_dir = Path(config["parts_dir"])
    output_root = Path(config["output_root"])

    cameras_path = colmap_dir / "Cameras.txt"
    images_path = colmap_dir / "Images.txt"
    points_path = colmap_dir / "Points3D.txt"

    if not cameras_path.exists():
        raise FileNotFoundError(f"找不到 Cameras.txt: {cameras_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"找不到 Images.txt: {images_path}")
    if not points_path.exists():
        raise FileNotFoundError(f"找不到 Points3D.txt: {points_path}")
    if not parts_dir.exists():
        raise FileNotFoundError(f"找不到 parts_dir: {parts_dir}")

    header_lines, name_to_id, _ = build_image_index(images_path)
    parts = load_partitions(parts_dir)
    if not parts:
        raise ValueError("未找到 part_*.json")

    print(f"[-] parts: {len(parts)}")
    print(f"[-] images indexed: {len(name_to_id)}")

    for part_name, pdata in parts:
        image_names = pdata.get("image_list", [])
        selected_image_ids = set()
        missing_names = []
        for name in image_names:
            img_id = name_to_id.get(name)
            if img_id is None:
                missing_names.append(name)
                continue
            selected_image_ids.add(img_id)

        part_dir = output_root / part_name
        sparse_dir = part_dir / config["sparse_relpath"]
        images_out_dir = part_dir / config["images_relpath"]

        bbox_min = None
        bbox_max = None
        if config.get("point_filter") == "strict":
            bbox_min = pdata.get("strict_min")
            bbox_max = pdata.get("strict_max")
        elif config.get("point_filter") == "overlap":
            bbox_min = pdata.get("overlap_min")
            bbox_max = pdata.get("overlap_max")

        if part_dir.exists():
            if not config["overwrite"]:
                raise FileExistsError(f"输出已存在: {part_dir} (请开启 overwrite)")
            if not config["dry_run"]:
                shutil.rmtree(part_dir)

        if not config["dry_run"]:
            os.makedirs(sparse_dir, exist_ok=True)
            os.makedirs(images_out_dir, exist_ok=True)

            # Cameras.txt 直接复制
            shutil.copy2(cameras_path, sparse_dir / "Cameras.txt")

        # Points3D.txt
        points_out = sparse_dir / "Points3D.txt"
        kept_point_ids, kept_count = filter_points3d(
            points_path,
            points_out,
            selected_image_ids,
            config["min_track_len"],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        # Images.txt
        images_out = sparse_dir / "Images.txt"
        written_images = filter_images(
            images_path, images_out, selected_image_ids, kept_point_ids, header_lines
        )

        # images folder
        copied, missing = copy_images(
            image_names,
            images_dir,
            images_out_dir,
            config["image_mode"],
            config["overwrite"],
            config["dry_run"],
        )

        print(
            f"[{part_name}] images={written_images}, points={kept_count}, "
            f"copied={copied}, missing_imgs={missing}, missing_names={len(missing_names)}"
        )

    print("[-] 完成。")


if __name__ == "__main__":
    main()
