import os
import json
import math
import random
import shutil
from collections import defaultdict, Counter

import numpy as np

# -----------------------------
# Config (edit here, no argparse)
# -----------------------------
config = {
    "colmap_txt_dir": "old_street_colmap",
    "images_dir": "old_street_colmap\images",
    "out_dir": "out_split",

    # Voxel + clustering
    "voxel_size": 0.2,
    "target_voxel_points_per_cluster": 50000,
    "k_min": 2,
    "k_max": 50,
    "kmeans_method": "kmeans",            # "kmeans" or "minibatch"

    # Visibility thresholds
    "visibility_min_points": 150,
    "visibility_min_ratio": 0.05,

    # Image count constraints
    "min_images_per_cluster": 120,
    "max_images_per_cluster": 600,

    # Overlap
    "overlap_neighbor_k": 2,
    "overlap_extra_image_ratio": 0.15,

    # Export options
    "copy_images": False,
    "export_ply": True,
    "random_seed": 0,
}


def parse_cameras_txt(path):
    cameras = {}
    header_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.lstrip().startswith("#"):
                header_lines.append(line)
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "raw_line": raw,
            }
    return cameras, header_lines


def parse_images_txt(path):
    images = {}
    image_points2d_raw = {}
    header_lines = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            raw = line.rstrip("\n")
            if not raw or raw.lstrip().startswith("#"):
                header_lines.append(line)
                continue
            parts = raw.split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qvec = list(map(float, parts[1:5]))
            tvec = list(map(float, parts[5:8]))
            camera_id = int(parts[8])
            name = " ".join(parts[9:])
            line2 = f.readline()
            if not line2:
                line2 = ""
            raw2 = line2.rstrip("\n")
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "raw_line1": raw,
            }
            image_points2d_raw[image_id] = raw2
    return images, image_points2d_raw, header_lines


def parse_points3d_txt(path):
    points = {}
    header_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.lstrip().startswith("#"):
                header_lines.append(line)
                continue
            parts = raw.split()
            if len(parts) < 8:
                continue
            p3d_id = int(parts[0])
            x_str, y_str, z_str = parts[1], parts[2], parts[3]
            x, y, z = float(x_str), float(y_str), float(z_str)
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            err_str = parts[7]
            error = float(err_str)
            track = []
            for k in range(8, len(parts), 2):
                if k + 1 >= len(parts):
                    break
                try:
                    img_id = int(parts[k])
                    p2d_idx = int(parts[k + 1])
                except ValueError:
                    continue
                track.append((img_id, p2d_idx))
            points[p3d_id] = {
                "id": p3d_id,
                "xyz": np.array([x, y, z], dtype=np.float64),
                "xyz_str": (x_str, y_str, z_str),
                "rgb": (r, g, b),
                "error": error,
                "error_str": err_str,
                "track": track,
            }
    return points, header_lines


def build_visibility_maps(images_dict, image_points2d_dict, points3d_dict):
    image_to_p3d = {}
    image_total_visible = {}
    for image_id, raw_line in image_points2d_dict.items():
        p3d_ids = set()
        total_visible = 0
        if raw_line:
            tokens = raw_line.split()
            for k in range(0, len(tokens), 3):
                if k + 2 >= len(tokens):
                    break
                try:
                    p3d_id = int(tokens[k + 2])
                except ValueError:
                    p3d_id = -1
                if p3d_id > 0:
                    p3d_ids.add(p3d_id)
                    total_visible += 1
        image_to_p3d[image_id] = p3d_ids
        image_total_visible[image_id] = total_visible

    # Build reverse map from image_to_p3d for robustness
    p3d_to_images = defaultdict(set)
    for img_id, p3d_ids in image_to_p3d.items():
        for p3d_id in p3d_ids:
            p3d_to_images[p3d_id].add(img_id)
    # Merge in points3D tracks if present
    for p3d_id, p in points3d_dict.items():
        for img_id, _ in p["track"]:
            p3d_to_images[p3d_id].add(img_id)
    return image_to_p3d, p3d_to_images, image_total_visible


def voxelize_points(points3d_dict, voxel_size):
    if not points3d_dict:
        return np.zeros((0, 3), dtype=np.float64), [], {}
    xyz = np.stack([p["xyz"] for p in points3d_dict.values()], axis=0)
    min_xyz = xyz.min(axis=0)
    voxel_map = defaultdict(list)
    point_ids = list(points3d_dict.keys())
    for idx, p_id in enumerate(point_ids):
        v = tuple(np.floor((points3d_dict[p_id]["xyz"] - min_xyz) / voxel_size).astype(int))
        voxel_map[v].append(p_id)
    voxel_keys = list(voxel_map.keys())
    voxel_points = []
    for v in voxel_keys:
        pts = np.stack([points3d_dict[p_id]["xyz"] for p_id in voxel_map[v]], axis=0)
        voxel_points.append(pts.mean(axis=0))
    voxel_points = np.stack(voxel_points, axis=0) if voxel_points else np.zeros((0, 3))
    voxel_to_p3d_ids = {i: voxel_map[v] for i, v in enumerate(voxel_keys)}
    return voxel_points, voxel_keys, voxel_to_p3d_ids


def _simple_kmeans(points, k, seed=0, max_iter=50):
    n = points.shape[0]
    rng = np.random.default_rng(seed)
    if k >= n:
        labels = np.arange(n)
        return labels, points.copy()
    centers = points[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = points[mask].mean(axis=0)
            else:
                centers[c] = points[rng.integers(0, n)]
    return labels, centers


def _simple_minibatch_kmeans(points, k, seed=0, max_iter=100, batch_size=2048):
    n = points.shape[0]
    rng = np.random.default_rng(seed)
    if k >= n:
        labels = np.arange(n)
        return labels, points.copy()
    centers = points[rng.choice(n, size=k, replace=False)]
    counts = np.zeros(k, dtype=np.int64)
    for _ in range(max_iter):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        batch = points[idx]
        dists = np.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        for j, c in enumerate(nearest):
            counts[c] += 1
            eta = 1.0 / counts[c]
            centers[c] = (1 - eta) * centers[c] + eta * batch[j]
    # final assignment
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels, centers


def cluster_voxels(voxel_points, k, method, seed):
    n = voxel_points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3))
    k = max(1, min(k, n))
    method = method.lower()
    if method == "minibatch":
        try:
            from sklearn.cluster import MiniBatchKMeans
            mbk = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                batch_size=min(2048, n),
                n_init=10,
            )
            labels = mbk.fit_predict(voxel_points)
            centers = mbk.cluster_centers_
            return labels, centers
        except Exception:
            return _simple_minibatch_kmeans(voxel_points, k, seed=seed)
    else:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(voxel_points)
            centers = km.cluster_centers_
            return labels, centers
        except Exception:
            return _simple_kmeans(voxel_points, k, seed=seed)


def build_clusters(labels, voxel_to_p3d_ids, points3d_dict):
    clusters = []
    if labels.size == 0:
        return clusters
    k = int(labels.max()) + 1
    voxel_ids_by_cluster = defaultdict(list)
    for voxel_idx, lab in enumerate(labels):
        voxel_ids_by_cluster[int(lab)].append(voxel_idx)
    for cid in range(k):
        voxel_ids = voxel_ids_by_cluster.get(cid, [])
        point_ids = set()
        for v in voxel_ids:
            point_ids.update(voxel_to_p3d_ids.get(v, []))
        if point_ids:
            pts = np.stack([points3d_dict[p]["xyz"] for p in point_ids], axis=0)
            center = pts.mean(axis=0)
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
        else:
            center = np.zeros(3, dtype=np.float64)
            bbox_min = np.zeros(3, dtype=np.float64)
            bbox_max = np.zeros(3, dtype=np.float64)
        clusters.append({
            "id": cid,
            "voxel_ids": voxel_ids,
            "point_ids": point_ids,
            "center": center,
            "bbox": (bbox_min, bbox_max),
            "image_ids": set(),
            "hit_count_map": {},
            "overlap_added": 0,
            "orig_ids": [cid],
            "warnings": [],
        })
    return clusters


def select_images_for_cluster(cluster, p3d_to_images, image_total_visible_count, thresholds):
    min_points = thresholds["visibility_min_points"]
    min_ratio = thresholds["visibility_min_ratio"]
    hit_counts = Counter()
    for p3d_id in cluster["point_ids"]:
        for img_id in p3d_to_images.get(p3d_id, []):
            hit_counts[img_id] += 1
    selected = set()
    for img_id, cnt in hit_counts.items():
        total = image_total_visible_count.get(img_id, 0)
        ratio = cnt / total if total > 0 else 0.0
        if cnt >= min_points or ratio >= min_ratio:
            selected.add(img_id)
    return selected, dict(hit_counts)


def add_overlap_images(cluster, neighbors, hit_count_map, config):
    base_images = set(cluster["image_ids"])
    neighbor_images = set()
    for nb in neighbors:
        neighbor_images.update(nb["image_ids"])
    candidates = list(neighbor_images - base_images)
    candidates.sort(key=lambda img_id: hit_count_map.get(img_id, 0), reverse=True)
    extra_num = int(math.ceil(len(base_images) * config["overlap_extra_image_ratio"]))
    extra = set(candidates[:extra_num])
    cluster["image_ids"].update(extra)
    cluster["overlap_added"] = len(extra)


def _cluster_distance(a, b):
    return float(np.linalg.norm(a["center"] - b["center"]))


def merge_small_clusters(clusters, config):
    min_images = config["min_images_per_cluster"]
    merge_map = {}
    active = {c["id"]: c for c in clusters}
    changed = True
    while changed:
        changed = False
        small_clusters = [c for c in active.values() if len(c["image_ids"]) < min_images]
        if not small_clusters:
            break
        small_clusters.sort(key=lambda c: len(c["image_ids"]))
        c = small_clusters[0]
        candidates = [o for o in active.values() if o["id"] != c["id"]]
        if not candidates:
            break
        best = None
        best_shared = -1
        best_dist = float("inf")
        for o in candidates:
            shared = len(c["image_ids"] & o["image_ids"])
            dist = _cluster_distance(c, o)
            if shared > best_shared or (shared == best_shared and dist < best_dist):
                best = o
                best_shared = shared
                best_dist = dist
        if best is None:
            break
        # merge c into best
        best["point_ids"].update(c["point_ids"])
        best["voxel_ids"].extend(c["voxel_ids"])
        best["image_ids"].update(c["image_ids"])
        best["orig_ids"].extend(c["orig_ids"])
        # update bbox/center
        bmin = np.minimum(best["bbox"][0], c["bbox"][0])
        bmax = np.maximum(best["bbox"][1], c["bbox"][1])
        best["bbox"] = (bmin, bmax)
        # weighted center by point count
        total_pts = len(best["point_ids"])
        if total_pts > 0:
            # approximate: recompute center from bbox mid if empty
            if c["point_ids"]:
                best["center"] = (bmin + bmax) / 2.0
        merge_map[c["id"]] = best["id"]
        del active[c["id"]]
        changed = True
    merged_clusters = list(active.values())
    return merged_clusters, merge_map


def export_cluster_txt(cluster, cameras, cameras_header, images, image_2d_lines,
                       images_header, points3d, points_header, out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    # cameras.txt (copy all cameras)
    cam_path = os.path.join(out_dir, "cameras.txt")
    with open(cam_path, "w", encoding="utf-8") as f:
        for line in cameras_header:
            f.write(line if line.endswith("\n") else line + "\n")
        for cam_id in sorted(cameras.keys()):
            f.write(cameras[cam_id]["raw_line"] + "\n")

    # points3D.txt (filter by cluster images, trim track)
    image_ids = cluster["image_ids"]
    filtered_points = {}
    for p3d_id in cluster["point_ids"]:
        p = points3d[p3d_id]
        track = [(img_id, p2d_idx) for img_id, p2d_idx in p["track"] if img_id in image_ids]
        if not track:
            continue
        filtered_points[p3d_id] = (p, track)
    point_ids_set = set(filtered_points.keys())
    pts_path = os.path.join(out_dir, "points3D.txt")
    with open(pts_path, "w", encoding="utf-8") as f:
        for line in points_header:
            f.write(line if line.endswith("\n") else line + "\n")
        for p3d_id in sorted(filtered_points.keys()):
            p, track = filtered_points[p3d_id]
            x_str, y_str, z_str = p["xyz_str"]
            r, g, b = p["rgb"]
            err_str = p["error_str"]
            track_tokens = []
            for img_id, p2d_idx in track:
                track_tokens.append(str(img_id))
                track_tokens.append(str(p2d_idx))
            line = " ".join([
                str(p3d_id), x_str, y_str, z_str,
                str(r), str(g), str(b), err_str
            ] + track_tokens)
            f.write(line + "\n")

    # images.txt (filter and replace point3D_id not in cluster with -1)
    img_path = os.path.join(out_dir, "images.txt")
    with open(img_path, "w", encoding="utf-8") as f:
        for line in images_header:
            f.write(line if line.endswith("\n") else line + "\n")
        for img_id in sorted(image_ids):
            if img_id not in images:
                continue
            f.write(images[img_id]["raw_line1"] + "\n")
            raw2d = image_2d_lines.get(img_id, "")
            if not raw2d:
                f.write("\n")
                continue
            tokens_in = raw2d.split()
            tokens = []
            for k in range(0, len(tokens_in), 3):
                if k + 2 >= len(tokens_in):
                    break
                x_str, y_str, p3d_str = tokens_in[k], tokens_in[k + 1], tokens_in[k + 2]
                try:
                    p3d_id = int(p3d_str)
                except ValueError:
                    p3d_id = -1
                p3d_out = str(p3d_id) if p3d_id in point_ids_set else "-1"
                tokens.extend([x_str, y_str, p3d_out])
            f.write(" ".join(tokens) + "\n")

    # Optional image copy
    if config.get("copy_images"):
        img_out_dir = os.path.join(out_dir, "images")
        os.makedirs(img_out_dir, exist_ok=True)
        for img_id in sorted(image_ids):
            if img_id not in images:
                continue
            src = os.path.join(config["images_dir"], images[img_id]["name"])
            dst = os.path.join(img_out_dir, images[img_id]["name"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # Optional PLY
    if config.get("export_ply"):
        try:
            import open3d as o3d
            pts = [points3d[p]["xyz"] for p in point_ids_set]
            if pts:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.stack(pts, axis=0))
                ply_path = os.path.join(out_dir, "points3D.ply")
                o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
        except Exception:
            # silently skip if open3d missing
            pass


def main():
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    colmap_dir = config["colmap_txt_dir"]
    def _resolve_txt(name):
        direct = os.path.join(colmap_dir, name)
        if os.path.isfile(direct):
            return direct
        lower = os.path.join(colmap_dir, name.lower())
        if os.path.isfile(lower):
            return lower
        # Case-insensitive match in directory
        target = name.lower()
        for fn in os.listdir(colmap_dir):
            if fn.lower() == target:
                return os.path.join(colmap_dir, fn)
        return direct

    cameras_path = _resolve_txt("cameras.txt")
    images_path = _resolve_txt("images.txt")
    points_path = _resolve_txt("points3D.txt")

    cameras, cameras_header = parse_cameras_txt(cameras_path)
    images, image_2d_lines, images_header = parse_images_txt(images_path)
    points3d, points_header = parse_points3d_txt(points_path)

    image_to_p3d, p3d_to_images, image_total_visible = build_visibility_maps(
        images, image_2d_lines, points3d
    )

    voxel_points, voxel_keys, voxel_to_p3d_ids = voxelize_points(
        points3d, config["voxel_size"]
    )

    n_vox = voxel_points.shape[0]
    if n_vox == 0:
        raise RuntimeError("No voxel points found. Check points3D.txt")
    k_est = int(math.ceil(n_vox / float(config["target_voxel_points_per_cluster"])))
    k_est = max(config["k_min"], min(config["k_max"], k_est))

    labels, centers = cluster_voxels(voxel_points, k_est, config["kmeans_method"], config["random_seed"])
    clusters = build_clusters(labels, voxel_to_p3d_ids, points3d)

    # Select images by visibility
    for c in clusters:
        image_set, hit_count = select_images_for_cluster(
            c, p3d_to_images, image_total_visible,
            {
                "visibility_min_points": config["visibility_min_points"],
                "visibility_min_ratio": config["visibility_min_ratio"],
            }
        )
        c["image_ids"] = image_set
        c["hit_count_map"] = hit_count

    # Overlap
    for c in clusters:
        # find nearest neighbors
        dists = []
        for other in clusters:
            if other["id"] == c["id"]:
                continue
            dists.append((other, _cluster_distance(c, other)))
        dists.sort(key=lambda x: x[1])
        neighbors = [o for o, _ in dists[:config["overlap_neighbor_k"]]]
        add_overlap_images(c, neighbors, c["hit_count_map"], config)

    # Merge small clusters
    merged_clusters, merge_map = merge_small_clusters(clusters, config)

    # Warnings for too many images
    for c in merged_clusters:
        if len(c["image_ids"]) > config["max_images_per_cluster"]:
            c["warnings"].append("image_count_exceeds_max")

    # Export
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "merge_map": merge_map,
        "clusters": [],
    }
    for idx, c in enumerate(sorted(merged_clusters, key=lambda x: x["id"])):
        cluster_name = f"cluster_{idx:03d}"
        cluster_out = os.path.join(out_dir, cluster_name)
        export_cluster_txt(
            c, cameras, cameras_header, images, image_2d_lines,
            images_header, points3d, points_header, cluster_out, config
        )
        bbox_min, bbox_max = c["bbox"]
        summary["clusters"].append({
            "name": cluster_name,
            "id": c["id"],
            "merged_from": c["orig_ids"],
            "point_count": len(c["point_ids"]),
            "voxel_count": len(c["voxel_ids"]),
            "image_count": len(c["image_ids"]),
            "center": c["center"].tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "overlap_added": c["overlap_added"],
            "warnings": c["warnings"],
        })

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
