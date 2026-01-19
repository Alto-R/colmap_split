import os
import numpy as np
import json

def _default_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)

# ================= 配置区域 =================
config = {
    # COLMAP 文件夹路径 (文件夹内应包含 Points3D.txt / Images.txt)
    "input_path": _default_path("old_street_colmap"),

    # 输出 JSON 文件的保存路径
    "output_path": _default_path("split_output"),

    # KD-Tree 的深度
    # 1 = 切成 2 块
    # 2 = 切成 4 块
    # 3 = 切成 8 块
    "depth": 4, 

    # 重叠率
    # 0.08 代表每个块向外扩展 8% 的长度作为缓冲区
    "overlap": 0.08,

    # 最小观测点数阈值
    # 如果一张照片观测到的点中，有超过 50 个点属于当前分块，则认为这张照片属于该分块
    "min_points_obs": 50,

    # 预处理：拆分前清洗点云（可选）
    # 方法: "statistical" | "abs_clip" | "bbox"
    "preprocess": {
        "enabled": True,

        # 1) 坐标绝对值裁剪
        "abs_clip": {
            "abs_max": 5000.0
        },

        # 2) 轨迹长度过滤
        "min_track_len": 3,

        # 3) 统计离群点移除
        "statistical": {
            "nb_neighbors": 50,
            "std_ratio": 1.5
        },

        # 导出清洗后的点云
        "export_clean_points": {
            "enabled": True,
            "output_path": _default_path("old_street_colmap", "points3D_clean.txt")
        }
    }
}
# ====================================================

class ColmapSplitter:
    def __init__(self, base_path):
        self.base_path = base_path
        self.points_file = self._resolve_colmap_file(
            base_path, ["Points3D.txt", "points3D.txt"]
        )
        self.images_file = self._resolve_colmap_file(
            base_path, ["Images.txt", "images.txt"]
        )
        
        self.points = [] 
        self.point_ids = [] 
        self.point_id_to_idx = {} 
        self.images = [] 

    @staticmethod
    def _resolve_colmap_file(base_path, candidates):
        for name in candidates:
            path = os.path.join(base_path, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"在 {base_path} 未找到文件: {', '.join(candidates)}"
        )

    def load_data(self):
        """读取 COLMAP 的 Points3D.txt / Images.txt"""
        print(f"[-] 正在读取点云: {self.points_file}")
        if not os.path.exists(self.points_file):
            raise FileNotFoundError(f"找不到文件: {self.points_file}")

        header_lines = []
        raw_lines = []
        raw_point_ids = []
        raw_points = []
        raw_track_lengths = []

        with open(self.points_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line)
                    continue
                parts = line.split()
                if len(parts) < 8: continue
                pid = int(parts[0])
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]

                track_len = (len(parts) - 8) // 2



                
                raw_lines.append(line)
                raw_points.append(xyz)
                raw_point_ids.append(pid)

                raw_track_lengths.append(track_len)

        
        raw_points = np.array(raw_points)

        raw_track_lengths = np.array(raw_track_lengths)

        print(f"    原始点数: {len(raw_points)}")

        print(f"    [Before] BBox:\\n      Min: {np.min(raw_points, axis=0)}\\n      Max: {np.max(raw_points, axis=0)}")


        # 预处理清洗（可选）
        points, keep_mask = self._preprocess_points(raw_points, raw_track_lengths)
        if keep_mask is None:
            keep_mask = np.ones(len(raw_points), dtype=bool)
            points = raw_points

        self.points = points
        self.point_ids = [pid for pid, keep in zip(raw_point_ids, keep_mask) if keep]
        self.point_id_to_idx = {pid: i for i, pid in enumerate(self.point_ids)}
        print(f"    清洗后点数: {len(self.points)}")

        if len(self.points) > 0:

            print(f"    [After]  BBox:\\n      Min: {np.min(self.points, axis=0)}\\n      Max: {np.max(self.points, axis=0)}")


        # 可选导出清洗后的 points3D.txt
        preprocess_cfg = config.get("preprocess", {})
        export_cfg = preprocess_cfg.get("export_clean_points", {})
        if export_cfg.get("enabled", False):
            out_path = export_cfg.get("output_path")
            if not out_path:
                raise ValueError("preprocess.export_clean_points.output_path 未设置")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                for line in header_lines:
                    f.write(line)
                for line, keep in zip(raw_lines, keep_mask):
                    if keep:
                        f.write(line)
            print(f"    已导出清洗后的点云: {out_path}")

        print(f"[-] 正在读取相机视图: {self.images_file}")
        if not os.path.exists(self.images_file):
            raise FileNotFoundError(f"找不到文件: {self.images_file}")

        with open(self.images_file, "r") as f:
            current_image = None
            is_header = True
            for line in f:
                if line.startswith("#") or line.strip() == "": continue
                
                if is_header:
                    parts = line.split()
                    # images.txt 格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    current_image = {
                        "name": parts[9],
                        "seen_points": []
                    }
                    is_header = False
                else:
                    parts = line.split()
                    # 每 3 个数据为一组: X, Y, POINT3D_ID
                    for i in range(2, len(parts), 3):
                        pid = int(parts[i])
                        if pid != -1 and pid in self.point_id_to_idx:
                            current_image["seen_points"].append(pid)
                    
                    self.images.append(current_image)
                    is_header = True 
        print(f"    已加载 {len(self.images)} 张图片信息。")

    def _preprocess_points(self, points_np, track_lengths=None):
        """
        预处理流水线：坐标裁剪 -> 轨迹长度 -> 统计离群点
        """
        preprocess_cfg = config.get("preprocess", {})
        if not preprocess_cfg.get("enabled", False):
            return points_np, np.ones(len(points_np), dtype=bool)

        final_mask = np.ones(len(points_np), dtype=bool)

        # 1) 坐标绝对值裁剪
        if "abs_clip" in preprocess_cfg:
            abs_cfg = preprocess_cfg["abs_clip"]
            abs_max = abs_cfg.get("abs_max", None)
            if abs_max is not None:
                print(f"    [Preprocess] Absolute clip: max={abs_max}")
                mask_clip = np.all(np.abs(points_np) < float(abs_max), axis=1)
                final_mask &= mask_clip

        # 2) 轨迹长度过滤
        if "min_track_len" in preprocess_cfg and track_lengths is not None:
            min_len = int(preprocess_cfg["min_track_len"])
            print(f"    [Preprocess] Track length: min_track={min_len}")
            mask_track = track_lengths >= min_len
            final_mask &= mask_track

        # 3) 统计离群点移除
        if "statistical" in preprocess_cfg:
            st_cfg = preprocess_cfg["statistical"]
            nb = int(st_cfg.get("nb_neighbors", 20))
            std = float(st_cfg.get("std_ratio", 2.0))
            print(f"    [Preprocess] Statistical: nb={nb}, std={std}")
            try:
                import open3d as o3d

                current_valid_indices = np.where(final_mask)[0]
                if len(current_valid_indices) > 0:
                    valid_points = points_np[current_valid_indices]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_points)

                    _, inlier_indices_local = pcd.remove_statistical_outlier(
                        nb_neighbors=nb, std_ratio=std
                    )

                    new_mask_local = np.zeros(len(valid_points), dtype=bool)
                    new_mask_local[inlier_indices_local] = True
                    final_mask[current_valid_indices] = new_mask_local
            except ImportError:
                print("    [Warning] Open3D not found, skipping statistical")
            except Exception as e:
                print(f"    [Warning] Statistical failed: {e}")

        return points_np[final_mask], final_mask    
    
    def kdtree_split(self, point_indices, depth, current_depth=0):
        """KD-Tree 递归切分核心逻辑"""
        # 递归终止条件
        if current_depth >= depth or len(point_indices) < 50:
            return [point_indices]

        # 1. 寻找跨度最大的轴 (X, Y, Z)
        current_points = self.points[point_indices]
        min_xyz = np.min(current_points, axis=0)
        max_xyz = np.max(current_points, axis=0)
        span = max_xyz - min_xyz
        split_axis = np.argmax(span) 

        # 2. 寻找中位数以保证点云数量均衡
        axis_values = current_points[:, split_axis]
        median_val = np.median(axis_values)

        # 3. 拆分
        left_indices = []
        right_indices = []
        
        # 优化：使用 numpy 的 mask 可能更快，但为保持索引对应关系，这里用循环更稳妥
        # 若数据量极大(千万级)，此处可进一步优化
        for idx in point_indices:
            val = self.points[idx][split_axis]
            if val <= median_val:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
        
        # 4. 递归下一层
        return (self.kdtree_split(left_indices, depth, current_depth + 1) + 
                self.kdtree_split(right_indices, depth, current_depth + 1))

    def export_partitions(self, partitions, output_dir, overlap_ratio, min_points_obs):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[-] 开始导出分块配置到: {output_dir}")
        
        for i, indices in enumerate(partitions):
            part_name = f"part_{i}"
            
            # --- 核心几何计算 ---
            pts = self.points[indices]
            
            # 1. 严格包围盒：用于最后合并时的裁剪
            min_strict = np.min(pts, axis=0)
            max_strict = np.max(pts, axis=0)
            
            # 2. 重叠包围盒：用于训练时的加载
            dimensions = max_strict - min_strict
            margin = dimensions * overlap_ratio
            min_overlap = min_strict - margin
            max_overlap = max_strict + margin

            # 3. 重叠区点数量（包围盒内的点数）
            # 注意：这是全局点云中落在重叠包围盒内的点数
            overlap_mask = np.all(
                (self.points >= min_overlap) & (self.points <= max_overlap),
                axis=1,
            )
            overlap_n_points = int(np.sum(overlap_mask))
            
            # --- 图片筛选逻辑 ---
            # 优化：使用 set 提高查找速度
            part_point_ids = set([self.point_ids[idx] for idx in indices])
            relevant_images = []
            
            for img in self.images:
                count = 0
                for pid in img["seen_points"]:
                    if pid in part_point_ids:
                        count += 1
                
                if count > min_points_obs:
                    relevant_images.append(img["name"])

            # --- 导出 JSON ---
            part_data = {
                "name": part_name,
                "n_points": len(indices),
                "overlap_n_points": overlap_n_points,
                "n_images": len(relevant_images),
                # 重点：这两个包围盒是后续流程的关键
                "strict_min": min_strict.tolist(),
                "strict_max": max_strict.tolist(),
                "overlap_min": min_overlap.tolist(),
                "overlap_max": max_overlap.tolist(),
                "image_list": relevant_images
            }
            
            out_file = os.path.join(output_dir, f"{part_name}.json")
            with open(out_file, "w") as f:
                json.dump(part_data, f, indent=4)
            
            print(f"    [Part {i}] Points: {len(indices)} | Images: {len(relevant_images)} | Output: {out_file}")

if __name__ == "__main__":
    # 实例化并运行
    try:
        splitter = ColmapSplitter(config["input_path"])
        splitter.load_data()
        
        # 获取所有点的初始索引
        all_indices = list(range(len(splitter.points)))
        
        print("[-] 正在进行 KD-Tree 计算...")
        partitions = splitter.kdtree_split(all_indices, config["depth"])
        
        splitter.export_partitions(
            partitions, 
            config["output_path"], 
            overlap_ratio=config["overlap"],
            min_points_obs=config["min_points_obs"]
        )
        print("[-] 所有任务完成。")
        
    except Exception as e:
        print(f"[Error] 发生错误: {e}")
