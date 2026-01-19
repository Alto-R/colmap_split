import os
import numpy as np
import json

def _default_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)

# ================= 配置区域 (Config) =================
config = {
    # COLMAP 文件夹路径 (文件夹内应包含 Points3D.txt / Images.txt)
    "input_path": _default_path("old_street_colmap"),

    # 输出 JSON 文件的保存路径
    "output_path": _default_path("split_output"),

    # KD-Tree 的深度
    # 1 = 切成 2 块
    # 2 = 切成 4 块
    # 3 = 切成 8 块
    "depth": 3, 

    # 重叠率 (Overlap Ratio)
    # 0.15 代表每个块向外扩展 15% 的长度作为缓冲区
    "overlap": 0.15,

    # 最小观测点数阈值
    # 如果一张照片观测到的点中，有超过 50 个点属于当前分块，则认为这张照片属于该分块
    "min_points_obs": 50
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

        with open(self.points_file, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 4: continue
                pid = int(parts[0])
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                
                self.point_id_to_idx[pid] = len(self.points)
                self.points.append(xyz)
                self.point_ids.append(pid)
        
        self.points = np.array(self.points)
        print(f"    已加载 {len(self.points)} 个 3D 点。")

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
                    # 每3个数据为一组: X, Y, POINT3D_ID
                    for i in range(2, len(parts), 3):
                        pid = int(parts[i])
                        if pid != -1 and pid in self.point_id_to_idx:
                            current_image["seen_points"].append(pid)
                    
                    self.images.append(current_image)
                    is_header = True 
        print(f"    已加载 {len(self.images)} 张图片信息。")

    def kdtree_split(self, point_indices, depth, current_depth=0):
        """KD-Tree 递归切分核心逻辑"""
        # 递归终止条件
        if current_depth >= depth or len(point_indices) < 100:
            return [point_indices]

        # 1. 寻找跨度最大的轴 (X, Y, Z)
        current_points = self.points[point_indices]
        min_xyz = np.min(current_points, axis=0)
        max_xyz = np.max(current_points, axis=0)
        span = max_xyz - min_xyz
        split_axis = np.argmax(span) 

        # 2. 寻找中位数 (Median) 以保证点云数量均衡
        axis_values = current_points[:, split_axis]
        median_val = np.median(axis_values)

        # 3. 拆分
        left_indices = []
        right_indices = []
        
        # 优化：使用 numpy mask 可能会更快，但为了保持索引对应关系，这里使用循环更稳妥
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
            
            # 1. Strict BBox (严格包围盒): 用于最后合并时的裁剪 (Crop)
            min_strict = np.min(pts, axis=0)
            max_strict = np.max(pts, axis=0)
            
            # 2. Overlap BBox (带重叠包围盒): 用于训练时的加载
            dimensions = max_strict - min_strict
            margin = dimensions * overlap_ratio
            min_overlap = min_strict - margin
            max_overlap = max_strict + margin
            
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
                "n_images": len(relevant_images),
                # 重点：这两个 box 是后续流程的关键
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
