# COLMAP 分块与分组导出

本目录用于将 COLMAP 稀疏重建结果进行空间分块，并生成每个分块的独立 COLMAP 子集（含图片分组）。

## 目录结构（推荐）

```
colmap_split/
  old_street_colmap/
    Cameras.txt
    Images.txt
    Points3D.txt
    points3D_clean.txt  (可选，清洗后点云)
    images/
  split_output/
  split_colmap/
  partition_colmap.py
  build_partitioned_colmap.py
```

## 工作流

1) 分块并生成 JSON  
运行 `partition_colmap.py`，输出每个分块的 bbox 与图片列表。

2) 生成分块 COLMAP  
运行 `build_partitioned_colmap.py`，为每个分块生成独立的 `sparse/0` 与 `images` 目录。
如果 `points3D_clean.txt` 存在，则默认使用清洗后点云；否则使用原始 `Points3D.txt`。

## 1. 分块脚本：partition_colmap.py

核心功能：
- 读取 `Points3D.txt` 与 `Images.txt`
- 可选的点云预处理（统计式离群点移除 / 坐标裁剪 / bbox 裁剪）
- KD-Tree 分块
- 输出 `part_*.json`

关键配置（文件顶部 `config`）：

```
input_path    # COLMAP 稀疏目录（含 Points3D.txt / Images.txt）
output_path   # JSON 输出目录
depth         # KD-Tree 深度（2^depth 个分块）
overlap       # 重叠比例（bbox 外扩）
min_points_obs# 图片归属阈值（分块内观测点数）

preprocess.enabled           # 可选点云清洗开关
preprocess.abs_clip          # 坐标绝对值裁剪
preprocess.min_track_len     # 轨迹长度过滤
preprocess.statistical       # 统计离群点移除
preprocess.export_clean_points # 导出清洗后点云
```

产物示例（JSON）：
- `strict_min / strict_max`：严格包围盒
- `overlap_min / overlap_max`：带重叠包围盒
- `image_list`：分块内相关图片

## 2. 分块导出脚本：build_partitioned_colmap.py

核心功能：
- 根据 `split_output/part_*.json` 生成分块 COLMAP
- 支持按 bbox 过滤点（适合并行训练）
- 复制/硬链接/软链接图片到分块目录

关键配置（文件顶部 `config`）：

```
colmap_dir    # 原始 COLMAP 文件夹
images_dir    # 原始图片目录
parts_dir     # 分块 JSON 目录
output_root   # 输出根目录
min_track_len # 点在该分块内的最小观测次数
point_filter  # "overlap" | "strict" | "none"
image_mode    # "copy" | "hardlink" | "symlink"
clean_points_filename  # 清洗后点云文件名（默认 points3D_clean.txt）
points_filename        # 原始点云文件名（默认 Points3D.txt）
overwrite     # 是否覆盖已有输出（会删除已有分块目录）
```

输出目录结构（每个分块）：

```
split_colmap/
  part_0/
    sparse/0/
      Cameras.txt
      Images.txt
      Points3D.txt
    images/
      *.jpg
```

## 3. 合并分块训练结果：merge_partitioned_gaussians.py

用途：将每个分块训练得到的高斯点云（PLY）合并为一个整体。  
默认会使用 `strict_min/max` 作为“清晰区域”，以避免 overlap 区域重复和拼接不自然。

关键配置（文件顶部 `config`）：

```
parts_dir        # partition_colmap.py 输出的 split_output 目录
gaussian_root    # 分块训练结果根目录（含 part_0, part_1, ...）
gaussian_relpath # 每个 part 内 PLY 相对路径（支持 glob，如 output/*.ply）
output_ply       # 合并后的输出 PLY
mode             # "strict" | "overlap"
keep_orphans     # strict 模式下，保留不在任何 strict bbox 内但落在 overlap 的点
chunk_size       # 分块读取大小
```

建议：
- `mode="strict"`：只保留每个分块最清晰的区域，拼接更自然。
- 若发现边缘有缺口，可开启 `keep_orphans=True`。

## 点过滤逻辑说明（build_partitioned_colmap.py）

- `point_filter="overlap"`（推荐并行训练）
  - 点需落在 `overlap_min/max` 的 bbox 内
  - 且在该分块图片中至少有 `min_track_len` 个观测
- `point_filter="strict"`：更严格，点数更少
- `point_filter="none"`：仅按图片可见点过滤（点数可能膨胀）

## 依赖

- Python 3
- 可选：`open3d`（仅当 `partition_colmap.py` 使用 statistical 清洗时需要）

## 常见问题

- 修改 `overlap` 后请先重新运行 `partition_colmap.py` 生成新的 JSON。
- `depth` 越大分块越多，但受最小叶子点数限制，实际块数可能少于 `2^depth`。
- 如需保证分块 bbox 与实际点筛选完全一致，请确保 `points3D_clean.txt` 存在；脚本会优先使用清洗后点云。
