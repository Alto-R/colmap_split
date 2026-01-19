# COLMAP 分块与分组导出

本目录用于将 COLMAP 稀疏重建结果进行空间分块，并生成每个分块的独立 COLMAP 子集（含图片分组）。

## 目录结构（推荐）

```
colmap_split/
  old_street_colmap/
    Cameras.txt
    Images.txt
    Points3D.txt
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

preprocess.enabled / method  # 可选点云清洗
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
