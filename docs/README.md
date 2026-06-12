# Dev Tools — Test Outputs (硬性规则)

> **所有 dev_tools 测试脚本生成的图像输出，必须写入本目录的对应阶段子目录。
> 违反此规则即为 Bug，必须修复。**

## 目录树

```
outputs/
├── README.md              # 本文件 — 唯一权威规则
├── sandbox/               # algorithm_sandbox.py 算法实验输出
├── sam/                   # SAM 分割结果（mask、overlay、分层可视化）
├── canny/                 # Canny LineArt 引擎输出（test_canny.py）
├── denoise/               # 物理降噪结果（连通域过滤前后对比）
├── connectivity/          # 连通性检查与修复结果（桥梁标注图）
├── svg/                   # SVG 矢量图输出
└── .gitkeep
```

## 职责分离（与 data/ 和 benchmarks/ 的区别）

| 目录 | 用途 | 文件类型 |
|------|------|----------|
| `outputs/` | **测试脚本运行时输出**（图像、中间结果） | `.png` `.jpg` `.svg` |
| `data/` | **训练数据与模型** | `.pkl` `.json` `.csv` |
| `benchmarks/` | **静态对比工具与归档** | `.html` `.png` `.json` |

## 规则

### 硬性规则
1. `dev_tools/scripts/` 下的所有测试脚本，必须将输出图写入 `outputs/<stage>/`
2. 禁止将测试图写入 `data/`、`benchmarks/` 或项目根目录
3. 每新增一个管线阶段，必须先在本目录下创建对应子目录

### 命名约定
- 输出文件名使用原始图片的 stem：`{image_stem}_{stage}.png`
- 例：`上海_canny.png`、`cityscape_6_sam_mask.png`、`abstract_1_denoised.png`

## 现有脚本对应关系

| 脚本 | 阶段 | 输出目录 |
|------|------|----------|
| `scripts/test_sam_segment.py` | SAM 分割 | `outputs/sam/` |
| `scripts/test_canny.py` | 线稿提取 | `outputs/canny/` |
| `scripts/test_denoise.py` | 物理降噪 | `outputs/denoise/` |
| `scripts/test_pipeline.py` | 连通+SVG | `outputs/connectivity/` `outputs/svg/` |

## 使用示例

```bash
# Canny 线稿测试
python dev_tools/scripts/test_canny.py
# → outputs/canny/{image_stem}_canny.png

# SAM 分割测试（未来）
python dev_tools/scripts/test_sam.py
# → outputs/sam/{image_stem}_mask.png
```

## 可移植引擎的特殊说明

`dev_tools/lineart_engine_export/` 下的脚本（`batch_run.py`、`test_single.py`）是独立可移植引擎，
不硬编码项目路径。在项目内使用时，**调用方负责将输出定向到 `outputs/canny/`**：

```bash
python dev_tools/lineart_engine_export/batch_run.py \
  dev_tools/test_imgs/ \
  dev_tools/outputs/canny/
```
