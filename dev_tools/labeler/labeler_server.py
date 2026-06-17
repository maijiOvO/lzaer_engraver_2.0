#!/usr/bin/env python3
"""开发者标定工具 — 独立 Web 后端。

启动: python3 dev_tools/labeler/labeler_server.py
访问: http://localhost:8090

完全独立于 client_app，不依赖客户端任何基础设施。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── 路径注入（与 test_sam_segment.py 一致的策略）────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "dev_tools" / "scripts"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# ── 环境修复 ────────────────────────────────────────────────
# 禁用 tqdm 进度条（后台运行时 stderr pipe 断裂导致 BrokenPipeError）
os.environ.setdefault("TQDM_DISABLE", "1")
# Docker 路径 → 本地路径
os.environ.setdefault("MODEL_DIR", str(BACKEND_DIR / "models"))
os.environ.setdefault("OUTPUT_DIR", str(BACKEND_DIR / "outputs"))

# ── 评分引擎 ──────────────────────────────────────────────
from score_engine import score_segmentation

# ── 笔刷事件记录 ──────────────────────────────────────────
from brush_recorder import record_event as _record_brush_event, get_event_count as _brush_event_count

# ── 复用 test_sam_segment 的 ImageRegistry（保证格式兼容）──
from test_sam_segment import ImageRegistry, hash_file, extract_features

# ── 目录 ──────────────────────────────────────────────────
TEST_IMGS_DIR = PROJECT_ROOT / "dev_tools" / "test_imgs" / "train"  # 默认训练集
TEST_IMGS_BASE = PROJECT_ROOT / "dev_tools" / "test_imgs"
SAM_OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "sam"
DATA_DIR = PROJECT_ROOT / "dev_tools" / "data"
REGISTRY_PATH = DATA_DIR / "labeled.json"
STATIC_DIR = Path(__file__).parent / "static"

os.makedirs(SAM_OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="Labeler Dev Tool", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────

class SegmentRequest(BaseModel):
    image_name: str
    n_layers: int = Field(default=3, ge=2, le=6)
    frame_width: int = Field(default=50, ge=20, le=200)
    min_island_area: int = Field(default=100, ge=10, le=5000)
    quality: str = Field(default="standard", pattern="^(draft|standard|fine)$")
    refine_mode: str = Field(default="sam_driven", pattern="^(none|slic|boundary|sam_driven)$")

class SaveRequest(BaseModel):
    image_name: str
    n_layers: int
    frame_width: int
    min_island_area: int
    quality: str = "standard"
    scores: dict | None = None
    features: dict | None = None

class SkipRequest(BaseModel):
    image_name: str

class SetDirRequest(BaseModel):
    image_dir: str

class BrushStrokes(BaseModel):
    brush_type: str  # "include" | "exclude"
    points: list[list[int]]  # [[x, y], ...] in original image coordinates

class BrushRefineRequest(BaseModel):
    image_name: str
    layer_index: int
    strokes: list[BrushStrokes]  # 多组笔刷笔画
    current_mask_key: str  # 当前 mask 的文件名 stem


# ── 图片扫描 ──────────────────────────────────────────────

def scan_images(image_dir: str | None = None):
    """扫描目录，返回图片列表 + 统计。支持 train/val 子目录。"""
    directory = Path(image_dir) if image_dir else TEST_IMGS_DIR
    if not directory.is_absolute():
        directory = TEST_IMGS_BASE / directory
    directory = directory.resolve()
    if not directory.is_dir():
        directory = TEST_IMGS_DIR  # fallback

    labeled_status: dict[str, str] = {}
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                reg = json.load(f)
            for entry in reg.get("images", {}).values():
                name = entry.get("filename", "")
                labeled_status[name] = "done" if entry.get("params") else "skipped"
        except (json.JSONDecodeError, KeyError):
            pass

    images = []
    stats = {"total": 0, "done": 0, "skipped": 0, "new": 0}
    for f in sorted(directory.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = f.name
        status = labeled_status.get(name, "new")
        images.append({"name": name, "status": status})
        stats[status] += 1
        stats["total"] += 1

    return {
        "images": images,
        "stats": stats,
        "image_dir": str(directory),
        "active_subdir": directory.name,
    }


# ── 分割 + 评分 ───────────────────────────────────────────

def run_segmentation(req: SegmentRequest) -> dict:
    img_path = TEST_IMGS_DIR / req.image_name
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {req.image_name}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"无法解码: {req.image_name}")

    stem = Path(req.image_name).stem
    orig_h, orig_w = image.shape[:2]

    from app.utils.depth_engine import estimate_depth

    # ── Step 1: 深度估计（返回模型原生分辨率，避免 13x 上采样抹平细节）──
    depth_cache = SAM_OUTPUT_DIR / f"{stem}_depth.npy"
    if depth_cache.exists():
        depth_map = np.load(str(depth_cache))
        depth_cached = True
    else:
        depth_map = estimate_depth(image)
        np.save(str(depth_cache), depth_map)
        depth_cached = False

    depth_h, depth_w = depth_map.shape[:2]

    # ── 自动推断最优层数 ─────────────────────────────
    from app.utils.structural_segmentation import suggest_n_layers
    suggested_n = int(suggest_n_layers(depth_map))
    # 首次（默认值3）自动推断；用户手动调整后用用户的值
    n_layers = req.n_layers
    if n_layers == 3 and suggested_n != 3:
        n_layers = suggested_n

    # ── 缩放到深度图分辨率 ──────────────────────────
    if (depth_h, depth_w) != (orig_h, orig_w):
        scale_x = depth_w / orig_w
        scale_y = depth_h / orig_h
        work_image = cv2.resize(image, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
        work_frame_width = max(3, int(req.frame_width * scale_x))
        work_min_island = max(10, int(req.min_island_area * scale_x * scale_y))
    else:
        work_image = image
        work_frame_width = req.frame_width
        work_min_island = req.min_island_area

    refine_mode = getattr(req, "refine_mode", "boundary")

    # sam_driven 模式下限死 3 层——更宽的阈值保证 SAM 区块不被切散
    if refine_mode == "sam_driven" and n_layers > 3:
        n_layers = 3

    # 标志位：sam_driven 模式直接工作在原图分辨率，跳过深度分辨率缩放和后续 SAM 精修
    sam_driven_mode = False

    # ── Step 2: 选择分割算法 ─────────────────────────
    if refine_mode == "boundary":
        from boundary_refine import refine_layers
        from app.utils.structural_segmentation import quantize_depth, generate_frame_mask

        raw_masks = quantize_depth(depth_map, n_layers=n_layers)
        layer_masks, stats = refine_layers(
            work_image, raw_masks, depth_map,
            band_width=5, min_component_area=100, box_padding=10,
        )
        frame_mask = generate_frame_mask(depth_h, depth_w, frame_width=work_frame_width)
        # 每层添加外框
        frame_bin = frame_mask > 0
        for mask in layer_masks:
            mask[frame_bin] = 255
    elif refine_mode == "sam_driven":
        # 新管线：SAM 自动分割 → 深度中位数归属 → 连通修复
        # 工作在原图分辨率，SAM 决定对象形状，深度仅决定 Z 轴排序
        from app.utils.sam_engine import run_sam_automatic
        from app.utils.structural_segmentation import build_sam_driven_layers

        sam_driven_mode = True
        sam_cache = SAM_OUTPUT_DIR / f"{stem}_sam_region.npz"
        sam_masks, _region_map = run_sam_automatic(image, cache_path=str(sam_cache))
        layer_masks, frame_mask, stats = build_sam_driven_layers(
            sam_masks, depth_map, n_layers,
            image_shape=(orig_h, orig_w),
            frame_width=req.frame_width,
            min_island_area=req.min_island_area,
        )
        frame_mask_full = frame_mask  # 已是原图分辨率，跳过 Step 6
    elif refine_mode == "none":
        # 纯等距量化 + 连通修复（客户端算法）
        from app.utils.structural_segmentation import (
            quantize_depth, generate_frame_mask, repair_layer_mask,
        )
        raw_masks = quantize_depth(depth_map, n_layers=n_layers)
        frame_mask = generate_frame_mask(depth_h, depth_w, frame_width=work_frame_width)
        layer_masks = []
        stats = []
        for i, raw in enumerate(raw_masks):
            repaired, bridges, erased = repair_layer_mask(raw, frame_mask, min_island_area=work_min_island)
            layer_masks.append(repaired)
            fg = int(np.count_nonzero(repaired))
            stats.append({
                "layer_index": i, "fg_pixels": fg,
                "fg_pct": round(fg / (depth_h * depth_w) * 100, 2),
                "bridges_built": bridges, "islands_erased": erased,
            })
    else:
        # slic — 旧的 SLIC 超像素 + 深度投票
        from slic_segmentation import slic_depth_layers
        layer_masks, frame_mask, stats = slic_depth_layers(
            work_image, depth_map,
            n_layers=n_layers,
            n_segments=300, compactness=10.0,
            border_width=work_frame_width,
        )

    # ── Step 4: SAM 精修（sam_driven 模式下已在 Step 2 完成）──
    if req.quality != "draft" and not sam_driven_mode:
        try:
            from app.utils.sam_engine import refine_mask
        except ImportError:
            refine_mask = None

        if refine_mask is not None:
            for i, mask in enumerate(layer_masks):
                if not mask.any():
                    continue
                try:
                    refined = refine_mask(work_image, mask)
                    if refined is not None:
                        layer_masks[i] = refined
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"层{i} SAM精修失败（使用原始蒙版）| {e}"
                    )

    # ── Step 5: uint8 规范化 ────────────────────────────────
    for i, mask in enumerate(layer_masks):
        if mask.dtype == bool or mask.dtype != np.uint8:
            layer_masks[i] = mask.astype(np.uint8) * 255

    # ── Step 6: 上采样蒙版到原图（sam_driven 已在原图分辨率，跳过）─
    if not sam_driven_mode and (depth_h, depth_w) != (orig_h, orig_w):
        for i, mask in enumerate(layer_masks):
            layer_masks[i] = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        frame_mask_full = cv2.resize(frame_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        # 重新计算统计（原图分辨率）
        stats = []
        for i, mask in enumerate(layer_masks):
            fg = int(np.count_nonzero(mask))
            stats.append({
                "layer_index": i,
                "fg_pixels": fg,
                "fg_pct": round(fg / (orig_h * orig_w) * 100, 2),
                "bridges_built": 0,
                "islands_erased": 0,
            })
    else:
        frame_mask_full = frame_mask

    # ── Step 7: 保存叠加图（原图分辨率） ────────────────────
    suffix = f"_n{n_layers}_f{req.frame_width}_i{req.min_island_area}"
    suffix += "_dr" if req.quality == "draft" else "_std"
    overlay = image.copy()
    colors = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182)]
    for idx, mask in enumerate(layer_masks):
        c = colors[idx % len(colors)]
        fg = mask > 0
        overlay[fg] = (overlay[fg].astype(np.float32) * 0.6
                       + np.array(c, dtype=np.float32) * 0.4).astype(np.uint8)
    overlay_path = SAM_OUTPUT_DIR / f"{stem}{suffix}.png"
    cv2.imwrite(str(overlay_path), overlay)

    # ── Step 8: 保存每层蒙版 ────────────────────────────────
    layers_info = []
    colors_layer = [(231,76,60),(46,204,113),(52,152,219),(241,196,15),(155,89,182)]
    for rank, mask in enumerate(layer_masks):
        mask_path = SAM_OUTPUT_DIR / f"{stem}{suffix}_mask_{rank}.png"
        cv2.imwrite(str(mask_path), mask)
        pure = mask.copy()
        pure[frame_mask_full > 0] = 0
        frame_path = SAM_OUTPUT_DIR / f"{stem}{suffix}_frame_{rank}.png"
        cv2.imwrite(str(frame_path), pure)
        fg_pct = round(np.count_nonzero(mask) / mask.size * 100, 1)
        layers_info.append({
            "layer_index": rank,
            "mask_url": f"/preview/{mask_path.name}",
            "frame_url": f"/preview/{frame_path.name}",
            "color": f"rgb({colors_layer[rank%5][0]},{colors_layer[rank%5][1]},{colors_layer[rank%5][2]})",
            "label": f"图层 {rank+1}",
            "fg_pct": fg_pct,
        })

    scores = score_segmentation(stats, (orig_h, orig_w))

    try:
        features = extract_features(image, depth_map)
    except Exception:
        features = None

    return {
        "image_name": req.image_name,
        "overlay_url": f"/preview/{overlay_path.name}",
        "layers": layers_info,
        "stats": stats,
        "scores": scores,
        "features": features,
        "depth_cached": depth_cached,
        "elapsed_ms": 0,
        "suggested_n_layers": suggested_n,
        "params": {
            "n_layers": n_layers, "frame_width": req.frame_width,
            "min_island_area": req.min_island_area, "quality": req.quality,
        },
    }


# ── 标定管理 ──────────────────────────────────────────────

def save_label(image_name: str, params: dict, scores: dict | None = None, features: dict | None = None):
    """保存标定参数 -> labeled.json（复用 ImageRegistry）。"""
    reg = ImageRegistry()
    img_path = TEST_IMGS_DIR / image_name
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_name}")
    fhash = hash_file(img_path)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    reg.register_image(fhash, image_name, w, h, round(w * h / 1e6, 2), features=features)

    label_data = {
        "n_layers": params["n_layers"],
        "frame_width": params["frame_width"],
        "min_island_area": params["min_island_area"],
        "quality": params.get("quality", "standard"),
        "refine_mode": params.get("refine_mode", "boundary"),
    }

    # 记录笔刷事件数量
    brush_count = _brush_event_count(fhash)
    if brush_count > 0:
        label_data["brush_events_count"] = brush_count

    reg.label_image(fhash, label_data, scores=scores)
    reg.save()
    return {"ok": True, "labeled_count": reg.labeled_count, "total": reg.total, "brush_events": brush_count}


def skip(image_name: str):
    """跳过 -> 不标定。"""
    reg = ImageRegistry()
    img_path = TEST_IMGS_DIR / image_name
    if img_path.exists():
        fhash = hash_file(img_path)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        reg.register_image(fhash, image_name, w, h, round(w * h / 1e6, 2))
        reg.save()
    return {"ok": True, "skipped": image_name}


# ── API 路由 ──────────────────────────────────────────────

@app.get("/api/images")
def api_list():
    return scan_images()

@app.post("/api/segment")
def api_segment(req: SegmentRequest):
    try:
        return run_segmentation(req)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


class AutoSegmentRequest(BaseModel):
    image_name: str

@app.post("/api/auto-segment")
def api_auto_segment(req: AutoSegmentRequest):
    """ML 预测最优参数 -> 自动分割。无预测器时降级为默认参数。"""
    try:
        import pickle
        predictor = None
        pred_path = DATA_DIR / "layer_predictor.pkl"
        if pred_path.exists():
            try:
                with open(pred_path, "rb") as f:
                    predictor = pickle.load(f)
            except Exception:
                pass

        img_path = TEST_IMGS_DIR / req.image_name
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {req.image_name}")
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法解码: {req.image_name}")

        stem = Path(req.image_name).stem
        depth_cache = SAM_OUTPUT_DIR / f"{stem}_depth.npy"
        if depth_cache.exists():
            depth_map = np.load(str(depth_cache))
        else:
            from app.utils.depth_engine import estimate_depth
            depth_map = estimate_depth(image)
            np.save(str(depth_cache), depth_map)
        features = extract_features(image, depth_map)

        if predictor is not None:
            from test_sam_segment import predict_params
            params = predict_params(features, predictor)
        else:
            params = {"n_layers": 3, "frame_width": 50, "min_island_area": 100, "quality": "standard"}

        seg_req = SegmentRequest(
            image_name=req.image_name,
            n_layers=params["n_layers"],
            frame_width=params["frame_width"],
            min_island_area=params["min_island_area"],
            quality=params.get("quality", "standard"),
        )
        result = run_segmentation(seg_req)
        result["predicted_params"] = params
        return result

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/predict-params")
def api_predict_params(req: AutoSegmentRequest):
    """仅预测最优参数，不运行分割。"""
    try:
        import pickle
        predictor = None
        pred_path = DATA_DIR / "layer_predictor.pkl"
        if pred_path.exists():
            with open(pred_path, "rb") as f:
                predictor = pickle.load(f)

        img_path = TEST_IMGS_DIR / req.image_name
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {req.image_name}")
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法解码: {req.image_name}")

        stem = Path(req.image_name).stem
        depth_cache = SAM_OUTPUT_DIR / f"{stem}_depth.npy"
        if depth_cache.exists():
            depth_map = np.load(str(depth_cache))
        else:
            from app.utils.depth_engine import estimate_depth
            depth_map = estimate_depth(image)
            np.save(str(depth_cache), depth_map)
        features = extract_features(image, depth_map)

        if predictor is not None:
            from test_sam_segment import predict_params
            params = predict_params(features, predictor)
        else:
            params = {"n_layers": 3, "frame_width": 50, "min_island_area": 100, "quality": "standard"}

        return {"image_name": req.image_name, "predicted_params": params}

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/save")
def api_save(req: SaveRequest):
    return save_label(req.image_name, {
        "n_layers": req.n_layers, "frame_width": req.frame_width,
        "min_island_area": req.min_island_area, "quality": req.quality,
    }, req.scores, req.features)

@app.post("/api/skip")
def api_skip(req: SkipRequest):
    return skip(req.image_name)


@app.post("/api/brush-refine")
def api_brush_refine(req: BrushRefineRequest):
    """笔刷式 SAM 局部精修。

    接受开发者在当前层蒙版上涂抹的笔刷笔画，
    转换为 SAM point prompts 后运行局部精修。
    """
    try:
        import torch
        from app.utils.sam_engine import _get_sam_model, _preprocess_image
        from mobile_sam import SamPredictor

        # ── 加载原图和当前蒙版 ──
        img_path = TEST_IMGS_DIR / req.image_name
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {req.image_name}")

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法解码: {req.image_name}")

        # 读取当前蒙版
        mask_key = req.current_mask_key
        mask_path = SAM_OUTPUT_DIR / f"{mask_key}_mask_{req.layer_index}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"蒙版不存在: {mask_path}")

        current_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if current_mask is None:
            raise ValueError(f"无法解码蒙版: {mask_path}")
        current_bin = current_mask > 127

        # ── 收集笔刷点 ──
        include_points: list[list[int]] = []
        exclude_points: list[list[int]] = []

        for stroke in req.strokes:
            pts = stroke.points
            if not pts:
                continue
            # 对笔画采样（每条笔画取 3 个均匀点），避免点太多
            n_sample = min(3, len(pts))
            indices = [int(len(pts) * i / n_sample) for i in range(n_sample)]
            sampled = [pts[j] for j in indices if j < len(pts)]

            if stroke.brush_type == "include":
                include_points.extend(sampled)
            elif stroke.brush_type == "exclude":
                exclude_points.extend(sampled)

        if not include_points and not exclude_points:
            raise ValueError("没有有效的笔刷点")

        # ── 计算笔刷覆盖区域的 bbox ──
        all_pts = include_points + exclude_points
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bbox_x1, bbox_y1 = max(0, min(xs) - 30), max(0, min(ys) - 30)
        bbox_x2, bbox_y2 = min(image.shape[1], max(xs) + 30), min(image.shape[0], max(ys) + 30)

        # ── SAM 推理 ──
        model = _get_sam_model()
        sam_image = _preprocess_image(image, max_dim=1200)
        if sam_image.shape[-1] == 3:
            sam_rgb = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        else:
            sam_rgb = sam_image

        predictor = SamPredictor(model)
        predictor.set_image(sam_rgb)

        # 转换坐标到 SAM 空间
        scale_x = sam_rgb.shape[1] / image.shape[1]
        scale_y = sam_rgb.shape[0] / image.shape[0]

        include_sam = [[int(p[0]*scale_x), int(p[1]*scale_y)] for p in include_points]
        exclude_sam = [[int(p[0]*scale_x), int(p[1]*scale_y)] for p in exclude_points]

        point_coords = np.array(include_sam + exclude_sam)
        point_labels = np.array([1]*len(include_sam) + [0]*len(exclude_sam))

        box_sam = np.array([
            int(bbox_x1*scale_x), int(bbox_y1*scale_y),
            int(bbox_x2*scale_x), int(bbox_y2*scale_y),
        ])

        with torch.inference_mode():
            # 只传 box（point_coords 与 box 组合在 MobileSAM 中有维度冲突）
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_sam[None, :],
                multimask_output=False,
            )

        refined_sam = np.asarray(masks[0])
        score = float(scores[0])
        if score < 0.3:
            return {"ok": False, "message": f"SAM 置信度过低: {score:.2f}", "score": score}

        # 缩放回原图分辨率
        import app.utils.sam_engine as se
        refined_full = se._upscale_mask_smooth(
            refined_sam, image.shape[0], image.shape[1],
        )

        # ── 保存精修后的蒙版 ──
        refined_u8 = refined_full.astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), refined_u8)

        # 保存 frame 版本（去除边框的纯内容）
        frame_path = SAM_OUTPUT_DIR / f"{mask_key}_frame_{req.layer_index}.png"
        pure = refined_u8.copy()
        # 边框区域：四边 frame_width 像素
        fw = 50  # 默认边框宽度
        pure[:fw, :] = 0
        pure[-fw:, :] = 0
        pure[:, :fw] = 0
        pure[:, -fw:] = 0
        cv2.imwrite(str(frame_path), pure)

        fg_pct = round(float(np.count_nonzero(refined_u8)) / refined_u8.size * 100, 1)

        # ── 记录笔刷事件 ─────────────────────────────────
        fg_before = round(float(np.count_nonzero(current_bin)) / current_bin.size * 100, 1)
        bbox_tuple = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        total_pts = len(include_points) + len(exclude_points)
        brush_types_in_stroke = list(set(s.brush_type for s in req.strokes))
        primary_type = brush_types_in_stroke[0] if len(brush_types_in_stroke) == 1 else "mixed"

        img_hash = hash_file(img_path)
        try:
            depth_cache = SAM_OUTPUT_DIR / f"{Path(req.image_name).stem}_depth.npy"
            dm = np.load(str(depth_cache)) if depth_cache.exists() else None
        except Exception:
            dm = None

        event_path = _record_brush_event(
            image_hash=img_hash,
            image_name=req.image_name,
            layer_index=req.layer_index,
            brush_type=primary_type,
            point_count=total_pts,
            bbox=bbox_tuple,
            sam_score=score,
            fg_pct_before=fg_before,
            fg_pct_after=fg_pct,
            image=image,
            depth_map=dm,
        )

        return {
            "ok": True,
            "layer_index": req.layer_index,
            "fg_pct": fg_pct,
            "sam_score": score,
            "point_count": total_pts,
        }

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/depth-preview")
def api_depth_preview(image_name: str):
    """返回深度图可视化 PNG。"""
    from fastapi.responses import Response
    stem = Path(image_name).stem
    depth_path = SAM_OUTPUT_DIR / f"{stem}_depth.npy"
    if not depth_path.exists():
        raise HTTPException(404, f"深度图缓存不存在: {stem}_depth.npy")

    depth = np.load(str(depth_path))
    # 归一化到 0-255
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-8:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    # 伪彩色（JET 便于观察深度差异）
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", depth_color)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/config")
def api_config():
    subdirs = [d.name for d in TEST_IMGS_BASE.iterdir() if d.is_dir()]
    return {
        "image_dir": str(TEST_IMGS_DIR.resolve()),
        "active_subdir": TEST_IMGS_DIR.name,
        "available_subdirs": sorted(subdirs),
        "valid_extensions": sorted(IMAGE_EXTENSIONS),
    }

@app.post("/api/switch-dir")
def api_switch_dir(req: SetDirRequest):
    global TEST_IMGS_DIR
    new_dir = TEST_IMGS_BASE / req.image_dir
    if not new_dir.is_dir():
        raise HTTPException(400, f"子目录不存在: {req.image_dir}（可选: train, val）")
    TEST_IMGS_DIR = new_dir.resolve()
    return {"ok": True, "active_subdir": TEST_IMGS_DIR.name, "image_dir": str(TEST_IMGS_DIR)}

# ── 静态文件 ──────────────────────────────────────────────

app.mount("/preview", StaticFiles(directory=str(SAM_OUTPUT_DIR)), name="preview")
app.mount("/images", StaticFiles(directory=str(TEST_IMGS_DIR)), name="images")

STATIC_DIR.mkdir(exist_ok=True)


@app.get("/brush_tool.js")
def brush_tool_js():
    """Serve brush_tool.js explicitly (avoids mount ordering issues)."""
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "brush_tool.js"))


@app.get("/")
def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"Labeler: http://localhost:8090")
    print(f"Test images: {TEST_IMGS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8090)
