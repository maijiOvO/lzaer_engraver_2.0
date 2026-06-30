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
    n_layers: int = Field(default=3, ge=2, le=10)
    frame_width: int = Field(default=50, ge=20, le=200)
    min_island_area: int = Field(default=100, ge=10, le=5000)
    quality: str = Field(default="standard", pattern="^(draft|standard|fine)$")
    refine_mode: str = Field(default="sam_driven", pattern="^(none|slic|sam_driven)$")

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
    frame_width: int = Field(default=50, ge=20, le=200)  # 分割时的外框宽度，用于裁剪frame图


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

    # ── 计算建议层数（仅信息展示，不覆盖用户选择）───
    from app.utils.structural_segmentation import suggest_n_layers
    suggested_n = int(suggest_n_layers(depth_map))
    n_layers = req.n_layers

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

    refine_mode = getattr(req, "refine_mode", "sam_driven")

    # 标志位：sam_driven 模式直接工作在原图分辨率，跳过深度分辨率缩放和后续 SAM 精修
    sam_driven_mode = False

    # ── Step 2: 选择分割算法 ─────────────────────────
    if refine_mode == "sam_driven":
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
            skip_connectivity_repair=True,
        )
        frame_mask_full = frame_mask  # 已是原图分辨率，跳过 Step 6

        # ── GrabCut 边缘吸附：层蒙版沿原图真实边缘对齐 ──
        if req.quality != "draft":
            try:
                from app.utils.sam_engine import _snap_mask_to_edges
                frame_bin = frame_mask_full > 0
                for i, mask in enumerate(layer_masks):
                    if not mask.any():
                        continue
                    content_only = (mask > 0) & ~frame_bin
                    if content_only.sum() < 10:
                        continue
                    try:
                        snapped = _snap_mask_to_edges(content_only, image, edge_band=5)
                        snapped_u8 = snapped.astype(np.uint8) * 255
                        snapped_u8[frame_bin] = 255
                        layer_masks[i] = snapped_u8
                        fg = int(np.count_nonzero(snapped_u8))
                        stats[i]["fg_pixels"] = fg
                        stats[i]["fg_pct"] = round(fg / (orig_h * orig_w) * 100, 2)
                    except Exception:
                        pass
            except ImportError:
                pass
    elif refine_mode == "none":
        # 纯等距量化 + 连通修复（客户端算法）
        from app.utils.structural_segmentation import (
            valley_quantize_depth, generate_frame_mask, repair_layer_mask,
        )
        raw_masks = valley_quantize_depth(depth_map, n_layers=n_layers)
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

    # ── Step 7: 保存叠加图（扩展画布尺寸） ────────────────────
    suffix = f"_n{n_layers}_f{req.frame_width}_i{req.min_island_area}"
    if req.quality == "draft":
        suffix += "_dr"
    elif sam_driven_mode:
        # GrabCut edge refinement (integrated in sam_driven post-processing)
        suffix += "_gc_fin" if req.quality == "fine" else "_gc"
    else:
        # none/slic + SAM refine_mask (internally uses GrabCut too)
        suffix += "_ref"

    # 向外延伸边框 → pad 原图以匹配扩展后的层蒙版
    if sam_driven_mode and frame_mask_full.shape != image.shape[:2]:
        fw = req.frame_width
        image_padded = cv2.copyMakeBorder(
            image, fw, fw, fw, fw, cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        out_h, out_w = image_padded.shape[:2]
    else:
        image_padded = image
        out_h, out_w = image.shape[:2]

    overlay = image_padded.copy()
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
        # frame 图：蒙版中去掉 frame 区域 = 纯内容
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

    scores = score_segmentation(stats, (out_h, out_w))

    try:
        features = extract_features(image, depth_map)
    except Exception:
        features = None

    mask_key = f"{stem}{suffix}"

    return {
        "image_name": req.image_name,
        "overlay_url": f"/preview/{overlay_path.name}",
        "mask_key": mask_key,
        "layers": layers_info,
        "stats": stats,
        "scores": scores,
        "features": features,
        "depth_cached": depth_cached,
        "elapsed_ms": 0,
        "suggested_n_layers": suggested_n,
        "actual_refine_mode": refine_mode,
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
        "refine_mode": params.get("refine_mode", "sam_driven"),
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
    """笔刷式 SAM 局部精修（修订版 — SAM mask_input + 布尔运算）。

    流程：
    1. 笔刷笔画 → 光栅化为原图分辨率粗掩码
    2. 粗掩码缩小到 256×256 → SAM mask_input
    3. SAM predict(mask_input=...) → 像素级精确边界
    4. 精修掩码上采样回原图分辨率 (_upscale_mask_smooth)
    5. 纳入: new_mask = old_mask | SAM精修掩码  （只加不删）
       排除: new_mask = old_mask & ~SAM精修掩码 （只删不加）
    6. 写回蒙版文件 + frame文件 + 笔刷事件记录
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

        orig_h, orig_w = image.shape[:2]

        # 读取当前蒙版
        mask_key = req.current_mask_key
        mask_path = SAM_OUTPUT_DIR / f"{mask_key}_mask_{req.layer_index}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"蒙版不存在: {mask_path}")

        current_mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if current_mask_u8 is None:
            raise ValueError(f"无法解码蒙版: {mask_path}")
        old_mask_bin = current_mask_u8 > 127

        # ── Step 1: 光栅化笔刷笔画为原图分辨率粗掩码 ──
        include_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        exclude_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for stroke in req.strokes:
            pts = stroke.points
            if not pts or len(pts) < 2:
                continue
            # 用 OpenCV 画连续笔画（带线宽）
            draw_canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
            pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(draw_canvas, [pts_array], isClosed=False,
                          color=255, thickness=20,
                          lineType=cv2.LINE_AA)
            if stroke.brush_type == "include":
                include_mask = np.maximum(include_mask, draw_canvas)
            elif stroke.brush_type == "exclude":
                exclude_mask = np.maximum(exclude_mask, draw_canvas)

        # 笔刷膨胀填补间隙
        brush_dilate_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (brush_dilate_size * 2 + 1, brush_dilate_size * 2 + 1))
        include_mask = cv2.dilate(include_mask, kernel, iterations=1)
        exclude_mask = cv2.dilate(exclude_mask, kernel, iterations=1)

        # 决定使用哪份笔刷掩码（多组笔画时按多数类型）
        include_count = sum(1 for s in req.strokes if s.brush_type == "include")
        exclude_count = sum(1 for s in req.strokes if s.brush_type == "exclude")
        primary_brush_type = "include" if include_count >= exclude_count else "exclude"

        if primary_brush_type == "include":
            rough_mask = include_mask
        else:
            rough_mask = exclude_mask

        if rough_mask.sum() == 0:
            raise ValueError("笔刷区域为空，请重新涂抹")

        # ── Step 2: 粗掩码 → 256×256 → SAM mask_input ──
        mask_256 = cv2.resize(rough_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        # SAM 的 mask_input 需要未归一化的 logits：前景=+10，背景=-10
        mask_logits = np.where(mask_256 > 127, 10.0, -10.0).astype(np.float32)
        mask_input = torch.as_tensor(mask_logits, dtype=torch.float32)
        mask_input = mask_input.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]

        # ── Step 3: SAM 推理 ──
        model = _get_sam_model()
        sam_image = _preprocess_image(image, max_dim=1200)
        if sam_image.shape[-1] == 3:
            sam_rgb = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        else:
            sam_rgb = sam_image

        predictor = SamPredictor(model)
        predictor.set_image(sam_rgb)

        # 提取 BBox prompt（约束 SAM 只在笔刷区域搜索，提升精准度）
        all_xs = [p[0] for s in req.strokes for p in s.points if s.points]
        all_ys = [p[1] for s in req.strokes for p in s.points if s.points]
        if all_xs and all_ys:
            x_min, y_min = max(0, min(all_xs)), max(0, min(all_ys))
            x_max, y_max = min(orig_w, max(all_xs)), min(orig_h, max(all_ys))
            x_max = max(x_min + 1, x_max)
            y_max = max(y_min + 1, y_max)
            box_prompt = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        else:
            box_prompt = None

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                mask_input=mask_input,
                multimask_output=False,
            )

        refined_sam = np.asarray(masks[0])
        score = float(scores[0])
        if score < 0.3:
            return {"ok": False, "message": f"SAM 置信度过低: {score:.2f}", "score": score}

        # ── Step 4: SAM predict 返回的 mask 已是原图分辨率，直接使用 ──

        # ── Step 5: 布尔运算 ──
        if primary_brush_type == "include":
            # 纳入：旧蒙版 | SAM精修掩码（只加不删）
            new_mask_bin = old_mask_bin | refined_sam
        else:
            # 排除：旧蒙版 & ~SAM精修掩码（只删不加）
            new_mask_bin = old_mask_bin & ~refined_sam

        new_mask_u8 = new_mask_bin.astype(np.uint8) * 255

        # ── Step 6: 写回蒙版文件 ──
        cv2.imwrite(str(mask_path), new_mask_u8)

        # 保存 frame 版本（去除边框的纯内容，fw 使用请求参数）
        frame_path = SAM_OUTPUT_DIR / f"{mask_key}_frame_{req.layer_index}.png"
        fw = req.frame_width
        pure = new_mask_u8.copy()
        pure[:fw, :] = 0
        pure[-fw:, :] = 0
        pure[:, :fw] = 0
        pure[:, -fw:] = 0
        cv2.imwrite(str(frame_path), pure)

        fg_pct = round(float(np.count_nonzero(new_mask_u8)) / new_mask_u8.size * 100, 1)

        # ── 笔刷事件记录 ─────────────────────────────────
        fg_before = round(float(np.count_nonzero(old_mask_bin)) / old_mask_bin.size * 100, 1)
        total_pts = sum(len(s.points) for s in req.strokes)

        # 笔刷覆盖bbox
        all_xs = [p[0] for s in req.strokes for p in s.points if s.points]
        all_ys = [p[1] for s in req.strokes for p in s.points if s.points]
        if all_xs and all_ys:
            bbox_tuple = (max(0, min(all_xs)), max(0, min(all_ys)),
                          min(orig_w, max(all_xs)), min(orig_h, max(all_ys)))
        else:
            bbox_tuple = (0, 0, orig_w, orig_h)

        img_hash = hash_file(img_path)
        try:
            depth_cache_path = SAM_OUTPUT_DIR / f"{Path(req.image_name).stem}_depth.npy"
            dm = np.load(str(depth_cache_path)) if depth_cache_path.exists() else None
        except Exception:
            dm = None

        event_path = _record_brush_event(
            image_hash=img_hash,
            image_name=req.image_name,
            layer_index=req.layer_index,
            brush_type=primary_brush_type,
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
            "brush_type": primary_brush_type,
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
