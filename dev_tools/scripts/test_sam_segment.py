#!/usr/bin/env python3
"""SAM 分割步骤 — 三合一开发者效率工具。

scan (默认): 扫描 test_imgs/ → 状态报告 → 新图自动预测+分割
label:       交互标定 (Y=确认/n=放弃/e=改参/q=退出)，深度缓存命中 <1s 重跑
train:       RandomForest 预测器训练 (≥10 标定样本)
search:      图片搜索 (接口就位，待 API 配置)

用法:
    # 扫描模式 (默认)
    python3 test_sam_segment.py
    python3 test_sam_segment.py --image city_street.jpg --quality draft

    # 标定模式
    python3 test_sam_segment.py --mode label --image 上海.jpg
    python3 test_sam_segment.py --mode label --image 上海.jpg --n-layers 4 --frame 30

    # 训练模式
    python3 test_sam_segment.py --mode train

    # 搜索模式
    python3 test_sam_segment.py --mode search --query "papercut layered art"

架构 (2026-07-02):
    ImageRegistry (SHA256 去重+状态管理) → extract_features (12维)
    → RandomForest 预测器 → scan/label/train/search 四模式 CLI
    核心管线: Depth-Anything-V2 → 等距量化 → 外框+连通修复(策略C) → [可选] SAM 精修
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── 项目根路径注入 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── 评分引擎（dev_tools 内部模块）─────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "dev_tools" / "labeler"))
from score_engine import score_segmentation

# ── 目录 ─────────────────────────────────────────────────────────
TEST_IMGS_DIR = PROJECT_ROOT / "dev_tools" / "test_imgs"
OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "sam"
DATA_DIR = PROJECT_ROOT / "dev_tools" / "data"
REGISTRY_PATH = DATA_DIR / "labeled.json"
PREDICTOR_PATH = DATA_DIR / "layer_predictor.pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 叠加图层颜色 (BGR, 最多 5 层) ─────────────────────────────────
LAYER_COLORS = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
]


def render_overlay(
    image: np.ndarray,
    layer_masks: list[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """将每层蒙版以不同颜色叠加到原图上。"""
    h, w = image.shape[:2]
    canvas = image.copy().astype(np.float32)

    for i, mask in enumerate(layer_masks):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        color_arr = np.array(color, dtype=np.float32)
        fg = mask > 0
        canvas[fg] = canvas[fg] * (1 - alpha) + color_arr * alpha

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
#  图片注册表 & 特征提取
# ══════════════════════════════════════════════════════════════════

def hash_file(path: Path) -> str:
    """SHA256 文件内容哈希"""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


class ImageRegistry:
    """SHA256 图片注册表 — 管理标定状态、去重、训练数据导出。

    三种状态:
      labeled  — params 有值, labeled_at 有值  (可用于训练)
      pending  — params null, labeled_at null    (特征已提取，待开发者确认)
      new      — 不在注册表中                    (test_imgs/ 新增)
    """

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self.data: dict[str, Any] = {"version": 2, "images": {}}
        self._load()

    # ── 持久化 ─────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            if not self._validate(data):
                raise ValueError("Schema validation failed")
            self.data = data
        except Exception:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            bak = self.path.with_suffix(f".json.bak.{ts}")
            self.path.rename(bak)
            print(f"⚠ labeled.json 损坏，已备份至 {bak.name}，重建空注册表")
            self.data = {"version": 1, "images": {}}

    def _validate(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        version = data.get("version")
        if version not in (1, 2):
            return False
        images = data.get("images")
        if not isinstance(images, dict):
            return False
        for entry in images.values():
            if not isinstance(entry, dict):
                return False
            for k in ("filename", "width", "height", "megapixels"):
                if k not in entry:
                    return False
        # v1 → v2 自动迁移：version 字段升级，数据不变
        if version == 1:
            data["version"] = 2
        return True

    def save(self) -> None:
        labeled = sum(1 for e in self.data["images"].values() if e.get("params") is not None)
        pending = sum(1 for e in self.data["images"].values() if e.get("params") is None)
        self.data["labeled_count"] = labeled
        self.data["pending_count"] = pending
        self.data["total_processed"] = labeled + pending
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(self.path))

    # ── 扫描 ───────────────────────────────────────────────────

    def scan_test_imgs(self, img_dir: Path) -> dict[str, list[dict]]:
        """扫描目录，返回状态分类。

        Returns:
            {"labeled": [...], "pending": [...], "new": [...], "duplicates": [...]}
        """
        result: dict[str, list[dict]] = {
            "labeled": [], "pending": [], "new": [], "duplicates": [],
        }
        seen_hashes: dict[str, str] = {}  # hash → first_filename

        if not img_dir.is_dir():
            return result

        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue

            try:
                fhash = hash_file(f)
            except OSError:
                continue

            # 去重检查（本次扫描内 + 注册表中）
            if fhash in seen_hashes:
                result["duplicates"].append({
                    "filename": f.name,
                    "hash": fhash,
                    "duplicate_of": seen_hashes[fhash],
                })
                continue

            entry = self.data["images"].get(fhash)
            if entry is not None:
                if f.name != entry["filename"]:
                    # 重命名了，更新 filename
                    entry["filename"] = f.name
                seen_hashes[fhash] = f.name
                if entry.get("params") is not None:
                    result["labeled"].append({"filename": f.name, "hash": fhash, **entry})
                else:
                    result["pending"].append({"filename": f.name, "hash": fhash, **entry})
            else:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                seen_hashes[fhash] = f.name
                result["new"].append({
                    "filename": f.name, "hash": fhash,
                    "width": w, "height": h,
                    "megapixels": round(w * h / 1e6, 2),
                })

        return result

    # ── 注册 & 标定 ────────────────────────────────────────────

    def register_image(
        self, filehash: str, filename: str,
        width: int, height: int, megapixels: float,
        features: dict[str, Any] | None = None,
    ) -> None:
        """将新图加入注册表（状态: pending）。"""
        self.data["images"][filehash] = {
            "filename": filename,
            "width": width,
            "height": height,
            "megapixels": megapixels,
            "features": features,
            "params": None,
            "labeled_at": None,
        }

    def label_image(self, filehash: str, params: dict[str, Any], scores: dict[str, Any] | None = None) -> None:
        """标定一张图（状态: pending → labeled）。"""
        entry = self.data["images"].get(filehash)
        if entry is None:
            raise KeyError(f"哈希 {filehash[:12]}… 不在注册表中")
        entry["params"] = params
        entry["labeled_at"] = datetime.now(timezone.utc).isoformat()
        if scores:
            entry["scores"] = scores

    # ── 训练数据导出 ───────────────────────────────────────────

    def get_labeled_dataset(self, min_quality: str = "any") -> list[dict[str, Any]]:
        """返回可用于训练的标定数据集。

        Args:
            min_quality: 最低质量要求。
                "any"     — 所有标定数据（含自动标定）
                "reviewed" — 仅含有人工审核的数据（有 brush_events_count 或通过 CLI label Y 确认）
        """
        dataset = [
            e for e in self.data["images"].values()
            if e.get("params") is not None and e.get("features") is not None
        ]
        if min_quality == "reviewed":
            dataset = [
                e for e in dataset
                if e.get("params", {}).get("brush_events_count", 0) > 0
                   or e.get("params", {}).get("refine_mode") is not None
            ]
        return dataset

    def get_train_val_split(self) -> tuple[list, list]:
        """按 test_imgs/train/ 和 test_imgs/val/ 物理目录分离训练/验证集。"""
        train_dir = PROJECT_ROOT / "dev_tools" / "test_imgs" / "train"
        val_dir = PROJECT_ROOT / "dev_tools" / "test_imgs" / "val"
        train_set = []
        val_set = []
        for e in self.data["images"].values():
            if e.get("params") is None or e.get("features") is None:
                continue
            fname = e.get("filename", "")
            if (val_dir / fname).exists():
                val_set.append(e)
            else:
                train_set.append(e)
        return train_set, val_set

    @property
    def labeled_count(self) -> int:
        return sum(1 for e in self.data["images"].values() if e.get("params") is not None)

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self.data["images"].values()
                   if e.get("params") is None and e.get("features") is not None)

    @property
    def total(self) -> int:
        return len(self.data["images"])


# ── 特征提取 ─────────────────────────────────────────────────────

def extract_features(
    image: np.ndarray,
    depth_map: np.ndarray,
    quantize_fn: Any = None,
) -> dict[str, Any]:
    """从图像 + 深度图提取 11 维特征。

    Args:
        image: BGR 图像 (H, W, 3)
        depth_map: 深度图 (H, W) float32
        quantize_fn: 可选，quantize_depth 函数引用（用于 layer_area_variance）

    Returns:
        dict with keys: depth_entropy, depth_peaks, depth_range,
        depth_mean, depth_std, edge_density, texture_variance,
        width, height, megapixels, aspect_ratio,
        layer_area_variance_3/4/5 (if quantize_fn)
    """
    h, w = image.shape[:2]
    features: dict[str, Any] = {
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1e6, 2),
        "aspect_ratio": round(w / h, 3),
    }

    # ── 深度特征 ──────────────────────────────────────────────
    d_flat = depth_map.ravel().astype(np.float64)
    features["depth_range"] = round(float(d_flat.max() - d_flat.min()), 4)
    features["depth_mean"] = round(float(d_flat.mean()), 4)
    features["depth_std"] = round(float(d_flat.std()), 4)

    # 深度熵 (100-bin 直方图)
    hist, _ = np.histogram(d_flat, bins=100)
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
        hist = hist[hist > 0]
        features["depth_entropy"] = round(float(-np.sum(hist * np.log2(hist))), 4)
    else:
        features["depth_entropy"] = 0.0

    # 深度峰数 (= 局部极大值, 高度 > 5% 总高度)
    hist_full, _ = np.histogram(d_flat, bins=100)
    threshold = hist_full.max() * 0.05
    peaks = 0
    for i in range(1, 99):
        if hist_full[i] > threshold and hist_full[i] > hist_full[i - 1] and hist_full[i] > hist_full[i + 1]:
            peaks += 1
    features["depth_peaks"] = peaks

    # ── 图像特征 ──────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘密度 (Canny low=50, high=150)
    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = round(float(np.count_nonzero(edges) / edges.size), 4)

    # 纹理方差 (16×16 窗口局部方差的中位数)
    gray_f = gray.astype(np.float32)
    kernel = np.ones((16, 16), np.float32) / 256.0
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    local_sq_mean = cv2.filter2D(gray_f * gray_f, -1, kernel)
    local_var = local_sq_mean - local_mean * local_mean
    valid_var = local_var[local_var > 0]
    features["texture_variance"] = round(float(np.median(valid_var)) if valid_var.size > 0 else 0.0, 4)

    # ── 层面极方差 (等距量化 N 层) ────────────────────────────
    if quantize_fn is not None:
        for n in [3, 4, 5]:
            try:
                masks = quantize_fn(depth_map, n_layers=n)
                areas = [np.count_nonzero(m) for m in masks]
                total_a = sum(areas)
                if total_a > 0:
                    normed = [a / total_a for a in areas]
                    features[f"layer_area_variance_{n}"] = round(float(np.var(normed)), 6)
                else:
                    features[f"layer_area_variance_{n}"] = 0.0
            except Exception:
                features[f"layer_area_variance_{n}"] = 0.0

    return features


# ══════════════════════════════════════════════════════════════════
#  ML 预测器 — RandomForest 训练 & 参数预测
# ══════════════════════════════════════════════════════════════════

# 特征顺序（必须与 extract_features 输出一致）
FEATURE_ORDER = [
    "depth_entropy", "depth_peaks", "depth_range", "depth_mean", "depth_std",
    "edge_density", "texture_variance",
    "width", "height", "megapixels", "aspect_ratio",
    "layer_area_variance_3", "layer_area_variance_4", "layer_area_variance_5",
]

# 默认参数（预测器不可用时的回退）
DEFAULT_PARAMS: dict[str, Any] = {
    "n_layers": 3,
    "frame_width": 50,
    "min_island_area": 100,
    "quality": "standard",
}

# 参数边界
PARAM_BOUNDS = {
    "n_layers": (2, 5),
    "frame_width": (20, 200),
    "min_island_area": (10, 5000),
}
QUALITY_OPTIONS = ["draft", "standard", "fine"]


def _check_sklearn() -> bool:
    """检查 sklearn 是否可用。"""
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        print("❌ scikit-learn 未安装")
        print("   安装: pip install scikit-learn --break-system-packages")
        return False


def _features_to_matrix(
    dataset: list[dict[str, Any]],
    feature_order: list[str],
) -> np.ndarray:
    """将标定数据集转为特征矩阵。缺失特征填 0。"""
    rows = []
    for entry in dataset:
        feats = entry.get("features", {})
        row = [feats.get(k, 0.0) for k in feature_order]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def train_predictor(
    labeled_dataset: list[dict[str, Any]],
    min_samples: int = 10,
) -> dict[str, Any] | None:
    """训练参数预测器。

    Args:
        labeled_dataset: get_labeled_dataset() 的返回值
        min_samples: 最少标定样本数

    Returns:
        {"models": {...}, "feature_order": [...], "training_X": ndarray} 或 None
    """
    if not _check_sklearn():
        return None

    n = len(labeled_dataset)
    if n < min_samples:
        print(f"❌ 标定样本不足: {n} < {min_samples}，拒绝训练")
        return None

    # 评分驱动：筛选高分样本 (combined_score >= 0.8)
    scored = [e for e in labeled_dataset if e.get("scores", {}).get("combined_score", 0) >= 0.8]
    has_scores = any(e.get("scores") for e in labeled_dataset)
    if has_scores:
        if len(scored) >= min_samples:
            print(f"📊 评分筛选: {len(scored)}/{n} 张优质样本 (combined_score ≥ 0.8)")
            labeled_dataset = scored
            n = len(scored)
        else:
            print(f"⚠ 优质样本不足 ({len(scored)}/{n}), 降级使用全部 {n} 张")

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import r2_score, accuracy_score

    # 训练/验证分离（基于物理目录）
    train_set, val_set = ImageRegistry().get_train_val_split()
    if val_set:
        print(f"\n📂 训练/验证分离: train={len(train_set)} val={len(val_set)}")
        # 只从训练集构建特征矩阵
        used_dataset = [e for e in labeled_dataset if e in train_set] or labeled_dataset
    else:
        print(f"\n📂 无 val/ 目录，使用全部样本训练")
        used_dataset = labeled_dataset

    X = _features_to_matrix(labeled_dataset, FEATURE_ORDER)
    X = _features_to_matrix(used_dataset, FEATURE_ORDER)
    used_cols = [i for i, k in enumerate(FEATURE_ORDER)
                 if any(e.get("features", {}).get(k) is not None for e in used_dataset)]
    X_used = X[:, used_cols]
    used_order = [FEATURE_ORDER[i] for i in used_cols]
    feature_dim = X_used.shape[1]

    print(f"\n📊 读取 labeled.json …")
    print(f"   已标定: {n} 张")
    print(f"   训练集: {len(used_dataset)} 张")
    print(f"   特征维度: {feature_dim}")

    models: dict[str, Any] = {}

    # ── n_layers 回归器 ───────────────────────────────────────
    y_n = np.array([e["params"]["n_layers"] for e in used_dataset], dtype=np.float64)
    rf_n = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf_n.fit(X_used, y_n)
    models["n_layers"] = rf_n
    print(f"\n🎯 训练 n_layers 回归器 …")
    print(f"   OOB R²: {rf_n.oob_score_:.3f}")

    # ── frame_width 回归器 ────────────────────────────────────
    y_f = np.array([e["params"]["frame_width"] for e in used_dataset], dtype=np.float64)
    rf_f = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf_f.fit(X_used, y_f)
    models["frame_width"] = rf_f
    print(f"🎯 训练 frame_width 回归器 …")
    print(f"   OOB R²: {rf_f.oob_score_:.3f}")

    # ── min_island_area 回归器 ────────────────────────────────
    y_i = np.array([e["params"]["min_island_area"] for e in used_dataset], dtype=np.float64)
    rf_i = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf_i.fit(X_used, y_i)
    models["min_island_area"] = rf_i
    print(f"🎯 训练 min_island_area 回归器 …")
    print(f"   OOB R²: {rf_i.oob_score_:.3f}")

    # ── quality 分类器 ────────────────────────────────────────
    y_q = np.array([QUALITY_OPTIONS.index(e["params"]["quality"]) for e in used_dataset])
    rf_q = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf_q.fit(X_used, y_q)
    models["quality"] = rf_q
    print(f"🎯 训练 quality 分类器 …")
    print(f"   OOB 准确率: {rf_q.oob_score_:.1%}")

    # ── 特征重要性 ────────────────────────────────────────────
    print_feature_importance(rf_n, used_order, "n_layers")

    # ── 验证集评估 ────────────────────────────────────────────
    if val_set:
        print(f"\n🧪 验证集评估 ({len(val_set)} 张):")
        val_features = _features_to_matrix(val_set, used_order)
        # 只取 used_cols 对应的列
        all_X = _features_to_matrix(val_set, FEATURE_ORDER)
        val_X = all_X[:, used_cols]

        for name, model, is_classifier in [
            ("n_layers", rf_n, False),
            ("frame_width", rf_f, False),
            ("min_island_area", rf_i, False),
            ("quality", rf_q, True),
        ]:
            y_true = np.array([e["params"][name] for e in val_set])
            y_pred = model.predict(val_X)
            if is_classifier:
                y_true_idx = np.array([QUALITY_OPTIONS.index(q) for q in y_true])
                acc = accuracy_score(y_true_idx, y_pred)
                print(f"   {name}: 验证准确率 = {acc:.1%}")
            else:
                r2 = r2_score(y_true, y_pred)
                mae = np.abs(y_true - y_pred).mean()
                print(f"   {name}: 验证 R² = {r2:.3f}  MAE = {mae:.1f}")

    return {
        "models": models,
        "feature_order": used_order,
        "training_X": X_used,
    }


def predict_params(
    features: dict[str, Any],
    predictor: dict[str, Any] | None,
) -> dict[str, Any]:
    """预测参数。

    Args:
        features: extract_features() 的输出
        predictor: train_predictor() 的输出，或 None

    Returns:
        {"n_layers": int, "frame_width": int, "min_island_area": int, "quality": str}
    """
    if predictor is None:
        return dict(DEFAULT_PARAMS)

    models = predictor["models"]
    feat_order = predictor["feature_order"]
    X_train = predictor.get("training_X")

    # 构建输入向量
    x = np.array([[features.get(k, 0.0) for k in feat_order]], dtype=np.float64)

    # ── 马氏距离异常检测 ─────────────────────────────────────
    if X_train is not None and X_train.shape[0] > 1:
        try:
            mean = X_train.mean(axis=0)
            cov = np.cov(X_train, rowvar=False)
            # 正则化协方差防止奇异
            cov += np.eye(cov.shape[0]) * 1e-6
            diff = x - mean
            mahal = float(np.sqrt(diff @ np.linalg.inv(cov) @ diff.T))
            # 卡方分布 95% 分位数 ~ sqrt(2*dim) 近似
            threshold = np.sqrt(2 * X_train.shape[1]) * 3
            if mahal > threshold:
                print(f"⚠ 特征显著偏离训练集 (马氏距离={mahal:.1f} > {threshold:.1f})，"
                      f"降级为默认参数")
                return dict(DEFAULT_PARAMS)
        except Exception:
            pass  # 协方差计算失败，跳过检测

    # ── 预测 ─────────────────────────────────────────────────
    params: dict[str, Any] = {}

    for key in ("n_layers", "frame_width", "min_island_area"):
        if key in models:
            val = float(models[key].predict(x)[0])
            lo, hi = PARAM_BOUNDS[key]
            params[key] = int(round(np.clip(val, lo, hi)))

    if "quality" in models:
        idx = int(models["quality"].predict(x)[0])
        params["quality"] = QUALITY_OPTIONS[min(idx, len(QUALITY_OPTIONS) - 1)]
    else:
        params["quality"] = DEFAULT_PARAMS["quality"]

    return params


def print_feature_importance(
    model: Any,
    feature_names: list[str],
    label: str,
) -> None:
    """打印特征重要性排名。"""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n📊 特征重要性 ({label}):")
    max_bar = 20
    for rank, idx in enumerate(indices):
        imp = importances[idx]
        if imp < 0.001:
            break
        bar = "█" * int(imp / importances[indices[0]] * max_bar)
        print(f"   {rank + 1}. {feature_names[idx]:<30s} {imp:.3f}  {bar}")


# ── 命令行参数 ───────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="SAM 分割步骤 — 开发者效率工具 (scan / label / train / search)",
    )
    # 模式
    p.add_argument("--mode", choices=["scan", "label", "train", "search"],
                   default="scan", help="运行模式 (默认: scan)")
    # 图片
    p.add_argument("--image", type=str, default=None,
                   help="指定图片 (scan/label 模式)")
    # 分割参数 (None = 自动预测/使用预测器)
    p.add_argument("--n-layers", type=int, default=None,
                   help="目标层数 (2-5)")
    p.add_argument("--frame", type=int, default=None,
                   help="外框宽度 px")
    p.add_argument("--min-island", type=int, default=None,
                   help="孤立岛丢弃阈值 px²")
    p.add_argument("--quality", choices=["draft", "standard", "fine"],
                   default=None, help="SAM 精修模式")
    # 选项
    p.add_argument("--no-cache", action="store_true",
                   help="强制重算深度，忽略缓存")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="叠加图透明度 (0-1, 默认 0.4)")
    p.add_argument("--refine", choices=["none", "boundary", "sam_driven"],
                   default="boundary", help="分割精修模式 (默认: boundary | sam_driven: SAM自动分割→深度归属)")
    # 搜索 (search 模式)
    p.add_argument("--query", type=str, default=None,
                   help="搜索关键词 (search 模式)")
    p.add_argument("--num", type=int, default=5,
                   help="下载数量 (search 模式, 1-10)")
    p.add_argument("--size", type=str, default="large",
                   choices=["icon", "small", "medium", "large", "xlarge", "xxlarge", "huge"],
                   help="图片尺寸过滤 (search 模式)")
    p.add_argument("--type", dest="img_type", type=str, default="photo",
                   choices=["clipart", "face", "lineart", "photo", "animated"],
                   help="图片类型过滤 (search 模式)")
    return p.parse_args()


# ── 引擎导入 ─────────────────────────────────────────────────────
def _import_engines():
    """惰性导入引擎，缺依赖时给出明确安装提示。"""
    imports: dict = {}

    # depth engine
    try:
        from app.utils.depth_engine import estimate_depth
        imports["estimate_depth"] = estimate_depth
    except ImportError as e:
        print(f"❌ 深度引擎导入失败: {e}")
        print("   请确保已安装: pip install transformers torch Pillow")
        sys.exit(1)

    # structural segmentation
    try:
        from app.utils.structural_segmentation import (
            build_structural_layers,
            quantize_depth,
            generate_frame_mask,
            repair_layer_mask,
        )
        imports["build_structural_layers"] = build_structural_layers
        imports["quantize_depth"] = quantize_depth
        imports["generate_frame_mask"] = generate_frame_mask
        imports["repair_layer_mask"] = repair_layer_mask
    except ImportError as e:
        print(f"❌ 结构分层引擎导入失败: {e}")
        print("   请检查 app/utils/structural_segmentation.py 是否存在")
        sys.exit(1)

    # sam engine (可选 — 仅 standard/fine 需要)
    try:
        from app.utils.sam_engine import refine_mask
        imports["refine_mask"] = refine_mask
    except ImportError as e:
        imports["refine_mask"] = None
        print("⚠ SAM 精修引擎不可用 (refine_mask 导入失败)")
        print(f"   原因: {e}")

    # boundary refine engine (可选 — 仅 --refine boundary 需要)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "dev_tools" / "labeler"))
        from boundary_refine import refine_layers
        imports["refine_layers"] = refine_layers
    except ImportError:
        imports["refine_layers"] = None
        print("⚠ 边界精修引擎不可用 (boundary_refine 导入失败)")

    # SAM 自动分割引擎 (可选 — 仅 --refine sam_driven 需要)
    try:
        from app.utils.sam_engine import run_sam_automatic
        imports["run_sam_automatic"] = run_sam_automatic
    except ImportError as e:
        imports["run_sam_automatic"] = None
        print(f"⚠ SAM自动分割引擎不可用 (run_sam_automatic 导入失败): {e}")

    try:
        from app.utils.structural_segmentation import build_sam_driven_layers
        imports["build_sam_driven_layers"] = build_sam_driven_layers
    except ImportError as e:
        imports["build_sam_driven_layers"] = None
        print(f"⚠ SAM驱动分层引擎不可用 (build_sam_driven_layers 导入失败): {e}")

    # loguru (引擎内部使用)
    try:
        import loguru  # noqa: F401
    except ImportError:
        print("⚠ loguru 未安装，引擎日志可能缺失")
        print("   安装: pip install loguru")

    return imports


# ── 深度缓存 ─────────────────────────────────────────────────────
def depth_cache_path(stem: str) -> Path:
    return OUTPUT_DIR / f"{stem}_depth.npy"


def load_depth_cache(stem: str) -> np.ndarray | None:
    p = depth_cache_path(stem)
    if not p.exists():
        return None
    try:
        depth = np.load(p)
        print(f"   📦 深度缓存命中: {p.name}")
        return depth
    except Exception:
        print(f"   ⚠ 深度缓存损坏，将重新推理")
        return None


def save_depth_cache(stem: str, depth: np.ndarray):
    p = depth_cache_path(stem)
    tmp = str(p) + ".tmp.npy"
    np.save(tmp, depth)
    os.replace(tmp, str(p))


# ── 查找图片 ─────────────────────────────────────────────────────
def find_image(image_arg: str | None) -> tuple[Path, str]:
    """返回 (path, stem)。"""
    if image_arg:
        p = Path(image_arg)
        if not p.is_absolute():
            p = TEST_IMGS_DIR / p
    else:
        # 取 test_imgs/ 下第一个图片
        p = None
        if TEST_IMGS_DIR.is_dir():
            for f in sorted(TEST_IMGS_DIR.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    p = f
                    break
        if p is None:
            print("❌ dev_tools/test_imgs/ 中没有图片，请放入测试图片或用 --image 指定")
            sys.exit(1)

    if not p.exists():
        print(f"❌ 图片不存在: {p}")
        sys.exit(1)

    return p, p.stem


# ══════════════════════════════════════════════════════════════════
#  核心分割管线 (从原 main 提取)
# ══════════════════════════════════════════════════════════════════

def _resolve_params(args, features=None, predictor=None):
    """解析参数优先级: CLI 手动指定 > 预测器 > 默认值。"""
    params = {}
    if predictor is not None and features is not None:
        pred = predict_params(features, predictor)
    else:
        pred = dict(DEFAULT_PARAMS)

    params["n_layers"] = args.n_layers if args.n_layers is not None else pred["n_layers"]
    params["frame_width"] = args.frame if args.frame is not None else pred["frame_width"]
    params["min_island_area"] = args.min_island if args.min_island is not None else pred["min_island_area"]
    params["quality"] = args.quality if args.quality is not None else pred["quality"]
    return params


def _generate_candidates(base_params: dict) -> list[dict]:
    """基于预测参数生成 3 组候选（含基准）。"""
    candidates = [dict(base_params)]
    n = base_params["n_layers"]
    f = base_params["frame_width"]
    i = base_params["min_island_area"]

    # 变体 1: ±1 层（clamp 2-5）
    if n < 5:
        candidates.append({"n_layers": n + 1, "frame_width": f,
                          "min_island_area": i, "quality": base_params["quality"]})
    elif n > 2:
        candidates.append({"n_layers": n - 1, "frame_width": f,
                          "min_island_area": i, "quality": base_params["quality"]})

    # 变体 2: 调 frame_width ±20%
    alt_frame = int(f * 1.2) if n <= 3 else max(20, int(f * 0.8))
    alt_frame = max(20, min(200, alt_frame))
    if alt_frame != f:
        candidates.append({"n_layers": n, "frame_width": alt_frame,
                          "min_island_area": i, "quality": base_params["quality"]})

    return candidates[:3]  # 最多 3 组


def _run_segmentation(
    img_path: Path, stem: str, image: np.ndarray,
    params: dict[str, Any], args, engines: dict,
) -> tuple[list, np.ndarray, list, float, float, float]:
    """执行核心分割管线，返回 (layer_masks, frame_mask, layer_stats, depth_ms, layer_ms, refine_ms)。"""
    estimate_depth = engines["estimate_depth"]
    build_structural_layers = engines["build_structural_layers"]
    refine_mask = engines.get("refine_mask")

    h, w = image.shape[:2]

    # 1. 深度估计
    depth_t0 = time.perf_counter()
    depth_map = None
    if not args.no_cache:
        depth_map = load_depth_cache(stem)
    if depth_map is not None:
        depth_ms = 0.0
    else:
        depth_map = estimate_depth(image)
        depth_ms = (time.perf_counter() - depth_t0) * 1000
        try:
            save_depth_cache(stem, depth_map)
        except Exception:
            pass

    # 2. 结构分层
    layer_t0 = time.perf_counter()

    if args.refine == "boundary" and engines.get("refine_layers") is not None:
        # ── 边界精修模式 ──
        from app.utils.structural_segmentation import quantize_depth, generate_frame_mask

        raw_masks = quantize_depth(depth_map, n_layers=params["n_layers"])
        layer_masks, bf_stats = engines["refine_layers"](
            image, raw_masks, depth_map,
            band_width=5, min_component_area=100, box_padding=10,
        )
        frame_mask = generate_frame_mask(depth_map.shape[0], depth_map.shape[1],
                                          frame_width=params["frame_width"])
        frame_bin = frame_mask > 0
        for mask in layer_masks:
            mask[frame_bin] = 255

        layer_stats = []
        for i, mask in enumerate(layer_masks):
            fg = int(np.count_nonzero(mask))
            bf_s = bf_stats[i] if i < len(bf_stats) else {}
            layer_stats.append({
                "layer_index": i,
                "fg_pixels": fg,
                "fg_pct": round(fg / (depth_map.shape[0] * depth_map.shape[1]) * 100, 2),
                "bridges_built": 0,
                "islands_erased": 0,
            })
        refine_ms = 0.0  # boundary refine 已包含 SAM，不需要额外 refine_mask
    elif args.refine == "sam_driven" and engines.get("run_sam_automatic") is not None:
        # ── SAM驱动模式：SAM 自动分割 → 深度中位数归属 → 连通修复 ──
        sam_cache = OUTPUT_DIR / f"{stem}_sam_region.npz"
        sam_masks, _region_map = engines["run_sam_automatic"](image, cache_path=str(sam_cache))
        layer_masks, frame_mask, layer_stats = engines["build_sam_driven_layers"](
            sam_masks, depth_map,
            n_layers=params["n_layers"],
            image_shape=(h, w),
            frame_width=params["frame_width"],
            min_island_area=params["min_island_area"],
        )
        refine_ms = 0.0  # SAM 已在 Step 2 完成，不需要额外 refine_mask
    else:
        # ── 原始模式（等距量化 + 连通修复 + 可选 SAM 逐层精修）──
        layer_masks, frame_mask, layer_stats = build_structural_layers(
            depth_map,
            n_layers=params["n_layers"],
            frame_width=params["frame_width"],
            min_island_area=params["min_island_area"],
        )
        # SAM per-layer refine
        refine_ms = 0.0
        if params["quality"] != "draft" and engines.get("refine_mask") is not None:
            refine_t0 = time.perf_counter()
            edge_band = 5 if params["quality"] == "fine" else 3
            for i, mask in enumerate(layer_masks):
                if not mask.any():
                    continue
                try:
                    refined = engines["refine_mask"](image, mask, edge_band=edge_band)
                    refined_u8 = refined.astype(np.uint8) * 255
                    refined_u8[frame_mask > 0] = 255
                    layer_masks[i] = refined_u8
                except Exception:
                    pass
            refine_ms = (time.perf_counter() - refine_t0) * 1000

    layer_ms = (time.perf_counter() - layer_t0) * 1000

    return layer_masks, frame_mask, layer_stats, depth_ms, layer_ms, refine_ms


def _save_outputs(stem: str, image, layer_masks, frame_mask, params, args):
    """保存叠加图、蒙版、纯内容。返回叠加图路径。"""
    quality_code = {"draft": "dr", "standard": "std", "fine": "fin"}[params["quality"]]
    file_tag = f"n{params['n_layers']}_f{params['frame_width']}_i{params['min_island_area']}_{quality_code}"

    overlay = render_overlay(image, layer_masks, alpha=args.alpha)
    overlay_name = f"{stem}_{file_tag}.png"
    overlay_path = OUTPUT_DIR / overlay_name
    cv2.imwrite(str(overlay_path), overlay)
    print(f"💾 叠加图: {overlay_name}")

    for i, mask in enumerate(layer_masks):
        mask_name = f"{stem}_{file_tag}_mask{i}.png"
        cv2.imwrite(str(OUTPUT_DIR / mask_name), mask)
        pure = mask.copy()
        pure[frame_mask > 0] = 0
        frame_name = f"{stem}_{file_tag}_frame{i}.png"
        cv2.imwrite(str(OUTPUT_DIR / frame_name), pure)

    return overlay_path


# ══════════════════════════════════════════════════════════════════
#  预测器持久化
# ══════════════════════════════════════════════════════════════════

def _load_predictor() -> dict[str, Any] | None:
    """尝试加载预测器。"""
    import pickle
    if not PREDICTOR_PATH.exists():
        return None
    try:
        with open(PREDICTOR_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_predictor(predictor: dict[str, Any]) -> None:
    """保存预测器。"""
    import pickle
    PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(PREDICTOR_PATH) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(predictor, f)
    os.replace(tmp, str(PREDICTOR_PATH))
    print(f"📦 保存: {PREDICTOR_PATH.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════
#  模式: scan (默认)
# ══════════════════════════════════════════════════════════════════

def cmd_scan(args, engines):
    """扫描 test_imgs/，状态报告 + 新图自动预测分割。"""
    reg = ImageRegistry()
    result = reg.scan_test_imgs(TEST_IMGS_DIR)
    n_labeled = len(result["labeled"])
    n_pending = len(result["pending"])
    n_new = len(result["new"])

    print(f"📊 test_imgs/ 状态:")
    print(f"   已标定: {n_labeled}  待标定: {n_pending}  新图: {n_new}  总计: {n_labeled + n_pending + n_new}")

    for d in result["duplicates"]:
        print(f"\n   ⚠ {d['filename']} 与 {d['duplicate_of']} 内容重复，跳过")

    if n_new > 0:
        print(f"\n   🆕 新图:")
        for img in result["new"]:
            print(f"     - {img['filename']} ({img['width']}×{img['height']}, {img['megapixels']}MP)")
    if n_pending > 0:
        print(f"\n   ⏳ 待标定 (特征已提取，等待确认):")
        for img in result["pending"]:
            print(f"     - {img['filename']}")

    # 如果指定了 --image，只处理那一张
    if args.image:
        img_path, stem = find_image(args.image)
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ 无法解码: {img_path}")
            sys.exit(1)
        features = None
        predictor = _load_predictor()
        params = _resolve_params(args, features, predictor)

        if params == dict(DEFAULT_PARAMS) and predictor is None:
            print("🔮 预测器不可用，使用默认参数")
        elif args.n_layers is None and args.frame is None and args.min_island is None and args.quality is None:
            print(f"🔮 预测参数 (基于 {predictor['training_X'].shape[0] if predictor else 0} 条标定): "
                  f"n={params['n_layers']} f={params['frame_width']} i={params['min_island_area']} {params['quality']}")
        else:
            print(f"⚡ 手动指定参数: n={params['n_layers']} f={params['frame_width']} "
                  f"i={params['min_island_area']} {params['quality']}")

        print(f"📷 {stem}{img_path.suffix} ({image.shape[1]}×{image.shape[0]})")
        print(f"🔬 结构分层 n={params['n_layers']} frame={params['frame_width']}px "
              f"min_island={params['min_island_area']}px:")
        layer_masks, frame_mask, layer_stats, depth_ms, layer_ms, refine_ms = \
            _run_segmentation(img_path, stem, image, params, args, engines)
        _save_outputs(stem, image, layer_masks, frame_mask, params, args)
        return

    # 无 --image：处理新图
    if n_new == 0:
        if n_pending == 0:
            print("\n💡 所有图片已处理。放入新图或使用 --mode label 标定。")
        else:
            print(f"\n💡 {n_pending} 张待标定 — 使用 --mode label 标定")
        return

    predictor = _load_predictor()

    print(f"\n{'─' * 60}")
    for img_info in result["new"]:
        img_path = TEST_IMGS_DIR / img_info["filename"]
        print(f"\n🆕 处理新图: {img_info['filename']}")
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        stem = img_path.stem
        h, w = image.shape[:2]

        # 深度估计 + 特征提取
        print("   🔬 深度估计 …", end=" ", flush=True)
        t0 = time.perf_counter()
        depth_map = engines["estimate_depth"](image)
        depth_ms_val = int((time.perf_counter() - t0) * 1000)
        print(f"{depth_ms_val}ms")

        try:
            features = extract_features(image, depth_map, quantize_fn=engines.get("quantize_depth"))
        except Exception:
            features = extract_features(image, depth_map)

        # 保存深度缓存
        try:
            save_depth_cache(stem, depth_map)
        except Exception:
            pass

        # 预测参数 + 生成 2 个候选变体
        base_params = _resolve_params(args, features, predictor)
        candidates = _generate_candidates(base_params)
        best_params = None
        best_score = -1.0
        best_layer_masks = None
        best_frame_mask = None
        best_stats = None

        for ci, cand in enumerate(candidates):
            label = "🎯" if ci == 0 else "🔄"
            n = cand["n_layers"]; f = cand["frame_width"]; i = cand["min_island_area"]; q = cand["quality"]
            print(f"   {label} 候选{ci}: n={n} f={f} i={i} {q} …", end=" ", flush=True)

            try:
                lms, fms, lstats, _, l_ms, r_ms = \
                    _run_segmentation(img_path, stem, image, cand, args, engines)
                score = score_segmentation(lstats, (h, w))
                cs = score["combined_score"]
                print(f"score={cs:.3f}")
                if cs > best_score:
                    best_score = cs
                    best_score_dict = score
                    best_params = cand
                    best_layer_masks = lms
                    best_frame_mask = fms
                    best_stats = lstats
            except Exception as e:
                print(f"❌ {e}")
                continue

        if best_params is None:
            print("   ❌ 所有候选均失败，跳过")
            continue

        print(f"   ✅ 最优: n={best_params['n_layers']} f={best_params['frame_width']} "
              f"i={best_params['min_island_area']} score={best_score:.3f}")

        # 用最优参数渲染输出
        for s in best_stats:
            pct = float(np.count_nonzero(best_layer_masks[s["layer_index"]]) / (h * w) * 100)
            print(f"   层{s['layer_index']}: fg={s['fg_pixels']:,}px ({pct:.1f}%)  "
                  f"bridges={s['bridges_built']}  erased={s['islands_erased']}")
        _save_outputs(stem, image, best_layer_masks, best_frame_mask, best_params, args)

        # 注册（附带评分信息）
        reg.register_image(
            img_info["hash"], img_info["filename"],
            img_info["width"], img_info["height"], img_info["megapixels"],
            features=features,
        )
        reg.label_image(img_info["hash"], best_params, scores={
            "combined_score": best_score,
            "layer_balance": best_score_dict.get("layer_balance", 0),
        })
        reg.save()

    print(f"\n{'─' * 60}")
    print(f"📊 处理完毕:")
    print(f"   已标定 (跳过): {n_labeled}")
    print(f"   新图 (已处理): {n_new}  → 待标定")
    print(f"   待标定 (跳过): {n_pending}  → 使用 --mode label 标定")
    if n_new > 0 or n_pending > 0:
        print(f"\n💡 提示: 标定后可运行 --mode train 更新预测器")


# ══════════════════════════════════════════════════════════════════
#  模式: label
# ══════════════════════════════════════════════════════════════════

def cmd_label(args, engines):
    """交互标定模式。"""
    reg = ImageRegistry()

    img_path, stem = find_image(args.image)
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ 无法解码: {img_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    fhash = hash_file(img_path)
    entry = reg.data["images"].get(fhash)

    print(f"📷 {stem}{img_path.suffix} ({w}×{h}, {round(w*h/1e6,2)}MP)")
    if entry is not None and entry.get("params") is not None:
        print(f"   状态: 已标定 — {entry['params']}")
    elif entry is not None:
        print(f"   状态: 待标定 (特征已提取)")
    else:
        print(f"   状态: 新图 — 首次处理")

    # 特征提取
    predictor = _load_predictor()
    features = entry.get("features") if entry else None

    # 如果手动指定了所有参数，跳过预测
    manual_all = (args.n_layers is not None and args.frame is not None
                  and args.min_island is not None and args.quality is not None)

    if manual_all:
        params = _resolve_params(args, None, None)
        print(f"⚡ 手动指定参数: n={params['n_layers']} f={params['frame_width']} "
              f"i={params['min_island_area']} {params['quality']}")
    else:
        if predictor is not None and features is not None:
            params = _resolve_params(args, features, predictor)
        else:
            # 需要先获取特征
            print("   🔬 深度估计 …", end=" ", flush=True)
            t0 = time.perf_counter()
            depth_map = engines["estimate_depth"](image)
            print(f"{int((time.perf_counter()-t0)*1000)}ms")
            # 保存深度缓存，避免后续分割重复估计
            try:
                save_depth_cache(stem, depth_map)
            except Exception:
                pass
            try:
                features = extract_features(image, depth_map, quantize_fn=engines.get("quantize_depth"))
            except Exception:
                features = extract_features(image, depth_map)
            params = _resolve_params(args, features, predictor)

        if predictor is not None:
            print(f"🔮 预测参数 (基于 {predictor['training_X'].shape[0]} 条标定):")
        else:
            print(f"🔮 预测器不可用，使用默认参数:")
        print(f"   n_layers={params['n_layers']}  frame={params['frame_width']}px  "
              f"min_island={params['min_island_area']}px  quality={params['quality']}")

    # 交互循环
    while True:
        print(f"\n🔬 结构分层 n={params['n_layers']} frame={params['frame_width']}px "
              f"min_island={params['min_island_area']}px:")
        layer_masks, frame_mask, layer_stats, depth_ms, layer_ms, refine_ms = \
            _run_segmentation(img_path, stem, image, params, args, engines)

        for s in layer_stats:
            pct = float(np.count_nonzero(layer_masks[s["layer_index"]]) / (h * w) * 100)
            print(f"   层{s['layer_index']}: fg={s['fg_pixels']:,}px ({pct:.1f}%)  "
                  f"bridges={s['bridges_built']}  erased={s['islands_erased']}")
        print(f"   ⏱ 分层+精修: {int(layer_ms + refine_ms)}ms")

        overlay_path = _save_outputs(stem, image, layer_masks, frame_mask, params, args)

        print(f"{'─' * 60}")
        print(f"✂️  请打开查看:\n   {overlay_path}")
        print()

        # 等待输入
        try:
            choice = input("   满意? [Y=确认 / n=放弃 / e=改参重跑 / q=退出] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n🛑 退出。")
            break

        if choice == "y" or choice == "":
            reg.register_image(fhash, stem, w, h, round(w * h / 1e6, 2), features=features)
            reg.label_image(fhash, params)
            reg.save()
            n_labeled = reg.labeled_count
            n_total = n_labeled + reg.pending_count
            print(f"✅ 已标定: {stem} → n={params['n_layers']}, f={params['frame_width']}, "
                  f"i={params['min_island_area']} (第 {n_labeled}/{n_total} 张)")
            break
        elif choice == "n":
            reg.register_image(fhash, stem, w, h, round(w * h / 1e6, 2), features=features)
            reg.save()
            print(f"⏭  跳过标定。图片保留在注册表 (状态: 待标定)。")
            print(f"   可稍后用 --mode label --image {stem}{img_path.suffix} 重新标定。")
            break
        elif choice == "q":
            print("🛑 退出。当前图片未标定 (状态: 待标定)。")
            break
        elif choice == "e":
            print("   ── 输入新参数 (回车保留当前值) ──")
            try:
                v = input(f"   n_layers [{params['n_layers']}]: ").strip()
                if v:
                    params["n_layers"] = max(2, min(5, int(v)))
                v = input(f"   frame [{params['frame_width']}]: ").strip()
                if v:
                    params["frame_width"] = max(20, min(200, int(v)))
                v = input(f"   min_island [{params['min_island_area']}]: ").strip()
                if v:
                    params["min_island_area"] = max(10, min(5000, int(v)))
                v = input(f"   quality [{params['quality']}]: ").strip()
                if v and v in ("draft", "standard", "fine"):
                    params["quality"] = v
            except (EOFError, KeyboardInterrupt):
                print("\n🛑 退出。")
                break
            print(f"   ── 新参数: n={params['n_layers']} f={params['frame_width']} "
                  f"i={params['min_island_area']} quality={params['quality']} ──")
            continue
        else:
            print(f"   ❓ 未知选择: '{choice}' — 请输入 Y/n/e/q")


# ══════════════════════════════════════════════════════════════════
#  模式: train
# ══════════════════════════════════════════════════════════════════

def cmd_train(args, engines):
    """训练预测器。"""
    reg = ImageRegistry()
    dataset = reg.get_labeled_dataset()
    dataset_reviewed = reg.get_labeled_dataset(min_quality="reviewed")
    total = reg.total

    print(f"📊 读取 labeled.json …")
    print(f"   注册表: {total} 张图片")
    print(f"   已标定: {len(dataset)} 张  ← 可训练")
    print(f"   已审核: {len(dataset_reviewed)} 张  ← 有人工反馈")

    pending = reg.pending_count
    new_count = total - reg.labeled_count - pending
    if pending > 0 or new_count > 0:
        print(f"   待标定: {pending} 张   ← 跳过")
        if new_count > 0:
            print(f"   新图:   {new_count} 张   ← 跳过")

    if len(dataset_reviewed) >= 5:
        print(f"\n💡 建议: 使用 --min-quality reviewed 基于 {len(dataset_reviewed)} 张审核数据训练")

    predictor = train_predictor(dataset, min_samples=10)
    if predictor is not None:
        _save_predictor(predictor)
        print("✅ 训练完成。下次 scan/label 将使用新预测器。")
    else:
        print("❌ 训练失败。scan/label 将继续使用默认参数。")


# ══════════════════════════════════════════════════════════════════
#  模式: search
# ══════════════════════════════════════════════════════════════════

def cmd_search(args, engines):
    """搜索模式 — 接口就位，待 API 配置后启用。"""
    print("=" * 60)
    print("🔍 图片搜索 — 接口已定义，尚未实现")
    print()
    print("前置准备:")
    print("  1. 获取 Google API Key:")
    print("     https://console.cloud.google.com/apis/credentials")
    print("     启用 Custom Search API")
    print()
    print("  2. 创建搜索引擎:")
    print("     https://programmablesearchengine.google.com/")
    print("     → 搜索整个网络 → 获取 CX ID")
    print()
    print("  3. 设置环境变量:")
    print("     export GOOGLE_IMAGE_SEARCH_API_KEY='***'")
    print("     export GOOGLE_IMAGE_SEARCH_CX='your-cx'")
    print()
    print(f"本次查询: q='{args.query}' num={args.num} size={args.size} type={args.img_type}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    engines = _import_engines()

    # 预检查 loguru
    try:
        import loguru  # noqa: F401, F811
    except ImportError:
        print("❌ loguru 未安装 — 结构分层引擎内部依赖 loguru 日志库")
        print("   安装: pip install loguru --break-system-packages")
        sys.exit(1)

    mode = args.mode
    if mode == "scan":
        cmd_scan(args, engines)
    elif mode == "label":
        cmd_label(args, engines)
    elif mode == "train":
        cmd_train(args, engines)
    elif mode == "search":
        cmd_search(args, engines)


if __name__ == "__main__":
    main()
