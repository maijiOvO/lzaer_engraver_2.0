═══════════════════════════════════════════════════════════
  Canny LineArt 线稿引擎 — 便携包
═══════════════════════════════════════════════════════════

来自 laser-engraver-app 项目，经过 117 张图片批量对比验证的最优线稿算法。
零项目依赖，可独立使用或嵌入任何 Python 项目。

─────────────────────────────────────────────────────────
1. 安装
─────────────────────────────────────────────────────────

    pip install -r requirements.txt

    仅需两个包：opencv-python + numpy。不需要 PyTorch/controlnet_aux/PIL。

─────────────────────────────────────────────────────────
2. 快速验证
─────────────────────────────────────────────────────────

    python test_single.py <图片路径>

    示例：
    python test_single.py D:/Desktop/images/city.jpg

    输出文件：<图片名>_lineart.png

─────────────────────────────────────────────────────────
3. 批量处理
─────────────────────────────────────────────────────────

    python batch_run.py <输入目录> [输出目录]

    示例：
    python batch_run.py ./test_imgs ./output

─────────────────────────────────────────────────────────
4. Python 调用
─────────────────────────────────────────────────────────

    from canny_lineart import canny_lineart
    import cv2

    # 基本用法
    result = canny_lineart("input.jpg")
    cv2.imwrite("output.png", result)

    # 从内存中的 numpy 数组
    img = cv2.imread("input.jpg")
    result = canny_lineart(img)

    # 调参
    result = canny_lineart(img, low=30, high=100)     # 更多细节
    result = canny_lineart(img, low=80, high=200)     # 更少噪点
    result = canny_lineart(img, smooth_level=1)       # 去除微小纹理
    result = canny_lineart(img, smooth_level=2)       # 抹平密集纹理

─────────────────────────────────────────────────────────
5. CLI 调用
─────────────────────────────────────────────────────────

    python canny_lineart.py input.jpg
    python canny_lineart.py input.jpg output.png
    python canny_lineart.py input.jpg --low 30 --high 100 --smooth 1

─────────────────────────────────────────────────────────
6. 参数说明
─────────────────────────────────────────────────────────

    low=50, high=150
        Canny 阈值。low 越低细节越多，high 越高噪点越少。
        推荐比例 high = low × 3。
        当前默认值经过 117 张图片批量对比验证。

    smooth_level=0
        0 = 默认 (Gaussian 3×3) — 保留最多细节
        1 = 轻量 (Bilateral 5×5 + Gaussian 3×3) — 去除微小纹理
        2 = 中等 (Bilateral 7×7 + Gaussian 5×5) — 抹平密集纹理

    预处理管线：
        BGR → Gray → CLAHE(2.0, 8×8) → 平滑 → Canny(low, high)

─────────────────────────────────────────────────────────
7. 集成到其他项目
─────────────────────────────────────────────────────────

    方式 A — 直接复制文件：
        将 canny_lineart.py 复制到你的项目中，直接 import。

    方式 B — 作为子模块：
        将整个目录放到项目中，sys.path 引入后 import。

    方式 C — pip install -e .：
        在项目根目录创建 setup.py/pyproject.toml 后安装。

─────────────────────────────────────────────────────────
8. 性能参考
─────────────────────────────────────────────────────────

    600×800:    ~2ms
    900×1200:   ~2ms
    2268×3504:  ~20ms
    3767×6697:  ~105ms

    无 GPU 依赖，纯 CPU 运算。
    首次调用无模型下载，即开即用。

═══════════════════════════════════════════════════════════
