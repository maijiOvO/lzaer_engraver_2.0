/**
 * BrushTool — 笔刷式蒙版精修引擎
 *
 * 在逐层审核模式下，提供类似 PS 笔刷的涂抹交互：
 *   - 🟢 纳入笔刷：将涂抹区域加入当前层
 *   - 🔴 排除笔刷：将涂抹区域从当前层移除
 *   - 点击「应用SAM」：笔画→SAM point prompts→局部精修
 *   - Ctrl+Z 撤销
 *
 * 所有笔画坐标以原图像素为单位存储，渲染时动态转换为 viewport 坐标。
 */

const BrushTool = {
  /** @type {'include'|'exclude'} */
  brushType: 'include',
  brushSize: 20,

  /** @type {{type:'include'|'exclude', points:[number,number][]}[]} */
  strokes: [],

  /** @type {HTMLCanvasElement|null} */
  canvas: null,
  /** @type {CanvasRenderingContext2D|null} */
  ctx: null,

  /** Current stroke being drawn (not yet committed) */
  _currentStroke: null,

  _drawing: false,
  _enabled: false,

  /** Reference to the original image natural dimensions */
  imgW: 0,
  imgH: 0,

  /** Frame width for coordinate compensation (same as segmentation frame_width) */
  frameWidth: 0,

  /** ── Initialise ───────────────────────────────── */
  init() {
    // Create brush overlay canvas
    let c = document.getElementById('brushCanvas');
    if (!c) {
      c = document.createElement('canvas');
      c.id = 'brushCanvas';
      c.style.cssText = 'position:absolute;top:0;left:0;pointer-events:auto;z-index:10;';
      document.getElementById('world').appendChild(c);
    }
    this.canvas = c;
    this.ctx = c.getContext('2d');

    this._bindEvents();
  },

  /** ── Enable / Disable brush mode ─────────────── */
  enable(imgW, imgH, frameWidth) {
    this._enabled = true;
    this.imgW = imgW;
    this.imgH = imgH;
    this.frameWidth = frameWidth || 0;
    this.strokes = [];

    // 笔刷 canvas 与图层 pane 对齐 — 图层蒙版尺寸为 orig + 2*fw（外框延伸），
    // 有效内容从 (fw, fw) 像素开始。 用 left: -fw / top: -fw 将笔刷画布偏移，
    // 使笔刷坐标与图层内容区的世界坐标对齐。
    // ⚠️ 此偏移必须与 index.html renderLayersView() 中的 world-pane 保持一致。
    if (this.canvas && this.frameWidth > 0) {
      this.canvas.style.left = (-this.frameWidth) + 'px';
      this.canvas.style.top = (-this.frameWidth) + 'px';
    }

    document.getElementById('brushToolbar').classList.add('show');
    document.getElementById('gestureHint').textContent = '🖱 滚轮缩放 · Ctrl+拖拽平移 · 笔刷涂抹 · 双击复位';
    this._resize();
    this._redraw();
  },

  disable() {
    this._enabled = false;
    this.strokes = [];
    document.getElementById('brushToolbar').classList.remove('show');
    document.getElementById('gestureHint').textContent = '🖱 滚轮缩放 · 拖拽平移 · 双击复位';
    if (this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  },

  /** ── Resize canvas to match world container ──── */
  _resize() {
    if (!this.canvas) return;
    const world = document.getElementById('world');
    const img = world.querySelector('img');
    if (!img || !img.naturalWidth) {
      this.canvas.width = this.imgW || 800;
      this.canvas.height = this.imgH || 600;
    } else {
      this.canvas.width = img.naturalWidth;
      this.canvas.height = img.naturalHeight;
    }
  },

  /** ── Redraw all strokes ──────────────────────── */
  _redraw() {
    if (!this.ctx || !this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (const stroke of this.strokes) {
      this._drawStroke(stroke);
    }

    // Draw current stroke in progress
    if (this._currentStroke) {
      this._drawStroke(this._currentStroke);
    }
  },

  _drawStroke(stroke) {
    if (!this.ctx || stroke.points.length < 1) return;
    const ctx = this.ctx;

    ctx.beginPath();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = this.brushSize;
    ctx.strokeStyle = stroke.type === 'include'
      ? 'rgba(46,204,113,0.5)'
      : 'rgba(231,76,60,0.5)';

    ctx.moveTo(stroke.points[0][0], stroke.points[0][1]);
    for (let i = 1; i < stroke.points.length; i++) {
      ctx.lineTo(stroke.points[i][0], stroke.points[i][1]);
    }
    ctx.stroke();
  },

  /** ── Coordinate conversion ───────────────────── */
  /** Viewport → Image coordinates */
  vp2img(vpX, vpY) {
    // 减去 #viewport 在页面中的偏移（修复侧面栏导致的位置偏差）
    const rect = document.getElementById('viewport').getBoundingClientRect();
    const cx = vpX - rect.left;
    const cy = vpY - rect.top;
    // Account for zoom/pan transform on #world
    const ix = (cx - zoom.x) / zoom.scale;
    const iy = (cy - zoom.y) / zoom.scale;
    return [Math.round(ix), Math.round(iy)];
  },

  /** ── Event binding ────────────────────────────── */
  _bindEvents() {
    if (this._eventsBound) return;  // 防止重复绑定（多次进出 review 模式）
    this._eventsBound = true;

    const vp = document.getElementById('viewport');

    vp.addEventListener('mousedown', (e) => {
      if (!this._enabled || e.button !== 0) return;
      // Ctrl+drag = pan, regular drag = brush
      if (e.ctrlKey) return;

      this._drawing = true;
      const [ix, iy] = this.vp2img(e.clientX, e.clientY);
      this._currentStroke = { type: this.brushType, points: [[ix, iy]] };
      e.preventDefault();
      e.stopPropagation();
    });

    window.addEventListener('mousemove', (e) => {
      if (!this._enabled || !this._drawing) return;
      const [ix, iy] = this.vp2img(e.clientX, e.clientY);
      this._currentStroke.points.push([ix, iy]);
      this._redraw();
      e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
      if (!this._enabled || !this._drawing) return;
      this._drawing = false;
      if (this._currentStroke && this._currentStroke.points.length === 1) {
        // 单击涂抹 — 补一个偏移1px的点，使笔画有效
        const [x, y] = this._currentStroke.points[0];
        this._currentStroke.points.push([x + 1, y]);
      }
      if (this._currentStroke && this._currentStroke.points.length > 1) {
        this.strokes.push(this._currentStroke);
      }
      this._currentStroke = null;
      this._updateBadge();
    });

    // Ctrl+Z = undo
    window.addEventListener('keydown', (e) => {
      if (!this._enabled) return;
      if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        this.undo();
      }
      if (e.key === '[') { this.brushSize = Math.max(5, this.brushSize - 5); this._updateSize(); }
      if (e.key === ']') { this.brushSize = Math.min(100, this.brushSize + 5); this._updateSize(); }
    });
  },

  /** ── Brush type toggle ───────────────────────── */
  setBrushType(type) {
    this.brushType = type;
    document.querySelectorAll('.brush-type-btn').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-btype') === type);
    });
  },

  _updateSize() {
    const el = document.getElementById('brushSizeVal');
    const sl = document.getElementById('brushSize');
    if (el) el.textContent = this.brushSize + 'px';
    if (sl) sl.value = this.brushSize;
  },

  _updateBadge() {
    const el = document.getElementById('brushCount');
    if (el) el.textContent = this.strokes.length;
  },

  /** ── Undo ────────────────────────────────────── */
  undo() {
    this.strokes.pop();
    this._redraw();
    this._updateBadge();
  },

  /** ── Clear all strokes ───────────────────────── */
  clearStrokes() {
    this.strokes = [];
    this._redraw();
    this._updateBadge();
  },

  /** ── Apply SAM ────────────────────────────────── */
  async applySam(layerIndex, currentMaskKey, frameWidth) {
    if (this.strokes.length === 0) {
      alert('没有笔刷笔画，请先涂抹需要修正的区域。');
      return;
    }

    const payload = {
      image_name: curImageName,
      layer_index: layerIndex,
      current_mask_key: currentMaskKey,
      frame_width: frameWidth || this.frameWidth || 50,
      strokes: this.strokes.map(s => ({
        brush_type: s.type,
        points: s.points,
      })),
    };

    const btn = document.getElementById('btnApplySam');
    if (btn) { btn.textContent = '⏳ SAM精修中…'; btn.disabled = true; }

    try {
      const r = await fetch('/api/brush-refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await r.json();

      if (!r.ok || !data.ok) {
        alert('SAM 精修失败: ' + (data.message || data.detail || '未知错误'));
        return;
      }

      // Success — clear strokes and reload the layer view
      this.clearStrokes();
      if (btn) btn.textContent = `✅ 已精修 (置信度:${(data.sam_score*100).toFixed(0)}%)`;

      // Refresh the layer display
      if (typeof refreshLayerDisplay === 'function') {
        refreshLayerDisplay(layerIndex);
      }

      setTimeout(() => {
        if (btn) { btn.textContent = '应用SAM'; btn.disabled = false; }
      }, 2000);

    } catch (e) {
      alert('SAM 精修失败: ' + e.message);
      if (btn) { btn.textContent = '应用SAM'; btn.disabled = false; }
    }
  },

  /** ── Reset layer to original ──────────────────── */
  resetLayer() {
    if (!confirm('重置本层到分割初始状态？所有笔刷修正将丢失。')) return;
    this.clearStrokes();
    if (typeof resetLayerDisplay === 'function') {
      resetLayerDisplay();
    }
  },
};
