import { useRef, useEffect, useState } from 'react';
import { useZoomPan } from '../hooks/useZoomPan';
import type { MultiViewMode, MultiLayerInfo, SvgResponse } from '../types';

/* ── Types ──────────────────────────────────────────────── */

interface Props {
  /** Layer metadata — one entry per SAM layer. */
  layers: MultiLayerInfo[];
  /** Current step determining which result URL to display. */
  currentStep: string;
  /** The original uploaded image URL (always visible in overlay mode). */
  originalUrl: string;
  /** Current tiling/overlay view mode. */
  viewMode: MultiViewMode;
  onViewModeChange: (m: MultiViewMode) => void;
  /** Overlay-mode per-layer visibility state. */
  layerVisible: boolean[];
  onToggle: (idx: number) => void;
  /** Overlay-mode per-layer opacity. */
  layerOpacities: number[];
  onOpacity: (idx: number, v: number) => void;
  /** Cache-busting version — increments when results update. */
  version: number;
  /** SVG-specific rendering info (only set on SVG step). */
  svgResult?: SvgResponse | null;
  multiSvgResult?: { svg_url: string; total_paths: number; processing_time_ms: number } | null;
}

/* ── Helpers ────────────────────────────────────────────── */

/** Map pipeline step to the MultiLayerInfo field holding its result URL. */
function resultUrlForStep(l: MultiLayerInfo, step: string): string | null {
  switch (step) {
    case 'segment':   return l.frameUrl;            // show framed mask
    case 'canny':   return l.cannyUrl;
    case 'denoise':   return l.denoiseUrl;
    case 'connectivity': return l.connectivityUrl;
    case 'svg':       return l.connectivityUrl;     // SVG is a separate merged file
    default:          return l.frameUrl;
  }
}

/* ── Component ──────────────────────────────────────────── */

export default function MultiLayerCanvas({
  layers, currentStep, originalUrl, viewMode, onViewModeChange,
  layerVisible, onToggle, layerOpacities, onOpacity, version,
  svgResult, multiSvgResult,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { transform, reset, setTransform } = useZoomPan(containerRef, {
    minScale: 0.1, maxScale: 10, wheelStep: 0.1,
    reconnectKey: viewMode,
  });

  // ── Local state: original image visibility toggle ─────
  const [origVisible, setOrigVisible] = useState(true);

  useEffect(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, [originalUrl, setTransform]);

  if (layers.length === 0) {
    return (
      <div className="canvas-empty">
        <p>请先运行 SAM 分割</p>
      </div>
    );
  }

  const nLayers = layers.length;
  const visibleLayerCount = layerVisible.filter((v, i) => v && !!resultUrlForStep(layers[i], currentStep)).length;
  const totalVisible = (origVisible ? 1 : 0) + visibleLayerCount;

  return (
    <>
      {/* ── View mode switch ──────────────────────────── */}
      <div className="view-mode-bar">
        {(['vertical', 'horizontal', 'overlay'] as MultiViewMode[]).map(m => (
          <button
            key={m}
            className={`view-mode-btn ${viewMode === m ? 'active' : ''}`}
            onClick={() => onViewModeChange(m)}
          >
            {m === 'vertical' ? '⬍ 纵向排列' : m === 'horizontal' ? '⬌ 横向排列' : '◎ 叠加模式'}
          </button>
        ))}
      </div>

      {/* ── Toggle buttons (overlay only) ─────────────── */}
      {viewMode === 'overlay' && (
        <>
          <div className="canvas-controls">
            <button
              className={`canvas-toggle-btn ${origVisible ? 'active' : ''}`}
              onClick={() => setOrigVisible(v => !v)}
            >
              {origVisible ? '✓ 原图' : '原图'}
            </button>
            {layers.map((l, i) => (
              <button
                key={i}
                className={`canvas-toggle-btn ${layerVisible[i] ? 'active' : ''}`}
                onClick={() => onToggle(i)}
              >
                {layerVisible[i] ? `✓ ${l.label}` : l.label}
              </button>
            ))}
          </div>

          <div className="canvas-view-label">
            {totalVisible} 个图层叠加
          </div>
        </>
      )}

      {/* ── Tiled mode: original image toggle ─────────── */}
      {viewMode !== 'overlay' && (
        <div className="canvas-controls">
          <button
            className={`canvas-toggle-btn ${origVisible ? 'active' : ''}`}
            onClick={() => setOrigVisible(v => !v)}
          >
            {origVisible ? '✓ 原图' : '原图'}
          </button>
        </div>
      )}

      {/* ── Persistent viewport (ref survives mode switches) ── */}
      <div className="canvas-viewport" ref={containerRef} onDoubleClick={reset} onDragStart={e => e.preventDefault()}>
        {viewMode === 'overlay' ? (
          /* ════════ OVERLAY — CSS grid ════════ */
          <div
            className="canvas-world canvas-world-overlay"
            style={{
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            }}
          >
            {/* Original image (toggleable bottom layer) */}
            {origVisible && (
              <div className="world-pane" style={{ gridArea: '1 / 1', opacity: 1 }}>
                <img
                  src={`${originalUrl}?v=${version}`}
                  key={`orig-${version}`}
                  alt="原图"
                  className="canvas-world-img"
                />
              </div>
            )}

            {/* Per-layer results */}
            {layers.map((l, i) => {
              const url = resultUrlForStep(l, currentStep);
              if (!url || !layerVisible[i]) return null;
              return (
                <div
                  key={i}
                  className="world-pane"
                  style={{ gridArea: '1 / 1', opacity: layerOpacities[i] }}
                >
                  <img
                    src={`${url}?v=${version}`}
                    key={`ml${i}-${version}`}
                    alt={l.label}
                    className="canvas-world-img"
                  />
                </div>
              );
            })}
          </div>
        ) : (
          /* ════════ TILED — flex layout ════════ */
          <div
            className={`layer-tiles tiles-${viewMode}`}
            style={{
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            }}
          >
            {/* Original image (first tile — first column in horizontal, first row in vertical) */}
            {origVisible && (
              <div className="layer-tile">
                <img
                  src={`${originalUrl}?v=${version}`}
                  alt="原图"
                  className="layer-tile-img"
                />
                <span className="layer-label">原图</span>
              </div>
            )}
            {layers.map((l, i) => {
              const url = resultUrlForStep(l, currentStep);
              return (
                <div className="layer-tile" key={i}>
                  {url ? (
                    <img
                      src={`${url}?v=${version}`}
                      alt={l.label}
                      className="layer-tile-img"
                    />
                  ) : (
                    <div className="layer-tile-empty">待处理</div>
                  )}
                  <span className="layer-label">{l.label}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Opacity sliders (overlay only) ─────────────── */}
      {viewMode === 'overlay' && (
        <div className="opacity-sliders">
          {layers.map((l, i) => {
            if (!layerVisible[i]) return null;
            return (
              <div key={i} className="opacity-slider">
                <span className="os-label">{l.label}</span>
                <input type="range" min={0} max={1} step={0.05}
                  value={layerOpacities[i]}
                  onChange={e => onOpacity(i, Number(e.target.value))} />
                <span className="os-pct">{Math.round(layerOpacities[i] * 100)}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* ── SVG info (multi-layer merged SVG) ──────────── */}
      {(multiSvgResult || svgResult) && (
        <div className="svg-info">
          {multiSvgResult ? (
            <>
              <p className="svg-stat">
                {multiSvgResult.total_paths} 条路径 · {nLayers} 层
              </p>
              <p className="svg-stat dim">{multiSvgResult.processing_time_ms} ms</p>
            </>
          ) : svgResult ? (
            <>
              <p className="svg-stat">
                {svgResult.total_paths} 条路径 · {svgResult.total_points} 个锚点
              </p>
              <p className="svg-stat dim">{svgResult.processing_time_ms} ms</p>
            </>
          ) : null}
          <a
            className="svg-dl-btn"
            href={multiSvgResult?.svg_url ?? svgResult?.svg_url ?? '#'}
            download
            target="_blank" rel="noopener noreferrer"
          >
            下载 SVG
          </a>
        </div>
      )}
    </>
  );
}
