import { useRef, useEffect } from 'react';
import { useZoomPan } from '../hooks/useZoomPan';
import type { SvgResponse } from '../types';

/* ── Types ──────────────────────────────────────────────── */

export interface LayerCfg {
  url: string | null;
  label: string;
  visible: boolean;
  opacity: number;
  disabled: boolean; // layer not applicable for this step
}

interface Props {
  layers: [LayerCfg, LayerCfg, LayerCfg];
  onToggle: (idx: 0 | 1 | 2) => void;
  onOpacity: (idx: 0 | 1 | 2, v: number) => void;
  version: number;
  svgLayer?: boolean;
  svgResult?: SvgResponse | null;
}

/* ── Component ──────────────────────────────────────────── */

export default function Canvas({
  layers, onToggle, onOpacity, version, svgLayer, svgResult,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { transform, reset, setTransform } = useZoomPan(containerRef, {
    minScale: 0.1, maxScale: 10, wheelStep: 0.1,
  });

  useEffect(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, [layers[0].url, setTransform]);

  if (!layers[0].url) {
    return (
      <div className="canvas-empty">
        <p>选择一张图片开始</p>
      </div>
    );
  }

  const anyVisible = layers.some(l => l.visible && l.url && !l.disabled);
  const labelText = anyVisible
    ? layers.filter(l => l.visible && l.url && !l.disabled).map(l => l.label).join(' + ')
    : '预览';

  return (
    <>
      {/* ── Controls row: 3 toggle buttons ─────────────── */}
      <div className="canvas-controls">
        {layers.map((l, i) => (
          <button
            key={i}
            className={`canvas-toggle-btn ${!l.disabled && l.visible ? 'active' : ''}`}
            disabled={l.disabled || !l.url}
            onClick={() => onToggle(i as 0 | 1 | 2)}
          >
            {!l.disabled && l.visible ? `✓ ${l.label}` : l.label}
          </button>
        ))}
      </div>

      {/* ── View label ──────────────────────────────────── */}
      <div className="canvas-view-label">{labelText}</div>

      {/* ── Viewport ────────────────────────────────────── */}
      <div className="canvas-viewport" ref={containerRef} onDoubleClick={reset} onDragStart={e => e.preventDefault()}>
        <div
          className="canvas-world canvas-world-overlay"
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
          }}
        >
          {layers.map((l, i) => {
            if (!l.url || l.disabled || !l.visible) return null;
            const isSvgLayer = i === 2 && svgLayer;
            return (
              <div
                key={i}
                className="world-pane"
                style={{ gridArea: '1 / 1', opacity: l.opacity }}
              >
                <img
                  src={`${l.url}?v=${version}`}
                  key={`l${i}-${version}`}
                  alt={l.label}
                  className="canvas-world-img"
                  style={isSvgLayer ? { filter: 'invert(1)' } : undefined}
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Opacity sliders ─────────────────────────────── */}
      <div className="opacity-sliders">
        {layers.map((l, i) => {
          if (l.disabled || !l.url || !l.visible) return null;
          return (
            <div key={i} className="opacity-slider">
              <span className="os-label">{l.label}</span>
              <input type="range" min={0} max={1} step={0.05}
                value={l.opacity}
                onChange={e => onOpacity(i as 0 | 1 | 2, Number(e.target.value))} />
              <span className="os-pct">{Math.round(l.opacity * 100)}%</span>
            </div>
          );
        })}
      </div>

      {/* ── SVG info (outside viewport) ─────────────────── */}
      {svgResult && (
        <div className="svg-info">
          <p className="svg-stat">
            {svgResult.total_paths} 条路径 · {svgResult.total_points} 个锚点
          </p>
          <p className="svg-stat dim">{svgResult.processing_time_ms} ms</p>
          <a className="svg-dl-btn" href={svgResult.svg_url} download
            target="_blank" rel="noopener noreferrer">
            下载 SVG
          </a>
        </div>
      )}
    </>
  );
}
