import { useState, useCallback, useRef, useEffect } from 'react';

export interface ZoomPanState {
  scale: number;
  x: number;
  y: number;
}

interface UseZoomPanOptions {
  minScale?: number;
  maxScale?: number;
  wheelStep?: number;
  /** When this value changes, listeners are detached and re-attached.
   *  Pass a view-mode or step identifier to survive DOM reconciliation. */
  reconnectKey?: string | number;
}

export function useZoomPan(
  containerRef: React.RefObject<HTMLDivElement | null>,
  options: UseZoomPanOptions = {},
) {
  const { minScale = 0.1, maxScale = 10, wheelStep = 0.1, reconnectKey } = options;

  const [transform, setTransform] = useState<ZoomPanState>({ scale: 1, x: 0, y: 0 });
  const transformRef = useRef(transform);
  transformRef.current = transform;

  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 });

  // ── Attach/detach all gesture listeners ───────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Wheel → zoom
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();

      const rect = el.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const cursorY = e.clientY - rect.top;

      setTransform((prev) => {
        const delta = e.deltaY > 0 ? -wheelStep : wheelStep;
        const s = Math.min(maxScale, Math.max(minScale, prev.scale * (1 + delta)));
        const r = s / prev.scale;
        return {
          scale: s,
          x: cursorX - r * (cursorX - prev.x),
          y: cursorY - r * (cursorY - prev.y),
        };
      });
    };

    // Mousedown → start pan
    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      isPanning.current = true;
      const cur = transformRef.current;
      panStart.current = { x: e.clientX, y: e.clientY, tx: cur.x, ty: cur.y };
      window.addEventListener('mousemove', onGlobalMouseMove);
      window.addEventListener('mouseup', onGlobalMouseUp);
    };

    const onGlobalMouseMove = (e: MouseEvent) => {
      if (!isPanning.current) return;
      const s = panStart.current;
      setTransform((prev) => ({
        ...prev,
        x: s.tx + (e.clientX - s.x),
        y: s.ty + (e.clientY - s.y),
      }));
    };

    const onGlobalMouseUp = () => {
      isPanning.current = false;
      window.removeEventListener('mousemove', onGlobalMouseMove);
      window.removeEventListener('mouseup', onGlobalMouseUp);
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('mousedown', onMouseDown);

    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onGlobalMouseMove);
      window.removeEventListener('mouseup', onGlobalMouseUp);
    };
  }, [containerRef, minScale, maxScale, wheelStep, reconnectKey]);

  // ── Reset ─────────────────────────────────────
  const reset = useCallback(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, []);

  return { transform, reset, setTransform };
}
