import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import apiClient from './api/client';
import ControlPanel from './components/ControlPanel';
import Canvas from './components/Canvas';
import MultiLayerCanvas from './components/MultiLayerCanvas';
import ProgressBar from './components/ProgressBar';
import ImageUploader from './components/ImageUploader';
import type { StepInfo } from './components/ProgressBar';
import type { LayerCfg } from './components/Canvas';
import type {
  UploadResponse, PipelineStepResponse,
  ConnectivityResponse, SvgResponse, SegmentResponse,
  MultiLayerInfo, MultiViewMode, MultiLayerSvgParams, MultiLayerSvgResponse,
} from './types';

/* ── Types ──────────────────────────────────────────────── */

export type StepStatus = 'idle' | 'loading' | 'done' | 'stale';
export type PipelineStep = 'canny' | 'connectivity' | 'svg' | 'segment' | 'denoise';
type PipelineMode = 'single' | 'multi' | null;

interface CannyCfg {
  low: number; high: number; smooth_level: number;
}

const STEP_ORDER: PipelineStep[] = ['segment', 'canny', 'denoise', 'connectivity', 'svg'];

/* ── Layer labels per step ───────────────────────────────── */

const STEP_LAYERS: Record<PipelineStep, [string, string | null, string]> = {
  segment:       ['原始图片', null,             '图层分割'],
  canny:         ['原始图片', null,             'Canny 边缘'],
  denoise:       ['原始图片', '线稿结果',        '降噪结果'],
  connectivity:  ['原始图片', '降噪结果',        '连通修复'],
  svg:           ['原始图片', '连通修复',        'SVG 矢量图'],
};

/* ── Component ──────────────────────────────────────────── */

function App() {
  const [imageId, setImageId] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  // Mode selection
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>(null);

  // Lineart
  const [cannyCfg, setCannyCfg] = useState<CannyCfg>({ low: 50, high: 150, smooth_level: 0 });
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [cannyLoading, setCannyLoading] = useState(false);
  const [cannyElapsed, setCannyElapsed] = useState<number | null>(null);
  const [cannyVersion, setCannyVersion] = useState(0);

  // Segment (Depth-guided structural layering)
  const [nLayers, setNLayers] = useState(3);
  const [frameWidth, setFrameWidth] = useState(50);
  const [minIslandArea, setMinIslandArea] = useState(100);
  const [samQuality, setSamQuality] = useState('standard');
  const [segmentResult, setSegmentResult] = useState<SegmentResponse | null>(null);
  const [segmentLoading, setSegmentLoading] = useState(false);
  const [segmentVersion, setSegmentVersion] = useState(0);
  const [segmentElapsed, setSegmentElapsed] = useState<number | null>(null);

  // Connectivity
  const [gapTolerance, setGapTolerance] = useState(5);
  const [connectivityResult, setConnectivityResult] = useState<ConnectivityResponse | null>(null);
  const [connectivityLoading, setConnectivityLoading] = useState(false);
  const [connectivityVersion, setConnectivityVersion] = useState(0);

  /* ── Multi-layer state ─────────────────────────────────── */
  const [multiViewMode, setMultiViewMode] = useState<MultiViewMode>('overlay');
  const [multiLayers, setMultiLayers] = useState<MultiLayerInfo[]>([]);
  const [multiLayerVisible, setMultiLayerVisible] = useState<boolean[]>([]);
  const [multiLayerOpacities, setMultiLayerOpacities] = useState<number[]>([]);
  const [cannyLayers, setCannyLayers] = useState<string[]>([]);
  const [denoiseLayers, setDenoiseLayers] = useState<string[]>([]);
  const [connectivityLayers, setConnectivityLayers] = useState<string[]>([]);
  const [multiVersion, setMultiVersion] = useState(0);
  const [multiSvgResult, setMultiSvgResult] = useState<MultiLayerSvgResponse | null>(null);

  // Denoise
  const [minComponentArea, setMinComponentArea] = useState(4);
  const [denoiseResult, setDenoiseResult] = useState<PipelineStepResponse | null>(null);
  const [denoiseLoading, setDenoiseLoading] = useState(false);
  const [denoiseVersion, setDenoiseVersion] = useState(0);

  // SVG
  const [simplifyTolerance, setSimplifyTolerance] = useState(1.0);
  const [svgResult, setSvgResult] = useState<SvgResponse | null>(null);
  const [svgLoading, setSvgLoading] = useState(false);
  const [svgVersion, setSvgVersion] = useState(0);

  // Layer visibility & opacity — per-step, 3 layers: [原图, 上一步, 当前]
  const [layerVisible, setLayerVisible] = useState<
    Record<PipelineStep, [boolean, boolean, boolean]>
  >({
    segment:       [true, false, true],
    canny:         [true, false, true],
    denoise:       [true, true,  true],
    connectivity:  [true, true,  true],
    svg:           [true, true,  true],
  });
  const [layerOpacities, setLayerOpacities] = useState<
    Record<PipelineStep, [number, number, number]>
  >({
    segment:       [1, 0, 1],
    canny:         [1, 0, 1],
    denoise:       [1, 1, 1],
    connectivity:  [1, 1, 1],
    svg:           [1, 1, 1],
  });

  // Pipeline state
  const [currentStep, setCurrentStep] = useState<PipelineStep>('canny');

  /* ── Pipeline generation counters & snapshots ──────────── */
  const [segmentGen, setSegmentGen] = useState(0);
  const [cannySegmentSnap, setCannySegmentSnap] = useState<number | null>(null);
  const [cannyGen, setCannyGen] = useState(0);
  const [denoiseGen, setDenoiseGen] = useState(0);
  const [denoiseCannySnap, setDenoiseLineartSnap] = useState<number | null>(null);
  const [connectivityGen, setConnectivityGen] = useState(0);
  const [connectivityDenoiseSnap, setConnectivityDenoiseSnap] = useState<number | null>(null);
  const [svgCannySnap, setSvgLineartSnap] = useState<number | null>(null);
  const [svgDenoiseSnap, setSvgDenoiseSnap] = useState<number | null>(null);
  const [svgConnectivitySnap, setSvgConnectivitySnap] = useState<number | null>(null);

  const segmentGenRef = useRef(segmentGen); segmentGenRef.current = segmentGen;
  const cannyGenRef = useRef(cannyGen); cannyGenRef.current = cannyGen;
  const denoiseGenRef = useRef(denoiseGen); denoiseGenRef.current = denoiseGen;
  const connectivityGenRef = useRef(connectivityGen); connectivityGenRef.current = connectivityGen;
  const connectivityResultRef = useRef(connectivityResult); connectivityResultRef.current = connectivityResult;
  const pipelineModeRef = useRef(pipelineMode); pipelineModeRef.current = pipelineMode;
  const multiLayersRef = useRef(multiLayers); multiLayersRef.current = multiLayers;
  const cannyLayersRef = useRef(cannyLayers); cannyLayersRef.current = cannyLayers;
  const denoiseLayersRef = useRef(denoiseLayers); denoiseLayersRef.current = denoiseLayers;
  const connectivityLayersRef = useRef(connectivityLayers); connectivityLayersRef.current = connectivityLayers;

  /* ── Refs for debounce ─────────────────────────────────── */
  const imageIdRef = useRef(imageId); imageIdRef.current = imageId;
  const cannyCfgRef = useRef(cannyCfg); cannyCfgRef.current = cannyCfg;
  const nLayersRef = useRef(nLayers); nLayersRef.current = nLayers;
  const frameWidthRef = useRef(frameWidth); frameWidthRef.current = frameWidth;
  const minIslandAreaRef = useRef(minIslandArea); minIslandAreaRef.current = minIslandArea;
  const samQualityRef = useRef(samQuality); samQualityRef.current = samQuality;
  const gapRef = useRef(gapTolerance); gapRef.current = gapTolerance;
  const simplRef = useRef(simplifyTolerance); simplRef.current = simplifyTolerance;
  const minAreaRef = useRef(minComponentArea); minAreaRef.current = minComponentArea;
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(false);

  /* ── Upload ────────────────────────────────────────────── */
  const handleUploaded = (data: UploadResponse) => {
    setImageId(data.image_id);
    setImageUrl(data.original_url);
    setPipelineMode(null);
    setOverlayUrl(null);
    setConnectivityResult(null);
    setSvgResult(null);
    setCannyElapsed(null);
    setCannyVersion(0);
    setCannyCfg({ low: 50, high: 150, smooth_level: 0 });
    setGapTolerance(5);
    setSimplifyTolerance(1.0);
    setDenoiseResult(null);
    setDenoiseVersion(0);
    setMinComponentArea(4);
    setSegmentResult(null);
    setSegmentVersion(0);
    setSegmentElapsed(null);
    setSegmentLoading(false);
    setNLayers(3);
    setFrameWidth(50);
    setMinIslandArea(100);
    setSamQuality('standard');
    setSegmentGen(0);
    setCannySegmentSnap(null);
    setCannyGen(0);
    setDenoiseGen(0);
    setDenoiseLineartSnap(null);
    setConnectivityGen(0);
    setConnectivityDenoiseSnap(null);
    setConnectivityVersion(0);
    setSvgVersion(0);
    setSvgLineartSnap(null);
    setSvgDenoiseSnap(null);
    setSvgConnectivitySnap(null);
    // Multi-layer reset
    setMultiLayers([]);
    setMultiLayerVisible([]);
    setMultiLayerOpacities([]);
    setCannyLayers([]);
    setDenoiseLayers([]);
    setConnectivityLayers([]);
    setMultiVersion(0);
    setMultiSvgResult(null);
    setCurrentStep('canny');
    setLayerVisible({ segment: [true, false, true], canny: [true, false, true], denoise: [true, true, true], connectivity: [true, true, true], svg: [true, true, true] });
    setLayerOpacities({ segment: [1, 0, 1], canny: [1, 0, 1], denoise: [1, 1, 1], connectivity: [1, 1, 1], svg: [1, 1, 1] });
  };

  /* ── Mode selection handler ────────────────────────────── */
  const handleSelectMode = useCallback((mode: 'single' | 'multi') => {
    setPipelineMode(mode);
    if (mode === 'multi') {
      setCurrentStep('segment');
    } else {
      setCurrentStep('canny');
    }
  }, []);

  /* ── Navigation ────────────────────────────────────────── */
  const goToStep = useCallback((step: PipelineStep) => {
    if (!pipelineMode) return;
    setCurrentStep(step);
  }, [pipelineMode]);

  const goPrev = useCallback(() => {
    const idx = STEP_ORDER.indexOf(currentStep);
    // Skip segment in single-layer mode
    if (pipelineMode === 'single' && idx === 2) {
      setCurrentStep('canny');
      return;
    }
    if (idx > 0) setCurrentStep(STEP_ORDER[idx - 1]);
  }, [currentStep, pipelineMode]);

  const goNext = useCallback(() => {
    const idx = STEP_ORDER.indexOf(currentStep);
    // Skip segment in single-layer mode
    if (pipelineMode === 'single' && currentStep === 'canny') {
      setCurrentStep('denoise');
      return;
    }
    if (idx < STEP_ORDER.length - 1) setCurrentStep(STEP_ORDER[idx + 1]);
  }, [currentStep, pipelineMode]);

  /* ── API call helpers ──────────────────────────────────── */
  const callSegment = useCallback(async () => {
    const id = imageIdRef.current;
    if (!id) return;
    setSegmentLoading(true);
    const t0 = performance.now();
    try {
      const { data } = await apiClient.post<SegmentResponse>('/pipeline/segment', {
        image_id: id,
        n_layers: nLayersRef.current,
        frame_width: frameWidthRef.current,
        min_island_area: minIslandAreaRef.current,
        sam_quality: samQualityRef.current,
      }, { timeout: 600_000 });  // Depth + SAM refinement on CPU
      // Guard against stale results: if the user re-uploaded while SAM was
      // running, discard the result for the old image.
      if (imageIdRef.current !== id) return;
      setSegmentResult(data);
      setSegmentVersion(v => v + 1);
      setSegmentElapsed(Math.round(performance.now() - t0));
      setSegmentGen(g => g + 1);
      // Populate multi-layer state from segmentation result
      const ml: MultiLayerInfo[] = data.layers.map((l: { layer_index: number; mask_url: string; frame_url: string }) => ({
        layerIndex: l.layer_index,
        maskUrl: l.mask_url,
        frameUrl: l.frame_url,
        cannyUrl: null,
        denoiseUrl: null,
        connectivityUrl: null,
        label: `图层 ${l.layer_index + 1}`,
      }));
      setMultiLayers(ml);
      setMultiLayerVisible(ml.map(() => true));
      setMultiLayerOpacities(ml.map(() => 1));
    } catch (err: any) {
      if (!err.response) {
        alert('SAM 分割连接中断 — 后端可能正在重启。请等待几秒后重试。');
      } else {
        const detail =
          err?.response?.data?.detail ??
          err?.response?.data?.error_msg ??
          err?.message ??
          '未知错误';
        alert(`SAM 分割失败\n\n${detail}`);
      }
    }
    finally { setSegmentLoading(false); }
  }, []);

  const callCanny = useCallback(async () => {
    const id = imageIdRef.current;
    if (!id || pipelineModeRef.current === null) return;
    setCannyLoading(true);
    try {
      if (pipelineModeRef.current === 'multi') {
        // Multi-layer: concurrently call per-layer canny
        const layers = multiLayersRef.current;
        const results = await Promise.all(
          layers.map((_, i) =>
            apiClient.post<PipelineStepResponse>('/pipeline/canny', {
              image_id: id, layer_index: i,
              low: cannyCfgRef.current.low,
              high: cannyCfgRef.current.high,
              smooth_level: cannyCfgRef.current.smooth_level,
            })
          )
        );
        const urls = results.map(r => r.data.result_url);
        setCannyLayers(urls);
        setMultiVersion(v => v + 1);
        // Keep single-layer overlayUrl for compatibility
        setOverlayUrl(urls[0] ?? null);
      } else {
        const { data } = await apiClient.post<PipelineStepResponse>('/pipeline/canny', {
          image_id: id, layer_index: null,
          low: cannyCfgRef.current.low,
          high: cannyCfgRef.current.high,
          smooth_level: cannyCfgRef.current.smooth_level,
        });
        setOverlayUrl(data.result_url);
      }
      setCannyElapsed(null);  // multi-layer doesn't track per-call elapsed
      setCannyVersion(v => v + 1);
      setCannySegmentSnap(segmentGenRef.current);
      setCannyGen(g => g + 1);
    } catch { alert('线稿生成失败'); }
    finally { setCannyLoading(false); }
  }, []);

  const callConnectivity = useCallback(async () => {
    const id = imageIdRef.current;
    if (!id || pipelineModeRef.current === null) return;
    setConnectivityLoading(true);
    try {
      if (pipelineModeRef.current === 'multi') {
        const layers = multiLayersRef.current;
        const results = await Promise.all(
          layers.map((_, i) =>
            apiClient.post<ConnectivityResponse>('/pipeline/connectivity', {
              image_id: id, layer_index: i,
              gap_tolerance: gapRef.current,
            })
          )
        );
        const urls = results.map(r => r.data.result_url);
        setConnectivityLayers(urls);
        setMultiVersion(v => v + 1);
        setConnectivityResult(results[0]?.data ?? null);
      } else {
        const { data } = await apiClient.post<ConnectivityResponse>('/pipeline/connectivity', {
          image_id: id, gap_tolerance: gapRef.current,
        });
        setConnectivityResult(data);
      }
      setConnectivityDenoiseSnap(denoiseGenRef.current);
      setConnectivityGen(g => g + 1);
      setConnectivityVersion(v => v + 1);
    } catch { alert('连通性修复失败'); }
    finally { setConnectivityLoading(false); }
  }, []);

  const callDenoise = useCallback(async () => {
    const id = imageIdRef.current;
    if (!id || pipelineModeRef.current === null) return;
    setDenoiseLoading(true);
    try {
      if (pipelineModeRef.current === 'multi') {
        const layers = multiLayersRef.current;
        const results = await Promise.all(
          layers.map((_, i) =>
            apiClient.post<PipelineStepResponse>('/pipeline/denoise', {
              image_id: id, layer_index: i,
              min_component_area: minAreaRef.current,
            })
          )
        );
        const urls = results.map(r => r.data.result_url);
        setDenoiseLayers(urls);
        setMultiVersion(v => v + 1);
        setDenoiseResult(results[0]?.data ?? null);
      } else {
        const { data } = await apiClient.post<PipelineStepResponse>('/pipeline/denoise', {
          image_id: id, min_component_area: minAreaRef.current,
        });
        setDenoiseResult(data);
      }
      setDenoiseVersion(v => v + 1);
      setDenoiseLineartSnap(cannyGenRef.current);
      setDenoiseGen(g => g + 1);
    } catch { alert('降噪处理失败'); }
    finally { setDenoiseLoading(false); }
  }, []);

  const callSvg = useCallback(async () => {
    const id = imageIdRef.current;
    if (!id || pipelineModeRef.current === null) return;
    setSvgLoading(true);
    try {
      if (pipelineModeRef.current === 'multi') {
        const { data } = await apiClient.post<MultiLayerSvgResponse>('/pipeline/svg/multi-layer', {
          image_id: id,
          n_layers: multiLayersRef.current.length,
          simplify_tolerance: simplRef.current,
        } as MultiLayerSvgParams);
        setMultiSvgResult(data);
        // Also set single-layer svgResult for backward compatibility
        setSvgResult({ svg_url: data.svg_url, total_paths: data.total_paths, total_points: data.total_points, processing_time_ms: data.processing_time_ms });
      } else {
        const { data } = await apiClient.post<SvgResponse>('/pipeline/svg', {
          image_id: id, simplify_tolerance: simplRef.current,
        });
        setSvgResult(data);
      }
      setSvgLineartSnap(cannyGenRef.current);
      setSvgDenoiseSnap(denoiseGenRef.current);
      setSvgConnectivitySnap(connectivityResultRef.current ? connectivityGenRef.current : null);
      setSvgVersion(v => v + 1);
    } catch { alert('SVG 生成失败'); }
    finally { setSvgLoading(false); }
  }, []);

  /* ── Manual trigger ────────────────────────────────────── */
  const handleGenerate = useCallback((type: 'segment' | 'canny' | 'denoise' | 'connectivity' | 'svg') => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (type === 'segment') callSegment();
    else if (type === 'canny') callCanny();
    else if (type === 'denoise') callDenoise();
    else if (type === 'connectivity') callConnectivity();
    else callSvg();
  }, [callSegment, callCanny, callDenoise, callConnectivity, callSvg]);

  /* ── Debounced auto on param change (AI_RULES §7) ──────── */
  // NOTE: segment has NO debounce — SAM is too slow (>30s) for auto-trigger
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (!imageId || !pipelineMode) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => callCanny(), 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [cannyCfg, imageId, pipelineMode, callCanny]);

  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (!imageId) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => callConnectivity(), 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [gapTolerance, callConnectivity]);

  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (!imageId) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => callDenoise(), 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [minComponentArea, callDenoise]);

  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (!imageId) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => callSvg(), 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [simplifyTolerance, callSvg]);

  /* ── Derived: step statuses ────────────────────────────── */
  const stepStatuses = useMemo<Record<PipelineStep, StepStatus>>(() => {
    const segment: StepStatus = !segmentResult ? 'idle'
      : segmentLoading ? 'loading'
      : 'done';  // segment has no upstream dependency

    const canny: StepStatus = !imageId ? 'idle'
      : cannyLoading ? 'loading'
      : pipelineMode === 'multi' && cannySegmentSnap !== segmentGen ? 'stale'
      : overlayUrl ? 'done'
      : 'idle';

    const denoise: StepStatus = !denoiseResult ? 'idle'
      : denoiseLoading ? 'loading'
      : denoiseCannySnap !== cannyGen ? 'stale'
      : 'done';

    const connectivity: StepStatus = !connectivityResult ? 'idle'
      : connectivityLoading ? 'loading'
      : connectivityDenoiseSnap !== denoiseGen ? 'stale'
      : 'done';

    const svg: StepStatus = !svgResult ? 'idle'
      : svgLoading ? 'loading'
      : (svgCannySnap !== cannyGen ||
         svgDenoiseSnap !== denoiseGen ||
         (svgConnectivitySnap !== null && svgConnectivitySnap !== connectivityGen))
        ? 'stale'
      : 'done';

    return { segment, canny, denoise, connectivity, svg };
  }, [imageId, pipelineMode,
      segmentResult, segmentLoading, segmentGen,
      cannyLoading, overlayUrl, cannySegmentSnap, cannyGen,
      denoiseResult, denoiseLoading, denoiseCannySnap, denoiseGen,
      connectivityResult, connectivityLoading, connectivityDenoiseSnap,
      svgResult, svgLoading, svgCannySnap, svgDenoiseSnap, svgConnectivitySnap, connectivityGen]);

  /* ── Progress bar steps ────────────────────────────────── */
  const progressSteps: StepInfo[] = useMemo(() => [
    { key: 'upload', label: '上传', status: imageUrl ? 'done' : 'idle' },
    { key: 'segment', label: 'SAM 分割', status: stepStatuses.segment },
    { key: 'canny', label: '线稿提取', status: stepStatuses.canny },
    { key: 'denoise', label: '降噪', status: stepStatuses.denoise },
    { key: 'connectivity', label: '连通修复', status: stepStatuses.connectivity },
    { key: 'svg', label: 'SVG 生成', status: stepStatuses.svg },
  ], [imageUrl, stepStatuses]);

  /* ── Navigation guards ─────────────────────────────────── */
  const canPrev = pipelineMode !== null && STEP_ORDER.indexOf(currentStep) > 0;
  const canNext = pipelineMode !== null &&
    stepStatuses[currentStep] === 'done' &&
    STEP_ORDER.indexOf(currentStep) < STEP_ORDER.length - 1;

  /* ── Canvas layers (computed from currentStep) ─────────── */
  const stepLabels = STEP_LAYERS[currentStep];
  const vis = layerVisible[currentStep];
  const ops = layerOpacities[currentStep];

  const layers: [LayerCfg, LayerCfg, LayerCfg] = useMemo(() => {
    const s = currentStep;

    // L1 URL — 上一步结果
    let l1Url: string | null;
    switch (s) {
      case 'segment':   l1Url = null; break;
      case 'canny':   l1Url = null; break;
      case 'denoise':   l1Url = overlayUrl; break;
      case 'connectivity': l1Url = denoiseResult?.result_url || overlayUrl; break;
      case 'svg':       l1Url = connectivityResult?.result_url || denoiseResult?.result_url || overlayUrl; break;
      default:          l1Url = null;
    }

    // L2 URL — 当前步骤结果
    let l2Url: string | null;
    switch (s) {
      case 'segment':   l2Url = segmentResult?.overlay_url ?? null; break;
      case 'canny':   l2Url = overlayUrl; break;
      case 'denoise':   l2Url = denoiseResult?.result_url ?? null; break;
      case 'connectivity': l2Url = connectivityResult?.result_url ?? null; break;
      case 'svg':       l2Url = svgResult?.svg_url ?? null; break;
      default:          l2Url = null;
    }

    return [
      { url: imageUrl, label: stepLabels[0], visible: vis[0], opacity: ops[0], disabled: false },
      { url: l1Url, label: stepLabels[1] ?? '', visible: vis[1], opacity: ops[1], disabled: stepLabels[1] === null },
      { url: l2Url, label: stepLabels[2], visible: vis[2], opacity: ops[2], disabled: false },
    ];
  }, [currentStep, imageUrl, overlayUrl, segmentResult, denoiseResult, connectivityResult, svgResult, vis, ops, stepLabels]);

  const isSvgLayer = currentStep === 'svg';

  /* ── Layer toggle handler ──────────────────────────────── */
  const handleToggle = useCallback((idx: 0 | 1 | 2) => {
    setLayerVisible(prev => {
      const cur = prev[currentStep];
      const next: [boolean, boolean, boolean] = [...cur];
      next[idx] = !next[idx];
      return { ...prev, [currentStep]: next };
    });
  }, [currentStep]);

  /* ── Opacity handler ───────────────────────────────────── */
  const handleOpacity = useCallback((idx: 0 | 1 | 2, v: number) => {
    setLayerOpacities(prev => {
      const cur = prev[currentStep];
      const next: [number, number, number] = [...cur];
      next[idx] = v;
      return { ...prev, [currentStep]: next };
    });
  }, [currentStep]);

  /* ── Multi-layer toggle/opacity handlers ────────────────── */
  const handleMultiToggle = useCallback((idx: number) => {
    setMultiLayerVisible(prev => {
      const next = [...prev];
      if (idx < next.length) next[idx] = !next[idx];
      return next;
    });
  }, []);

  const handleMultiOpacity = useCallback((idx: number, v: number) => {
    setMultiLayerOpacities(prev => {
      const next = [...prev];
      if (idx < next.length) next[idx] = v;
      return next;
    });
  }, []);

  /* ── Merged multi-layer data (inject per-step URLs) ────── */
  const mergedMultiLayers: MultiLayerInfo[] = useMemo(() => {
    return multiLayers.map((l, i) => ({
      ...l,
      cannyUrl: cannyLayers[i] ?? l.cannyUrl,
      denoiseUrl: denoiseLayers[i] ?? l.denoiseUrl,
      connectivityUrl: connectivityLayers[i] ?? l.connectivityUrl,
    }));
  }, [multiLayers, cannyLayers, denoiseLayers, connectivityLayers]);

  /* ── Render ────────────────────────────────────────────── */
  return (
    <div className="app-layout">
      {/* ── Progress bar (top, full width) ────────────────── */}
      <ProgressBar
        steps={progressSteps}
        currentStep={imageId && pipelineMode ? currentStep : null}
        onStepClick={goToStep}
        onReupload={() => {
          setImageId(null); setImageUrl(null);
          setPipelineMode(null);
          setCurrentStep('canny');
        }}
        hasImage={!!imageId}
      />

      {/* ── Content row ──────────────────────────────────── */}
      <div className="content-row">
        {/* ── Sidebar: mode selector → single-step panel ──── */}
        <aside className="sidebar">
          {imageId && !pipelineMode ? (
            <ModeSelector onSelect={handleSelectMode} />
          ) : imageId ? (
            <ControlPanel
              currentStep={currentStep}
              stepStatuses={stepStatuses}
              // segment
              nLayers={nLayers} onChangeNLayers={setNLayers}
              frameWidth={frameWidth} onChangeFrameWidth={setFrameWidth}
              minIslandArea={minIslandArea} onChangeMinIslandArea={setMinIslandArea}
              samQuality={samQuality} onChangeSamQuality={setSamQuality}
              segmentLoading={segmentLoading} segmentResult={segmentResult}
              segmentElapsed={segmentElapsed}
              // canny
              cannyCfg={cannyCfg}
              onChangeCanny={(k, v) => setCannyCfg(p => ({ ...p, [k]: v }))}
              cannyLoading={cannyLoading}
              cannyElapsed={cannyElapsed}
              // denoise
              minComponentArea={minComponentArea}
              onChangeMinComponentArea={setMinComponentArea}
              denoiseLoading={denoiseLoading}
              denoiseResult={denoiseResult}
              // connectivity
              gapTolerance={gapTolerance}
              onChangeGapTolerance={setGapTolerance}
              connectivityLoading={connectivityLoading}
              connectivityResult={connectivityResult}
              // svg
              simplifyTolerance={simplifyTolerance}
              onChangeSimplify={setSimplifyTolerance}
              svgLoading={svgLoading}
              svgResult={svgResult}
              // multi-layer
              pipelineMode={pipelineMode}
              multiViewMode={multiViewMode}
              onChangeMultiViewMode={setMultiViewMode}
              // actions
              onGenerate={handleGenerate}
              onPrev={goPrev}
              onNext={goNext}
              canPrev={canPrev}
              canNext={canNext}
            />
          ) : (
            <div className="sidebar-empty">
              <p className="hint">请先上传图片</p>
              <ImageUploader onUploaded={handleUploaded} />
            </div>
          )}
        </aside>

        {/* ── Canvas area ────────────────────────────────── */}
        <main className="main-area">
          {imageUrl && pipelineMode === 'multi' && multiLayers.length > 0 ? (
            <MultiLayerCanvas
              layers={mergedMultiLayers}
              currentStep={currentStep}
              originalUrl={imageUrl}
              viewMode={multiViewMode}
              onViewModeChange={setMultiViewMode}
              layerVisible={multiLayerVisible}
              onToggle={handleMultiToggle}
              layerOpacities={multiLayerOpacities}
              onOpacity={handleMultiOpacity}
              version={multiVersion}
              svgResult={currentStep === 'svg' ? svgResult : null}
              multiSvgResult={currentStep === 'svg' && multiSvgResult ? {
                svg_url: multiSvgResult.svg_url,
                total_paths: multiSvgResult.total_paths,
                processing_time_ms: multiSvgResult.processing_time_ms,
              } : null}
            />
          ) : imageUrl ? (
            <Canvas
              layers={layers}
              onToggle={handleToggle}
              onOpacity={handleOpacity}
              version={currentStep === 'canny' ? cannyVersion : currentStep === 'denoise' ? denoiseVersion : currentStep === 'connectivity' ? connectivityVersion : currentStep === 'svg' ? svgVersion : currentStep === 'segment' ? segmentVersion : 0}
              svgLayer={isSvgLayer}
              svgResult={currentStep === 'svg' ? svgResult : null}
            />
          ) : (
            <div className="canvas-empty">
              <p>上传一张图片来开始处理</p>
              <ImageUploader onUploaded={handleUploaded} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

/* ── Mode Selector Component ─────────────────────────────── */

function ModeSelector({ onSelect }: { onSelect: (mode: 'single' | 'multi') => void }) {
  return (
    <div className="control-panel">
      <div className="cp-step-header">
        <span className="cp-step-num">①</span>
        <span className="cp-step-title">选择模式</span>
      </div>
      <div className="cp-step-body">
        <p className="step-hint">请选择处理模式 — 选定后不可更改，需重新上传切换</p>
        <button className="generate-btn mode-btn"
          onClick={() => onSelect('single')}
          style={{ marginBottom: '0.75rem' }}
        >
          📄 单层模式 — 剪纸 / 书签
        </button>
        <button className="generate-btn mode-btn"
          onClick={() => onSelect('multi')}
        >
          📦 多层模式 — 纸雕灯 / 亚克力（SAM AI 分层）
        </button>
      </div>
    </div>
  );
}

export default App;
