import type { ConnectivityResponse, PipelineStepResponse, SegmentResponse, SvgResponse, MultiViewMode } from '../types';
import type { PipelineStep, StepStatus } from '../App';

/* ── Types ──────────────────────────────────────────────── */

interface CannyCfg {
  low: number; high: number; smooth_level: number;
}

interface Props {
  currentStep: PipelineStep;
  stepStatuses: Record<PipelineStep, StepStatus>;

  // Segment params (Depth-guided structural layering)
  nLayers: number;
  onChangeNLayers: (v: number) => void;
  frameWidth: number;
  onChangeFrameWidth: (v: number) => void;
  minIslandArea: number;
  onChangeMinIslandArea: (v: number) => void;
  samQuality: string;
  onChangeSamQuality: (v: string) => void;
  segmentLoading: boolean;
  segmentResult: SegmentResponse | null;
  segmentElapsed: number | null;

  // Canny params
  cannyCfg: CannyCfg;
  onChangeCanny: (k: string, v: number) => void;
  cannyLoading: boolean;
  cannyElapsed: number | null;

  // Denoise params
  minComponentArea: number;
  onChangeMinComponentArea: (v: number) => void;
  denoiseLoading: boolean;
  denoiseResult: PipelineStepResponse | null;

  // Connectivity params
  gapTolerance: number;
  onChangeGapTolerance: (v: number) => void;
  connectivityLoading: boolean;
  connectivityResult: ConnectivityResponse | null;

  // SVG params
  simplifyTolerance: number;
  onChangeSimplify: (v: number) => void;
  svgLoading: boolean;
  svgResult: SvgResponse | null;

  // Multi-layer view mode
  pipelineMode: 'single' | 'multi' | null;
  multiViewMode: MultiViewMode;
  onChangeMultiViewMode: (m: MultiViewMode) => void;

  // Actions
  onGenerate: (type: 'segment' | 'canny' | 'denoise' | 'connectivity' | 'svg') => void;
  onPrev: () => void;
  onNext: () => void;
  canPrev: boolean;
  canNext: boolean;
}

/* ── Status helpers ─────────────────────────────────────── */

const STATUS_LABEL: Record<StepStatus, string> = {
  idle: '待处理',
  loading: '处理中…',
  done: '✓ 已完成',
  stale: '⚠ 需更新',
};

const STEP_NUM: Partial<Record<PipelineStep, string>> = {
  segment: '①', canny: '②', denoise: '③', connectivity: '④', svg: '⑤',
};
const STEP_TITLE: Partial<Record<PipelineStep, string>> = {
  segment: '图层分割',
  canny: 'Canny 边缘',
  denoise: '降噪处理',
  connectivity: '连通性修复',
  svg: 'SVG 矢量生成',
};

const SAM_QUALITY_OPTIONS: [string, string][] = [
  ['draft', '快速预览 (跳过SAM精修)'],
  ['standard', '标准质量 (SAM边界精修)'],
  ['fine', '精细导出 (增强边缘)'],
];

/* ── Component ──────────────────────────────────────────── */

export default function ControlPanel(props: Props) {
  const {
    currentStep, stepStatuses,
    // segment
    nLayers, onChangeNLayers, frameWidth, onChangeFrameWidth,
    minIslandArea, onChangeMinIslandArea,
    samQuality, onChangeSamQuality,
    segmentLoading, segmentResult, segmentElapsed,
    // canny
    cannyCfg, onChangeCanny, cannyLoading, cannyElapsed,
    // denoise
    minComponentArea, onChangeMinComponentArea, denoiseLoading, denoiseResult,
    // connectivity
    gapTolerance, onChangeGapTolerance, connectivityLoading, connectivityResult,
    // svg
    simplifyTolerance, onChangeSimplify, svgLoading, svgResult,
    // multi-layer
    pipelineMode, multiViewMode, onChangeMultiViewMode,
    // actions
    onGenerate, onPrev, onNext, canPrev, canNext,
  } = props;

  const status = stepStatuses[currentStep];

  return (
    <div className="control-panel">
      {/* ── Step header ────────────────────────────────── */}
      <div className="cp-step-header">
        <span className="cp-step-num">{STEP_NUM[currentStep] ?? ''}</span>
        <span className="cp-step-title">{STEP_TITLE[currentStep] ?? ''}</span>
        <span className={`step-status status-${status}`}>{STATUS_LABEL[status]}</span>
      </div>

      {/* ── Step body (only current step's params) ──────── */}
      <div className="cp-step-body">
        {currentStep === 'segment' && (
          <>
            <p className="step-hint">深度引导结构分层 — AI 识别空间结构，自动切割为可支撑的图层</p>
            <Slider label={`分割层数: ${nLayers}`} min={2} max={5}
              value={segmentResult ? segmentResult.layers.length : nLayers} disabled={segmentLoading}
              onChange={onChangeNLayers} />
            <Select label="质量预设" value={samQuality}
              options={SAM_QUALITY_OPTIONS}
              disabled={segmentLoading} onChange={onChangeSamQuality} />
            <Slider label={`外框宽度: ${frameWidth} px`}
              min={20} max={200} step={10}
              value={frameWidth} disabled={segmentLoading}
              onChange={onChangeFrameWidth} />
            <Slider label={`孤立岛阈值: ${minIslandArea} px²`}
              min={10} max={5000} step={10}
              value={minIslandArea} disabled={segmentLoading}
              onChange={onChangeMinIslandArea} />
            <Btn loading={segmentLoading}
              label={segmentLoading ? '分割处理中…（深度估计 + SAM精修）' : '开始分割'}
              onClick={() => onGenerate('segment')} />
            {segmentElapsed !== null && !segmentLoading && segmentResult && (
              <Info>{segmentResult.layers.length} 层 · {(segmentElapsed / 1000).toFixed(1)}s</Info>
            )}
          </>
        )}

        {currentStep === 'canny' && (
          <>
            {status === 'stale' && (
              <p className="step-hint">上游分割结果已重新生成，当前结果可能已过期</p>
            )}
            <Slider label={`边缘灵敏度 low: ${cannyCfg.low}`} min={0} max={255}
              value={cannyCfg.low} disabled={cannyLoading}
              onChange={v => onChangeCanny('low', v)} />
            <Slider label={`噪点过滤 high: ${cannyCfg.high}`} min={0} max={255}
              value={cannyCfg.high} disabled={cannyLoading}
              onChange={v => onChangeCanny('high', v)} />
            <Slider label={`平滑等级: ${cannyCfg.smooth_level}`} min={0} max={2}
              value={cannyCfg.smooth_level} disabled={cannyLoading}
              onChange={v => onChangeCanny('smooth_level', v)} />
            <Btn loading={cannyLoading} label="Canny 边缘检测"
              onClick={() => onGenerate('canny')} />
            {cannyElapsed !== null && <Info>耗时: {cannyElapsed} ms</Info>}
          </>
        )}

        {currentStep === 'denoise' && (
          <>
            {status === 'stale' && (
              <p className="step-hint">上游线稿已重新生成，当前结果可能已过期</p>
            )}
            <Slider label={`最小连通域: ${minComponentArea} px²`} min={1} max={100}
              value={minComponentArea} disabled={denoiseLoading}
              onChange={onChangeMinComponentArea} />
            <Btn loading={denoiseLoading} label="降噪处理"
              onClick={() => onGenerate('denoise')} />
            {denoiseResult && <Info>耗时: {denoiseResult.processing_time_ms} ms</Info>}
          </>
        )}

        {currentStep === 'connectivity' && (
          <>
            {status === 'stale' && (
              <p className="step-hint">上游降噪结果已重新生成，当前结果可能已过期</p>
            )}
            <Slider label={`桥接容差: ${gapTolerance} px`} min={1} max={20}
              value={gapTolerance} disabled={connectivityLoading}
              onChange={onChangeGapTolerance} />
            <Btn loading={connectivityLoading} label="修复连通"
              onClick={() => onGenerate('connectivity')} />
            {connectivityResult && (
              <Info>
                缝合 {connectivityResult.bridges_built} px · {connectivityResult.processing_time_ms} ms
              </Info>
            )}
          </>
        )}

        {currentStep === 'svg' && (
          <>
            {status === 'stale' && (
              <p className="step-hint">上游结果已变更，当前 SVG 可能已过期</p>
            )}
            <Slider label={`简化容差: ${simplifyTolerance.toFixed(1)}`}
              min={0.1} max={10} step={0.1}
              value={simplifyTolerance} disabled={svgLoading}
              onChange={onChangeSimplify} />
            <Btn loading={svgLoading} label="生成 SVG"
              onClick={() => onGenerate('svg')} />
            {svgResult && (
              <>
                <Info>
                  {svgResult.total_paths} 路径 · {svgResult.total_points} 锚点 · {svgResult.processing_time_ms} ms
                </Info>
                <span className="svg-dl-hint">SVG 已生成 — 右侧预览区可直接下载</span>
              </>
            )}
          </>
        )}
      </div>

      {/* ── Multi-layer view mode (only in multi-layer mode) ── */}
      {pipelineMode === 'multi' && (
        <div className="view-mode-selector">
          <p className="param-label" style={{ marginBottom: '0.5rem' }}>预览排列:</p>
          <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
            {(['overlay', 'vertical', 'horizontal'] as MultiViewMode[]).map(m => (
              <button
                key={m}
                className={`cp-nav-btn ${multiViewMode === m ? 'cp-nav-btn-active' : ''}`}
                style={multiViewMode === m ? { borderColor: '#e74c3c', color: '#e74c3c' } : {}}
                onClick={() => onChangeMultiViewMode(m)}
              >
                {m === 'overlay' ? '叠加' : m === 'vertical' ? '纵向' : '横向'}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Navigation ──────────────────────────────────── */}
      <div className="cp-nav">
        <button className="cp-nav-btn" disabled={!canPrev} onClick={onPrev}>
          ← 上一步
        </button>
        <button className="cp-nav-btn" disabled={!canNext} onClick={onNext}>
          下一步 →
        </button>
      </div>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────── */

function Slider({ label, min, max, step = 1, value, disabled, onChange }: {
  label: string; min: number; max: number; step?: number;
  value: number; disabled: boolean; onChange: (v: number) => void;
}) {
  return (
    <div className="param-group">
      <label>
        <span className="param-label">{label}</span>
        <input type="range" min={min} max={max} step={step} value={value}
          disabled={disabled} onChange={e => onChange(Number(e.target.value))} />
      </label>
    </div>
  );
}

function Select({ label, value, options, disabled, onChange }: {
  label: string; value: string; options: [string, string][];
  disabled: boolean; onChange: (v: string) => void;
}) {
  return (
    <div className="param-group">
      <label>
        <span className="param-label">{label}</span>
        <select className="param-select" value={value}
          disabled={disabled} onChange={e => onChange(e.target.value)}>
          {options.map(([val, display]) => (
            <option key={val} value={val}>{display}</option>
          ))}
        </select>
      </label>
    </div>
  );
}

function Btn({ loading, label, onClick }: {
  loading: boolean; label: string; onClick: () => void;
}) {
  return (
    <button className="generate-btn" onClick={onClick} disabled={loading}>
      {loading ? '处理中...' : label}
    </button>
  );
}

function Info({ children }: { children: React.ReactNode }) {
  return <p className="elapsed">{children}</p>;
}
