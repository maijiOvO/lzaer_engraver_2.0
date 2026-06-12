import type { PipelineStep, StepStatus } from '../App';

/* ── Types ──────────────────────────────────────────────── */

export interface StepInfo {
  key: PipelineStep | 'upload';
  label: string;
  status: StepStatus;
}

interface Props {
  steps: StepInfo[];
  currentStep: PipelineStep | null;
  onStepClick: (step: PipelineStep) => void;
  onReupload: () => void;
  hasImage: boolean;
}

/* ── Status helpers ─────────────────────────────────────── */

const STATUS_ICON: Record<StepStatus, string> = {
  idle: '○',
  loading: '◌',
  done: '✓',
  stale: '⚠',
};

const STATUS_CLASS: Record<StepStatus, string> = {
  idle: 'p-idle',
  loading: 'p-loading',
  done: 'p-done',
  stale: 'p-stale',
};

/* ── Component ──────────────────────────────────────────── */

export default function ProgressBar({
  steps, currentStep, onStepClick, onReupload, hasImage,
}: Props) {
  return (
    <div className="progress-bar">
      <div className="progress-steps">
        {steps.map((step, i) => {
          const isCurrent = step.key === currentStep;
          const isUpload = step.key === 'upload';
          const clickable = !isUpload && step.status !== 'idle' && step.status !== 'loading';
          const isLast = i === steps.length - 1;

          return (
            <div key={step.key} className="progress-item-wrap">
              {/* ── Step node ─────────────────────────── */}
              <button
                className={[
                  'progress-node',
                  STATUS_CLASS[step.status],
                  isCurrent ? 'p-current' : '',
                  clickable ? 'p-clickable' : '',
                ].join(' ')}
                disabled={!clickable}
                onClick={() => {
                  if (isUpload) onReupload();
                  else if (clickable) onStepClick(step.key as PipelineStep);
                }}
                title={isUpload ? '重新上传' : step.label}
              >
                <span className="p-icon">{STATUS_ICON[step.status]}</span>
                <span className="p-label">{step.label}</span>
                {isCurrent && <span className="p-current-dot" />}
              </button>

              {/* ── Connector line ────────────────────── */}
              {!isLast && (
                <div
                  className={[
                    'progress-connector',
                    step.status === 'done' && !isCurrent ? 'conn-done' : '',
                  ].join(' ')}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* ── Re-upload shortcut ──────────────────────────── */}
      {hasImage && (
        <button className="progress-reupload" onClick={onReupload}>
          ↻ 重新上传
        </button>
      )}
    </div>
  );
}
