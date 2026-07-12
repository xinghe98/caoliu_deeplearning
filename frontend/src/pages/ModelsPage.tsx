import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { modelsApi, trainingApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import type { ModelRead } from '../api/types'
import { useToast } from '../components/Toast'
import { useState } from 'react'

export function ModelsPage() {
  const queryClient = useQueryClient()
  const { push } = useToast()
  const [comparisonId, setComparisonId] = useState<string | null>(null)

  const models = useQuery({ queryKey: ['models'], queryFn: modelsApi.list })
  const comparison = useQuery({
    queryKey: ['comparison', comparisonId],
    queryFn: () => trainingApi.comparison(comparisonId!),
    enabled: Boolean(comparisonId),
  })

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ['models'] })
  }

  const activate = useMutation({
    mutationFn: ({ id, force }: { id: string; force?: boolean }) => modelsApi.activate(id, force),
    onSuccess: (model) => {
      push({ message: `已发布 ${model.version}` })
      invalidate()
    },
    onError: (err) => push({ message: err instanceof HttpError ? err.detail : '发布失败' }),
  })

  const reject = useMutation({
    mutationFn: (id: string) => modelsApi.reject(id),
    onSuccess: () => {
      push({ message: '已拒绝候选' })
      invalidate()
    },
    onError: (err) => push({ message: err instanceof HttpError ? err.detail : '操作失败' }),
  })

  const rollback = useMutation({
    mutationFn: (id: string) => modelsApi.rollback(id),
    onSuccess: (model) => {
      push({ message: `已回滚到 ${model.version}` })
      invalidate()
    },
    onError: (err) => push({ message: err instanceof HttpError ? err.detail : '回滚失败' }),
  })

  const list = models.data ?? []

  return (
    <div className="space-y-8">
      <div className="border-b border-line pb-5">
        <div className="eyebrow">发布控制</div>
        <h1 className="page-heading mt-1 text-3xl">模型版本</h1>
        <p className="mt-1 text-sm text-muted">候选需人工发布；指标名称保持 PR-AUC / precision / recall 正确显示。</p>
      </div>

      <div className="space-y-3">
        {list.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            onCompare={() => setComparisonId(model.id)}
            onActivate={(force) => activate.mutate({ id: model.id, force })}
            onReject={() => reject.mutate(model.id)}
            onRollback={() => rollback.mutate(model.id)}
            busy={activate.isPending || reject.isPending || rollback.isPending}
          />
        ))}
        {!models.isLoading && list.length === 0 ? (
          <div className="py-14 text-center text-muted">还没有候选模型。先在训练页导入 GPU 训练产物。</div>
        ) : null}
      </div>

      {comparison.data ? (
        <section className="border-y border-line py-6">
          <h2 className="text-lg font-semibold">候选对比</h2>
          <div className="mt-4 grid divide-y divide-line border-y border-line md:grid-cols-2 md:divide-x md:divide-y-0">
            <MetricBlock title={`候选 ${comparison.data.candidate.version}`} model={comparison.data.candidate} />
            <MetricBlock title={comparison.data.active ? `当前 ${comparison.data.active.version}` : '无 active'} model={comparison.data.active} />
          </div>
          {comparison.data.warnings.length ? (
            <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-muted">
              {comparison.data.warnings.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
          {comparison.data.hard_blocks.length ? (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-like">
              {comparison.data.hard_blocks.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}

function ModelCard({
  model,
  onCompare,
  onActivate,
  onReject,
  onRollback,
  busy,
}: {
  model: ModelRead
  onCompare: () => void
  onActivate: (force?: boolean) => void
  onReject: () => void
  onRollback: () => void
  busy: boolean
}) {
  const prAuc = metric(model, 'pr_auc')
  const precision = metric(model, 'precision')
  const recall = metric(model, 'recall')

  return (
    <article className="border-b border-line py-5 last:border-b-0">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">{model.version}</h2>
            <StatusPill status={model.status} />
          </div>
          <div className="mt-2 text-sm text-muted">
            阈值 {model.decision_threshold.toFixed(3)} · 温度 {model.temperature.toFixed(2)}
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-sm">
            <span>PR-AUC {prAuc}</span>
            <span>precision {precision}</span>
            <span>recall {recall}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {model.status === 'candidate' ? (
            <>
              <button type="button" className="quiet-button" onClick={onCompare}>
                对比
              </button>
              <button
                type="button"
                disabled={busy}
                className="primary-button disabled:opacity-50"
                onClick={() => onActivate(false)}
              >
                发布
              </button>
              <button
                type="button"
                disabled={busy}
                className="quiet-button disabled:opacity-50"
                onClick={() => onActivate(true)}
              >
                强制发布
              </button>
              <button
                type="button"
                disabled={busy}
                className="quiet-button disabled:opacity-50"
                onClick={onReject}
              >
                拒绝
              </button>
            </>
          ) : null}
          {model.status === 'archived' ? (
            <button
              type="button"
              disabled={busy}
              className="quiet-button disabled:opacity-50"
              onClick={onRollback}
            >
              回滚到此版本
            </button>
          ) : null}
        </div>
      </div>
    </article>
  )
}

function MetricBlock({ title, model }: { title: string; model: ModelRead | null }) {
  return (
    <div className="p-4">
      <div className="font-medium">{title}</div>
      {model ? (
        <div className="mt-2 space-y-1 text-sm text-muted">
          <div>PR-AUC {metric(model, 'pr_auc')}</div>
          <div>precision {metric(model, 'precision')}</div>
          <div>recall {metric(model, 'recall')}</div>
        </div>
      ) : (
        <div className="mt-2 text-sm text-muted">—</div>
      )}
    </div>
  )
}

function metric(model: ModelRead, key: string): string {
  const value = model.metrics?.[key]
  return typeof value === 'number' ? value.toFixed(4) : '—'
}

function StatusPill({ status }: { status: string }) {
  const tone =
    status === 'active'
      ? 'bg-teal-soft text-teal'
      : status === 'candidate'
        ? 'bg-canvas text-ink'
        : 'bg-line text-muted'
  return <span className={`rounded-full px-2 py-0.5 text-xs ${tone}`}>{status}</span>
}
