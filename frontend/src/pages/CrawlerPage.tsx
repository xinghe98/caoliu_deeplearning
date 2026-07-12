import { useQuery } from '@tanstack/react-query'
import { healthApi, jobsApi } from '../api/endpoints'

export function CrawlerPage() {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: jobsApi.list, refetchInterval: 5000 })
  const stats = useQuery({ queryKey: ['job-stats'], queryFn: jobsApi.stats, refetchInterval: 3000 })
  const worker = useQuery({ queryKey: ['health-worker'], queryFn: healthApi.worker, refetchInterval: 5000 })

  const s = stats.data
  const scored = s?.scored_contents ?? 0
  const total = s?.contents_total ?? 0
  const pct = total > 0 ? Math.min(100, Math.round((scored / total) * 100)) : 0
  const remaining = (s?.predict_pending ?? 0) + (s?.predict_running ?? 0)

  return (
    <div className="space-y-8">
      <div className="border-b border-line pb-5">
        <div className="eyebrow">运行状态</div>
        <h1 className="page-heading mt-1 text-3xl">任务与抓取</h1>
      </div>

      <section className="border-y border-line py-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-semibold">模型打分进度</h2>
            <p className="mt-1 text-sm text-muted">
              模型 {s?.active_model_version || '—'} · 已完成 {scored}/{total}
              {remaining > 0 ? ` · 排队中 ${remaining}` : ''}
            </p>
          </div>
          <div className="text-2xl font-semibold tabular-nums text-teal">{pct}%</div>
        </div>
        <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-canvas">
          <div className="h-full rounded-full bg-teal transition-all duration-500" style={{ width: `${pct}%` }} />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted sm:grid-cols-4">
          <span>成功 {s?.predict_succeeded ?? '—'}</span>
          <span>运行中 {s?.predict_running ?? '—'}</span>
          <span>等待 {s?.predict_pending ?? '—'}</span>
          <span>失败 {s?.predict_failed ?? '—'}</span>
        </div>
        {remaining === 0 && total > 0 && scored > 0 ? (
          <p className="mt-3 text-sm text-teal">当前队列已清空。新入库内容会自动排队打分。</p>
        ) : null}
        {total > 0 && scored === 0 && remaining === 0 ? (
          <p className="mt-3 text-sm text-like">
            尚无打分任务。可运行：python -m platform_app.requeue_predictions
          </p>
        ) : null}
      </section>

      <section className="border-y border-line py-5">
        <h2 className="font-semibold">Worker 心跳</h2>
        <div className="mt-3 space-y-2">
          {(worker.data?.workers ?? []).map((row) => (
            <div key={row.worker_id} className="text-sm text-muted">
              {row.worker_id} · 模型 {row.model_version || '—'} · {new Date(row.last_seen_at).toLocaleString()}
            </div>
          ))}
          {!worker.isLoading && (worker.data?.workers.length ?? 0) === 0 ? (
            <div className="text-sm text-muted">暂无 worker 心跳。请运行 python -m platform_app.worker</div>
          ) : null}
        </div>
      </section>

      <section className="overflow-hidden border-y border-line bg-panel">
        <div className="border-b border-line py-3 font-semibold">最近任务</div>
        <div className="divide-y divide-line">
          {(jobs.data ?? []).map((job) => (
            <div key={job.id} className="py-4">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium">{job.job_type}</span>
                <span className="rounded-full bg-canvas px-2 py-0.5 text-xs">{job.status}</span>
                <span className="text-muted">attempts {job.attempts}</span>
              </div>
              <div className="mt-1 text-xs text-muted">{new Date(job.created_at).toLocaleString()}</div>
              {job.last_error ? <div className="mt-1 text-xs text-like">{job.last_error}</div> : null}
            </div>
          ))}
          {!jobs.isLoading && (jobs.data?.length ?? 0) === 0 ? (
            <div className="py-12 text-center text-muted">暂无任务。新内容入库后会在这里显示预测与训练任务。</div>
          ) : null}
        </div>
      </section>
    </div>
  )
}
