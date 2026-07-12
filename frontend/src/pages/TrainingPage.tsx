import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { trainingApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import { useToast } from '../components/Toast'

export function TrainingPage() {
  const queryClient = useQueryClient()
  const { push } = useToast()
  const fileRef = useRef<HTMLInputElement>(null)
  const [version, setVersion] = useState('')

  const status = useQuery({ queryKey: ['training-status'], queryFn: trainingApi.status })
  const snapshots = useQuery({ queryKey: ['training-snapshots'], queryFn: trainingApi.snapshots })

  const createSnapshot = useMutation({
    mutationFn: trainingApi.createSnapshot,
    onSuccess: () => {
      push({ message: '训练快照已生成' })
      void queryClient.invalidateQueries({ queryKey: ['training-status'] })
      void queryClient.invalidateQueries({ queryKey: ['training-snapshots'] })
    },
    onError: (err) => push({ message: err instanceof HttpError ? err.detail : '生成失败' }),
  })

  const importCandidate = useMutation({
    mutationFn: async (file: File) => trainingApi.importCandidate(file, version || undefined),
    onSuccess: (model) => {
      push({ message: `候选已导入：${model.version}` })
      void queryClient.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (err) => push({ message: err instanceof HttpError ? err.detail : '导入失败' }),
  })

  const s = status.data
  const progress = s ? Math.min(100, Math.round((s.labels_since_last_snapshot / Math.max(s.threshold, 1)) * 100)) : 0

  return (
    <div className="space-y-8">
      <div className="border-b border-line pb-5">
        <div className="eyebrow">训练数据</div>
        <h1 className="page-heading mt-1 text-3xl">训练生命周期</h1>
        <p className="mt-1 text-sm text-muted">累计明确标签达阈值后生成不可变训练包，再手动上传 GPU 训练产物。</p>
      </div>

      <section className="border-y border-line py-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-sm text-muted">距离下一训练包</div>
            <div className="mt-1 text-3xl font-semibold">
              {s ? `${s.labels_since_last_snapshot} / ${s.threshold}` : '—'}
            </div>
          </div>
          <button
            type="button"
            className="primary-button disabled:opacity-50"
            disabled={createSnapshot.isPending}
            onClick={() => createSnapshot.mutate()}
          >
            立即生成快照
          </button>
        </div>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-canvas">
          <div className="h-full bg-teal" style={{ width: `${progress}%` }} />
        </div>
        {s?.ready_for_snapshot ? (
          <p className="mt-3 text-sm text-teal">已达阈值，可生成或等待自动 export job。</p>
        ) : null}
      </section>

      <section className="border-y border-line py-6">
        <h2 className="text-lg font-semibold">导入候选模型</h2>
        <p className="mt-1 text-sm text-muted">上传 `candidate_*.zip`（含 best_model.pth 与 evaluation_report.json）。</p>
        <div className="mt-4 flex flex-wrap gap-3">
          <input
            className="min-h-11 rounded-xl border border-line bg-panel px-3 focus:border-teal"
            placeholder="可选版本名"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
          />
          <input
            ref={fileRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) importCandidate.mutate(file)
            }}
          />
          <button
            type="button"
            className="quiet-button"
            disabled={importCandidate.isPending}
            onClick={() => fileRef.current?.click()}
          >
            {importCandidate.isPending ? '上传中…' : '选择 ZIP'}
          </button>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">训练快照</h2>
        {(snapshots.data ?? []).map((snap) => (
          <div key={snap.id} className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-4 last:border-b-0">
            <div>
              <div className="font-medium">{snap.id.slice(0, 8)}…</div>
              <div className="mt-1 text-sm text-muted">
                {snap.sample_count} 样本 · 正 {snap.positive_count} / 负 {snap.negative_count} · {snap.status}
              </div>
              <div className="mt-1 font-mono text-xs text-muted">{snap.manifest_hash.slice(0, 16)}…</div>
            </div>
            <a
              className="quiet-button inline-flex items-center"
              href={trainingApi.downloadUrl(snap.id)}
            >
              下载 ZIP
            </a>
          </div>
        ))}
        {!snapshots.isLoading && (snapshots.data?.length ?? 0) === 0 ? (
          <div className="py-12 text-center text-muted">还没有训练快照。积累人工喜欢和不喜欢标签后即可生成。</div>
        ) : null}
      </section>
    </div>
  )
}
