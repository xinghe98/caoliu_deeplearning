import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { labelsApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import { useToast } from '../components/Toast'

export function LabelsPage() {
  const queryClient = useQueryClient()
  const { push } = useToast()
  const history = useQuery({
    queryKey: ['labels', 'history'],
    queryFn: () => labelsApi.history(),
  })

  const undo = useMutation({
    mutationFn: (eventId: string) => labelsApi.undo(eventId),
    onSuccess: () => {
      push({ message: '已撤销标签' })
      void queryClient.invalidateQueries({ queryKey: ['labels'] })
      void queryClient.invalidateQueries({ queryKey: ['feed'] })
      void queryClient.invalidateQueries({ queryKey: ['contents'] })
    },
    onError: (err) => {
      push({ message: err instanceof HttpError ? err.detail : '撤销失败' })
    },
  })

  const items = history.data ?? []
  const likes = items.filter((item) => item.label === 1).length
  const dislikes = items.filter((item) => item.label === 0).length

  return (
    <div className="space-y-7">
      <div className="border-b border-line pb-5">
        <div className="eyebrow">人工反馈</div>
        <h1 className="page-heading mt-1 text-3xl">标签历史</h1>
        <p className="mt-1 text-sm text-muted">仅明确喜欢/不喜欢进入训练；跳过不在此列。</p>
      </div>

      <div className="grid divide-y divide-line border-y border-line sm:grid-cols-3 sm:divide-x sm:divide-y-0">
        <Stat label="记录数" value={String(items.length)} />
        <Stat label="喜欢" value={String(likes)} />
        <Stat label="不喜欢" value={String(dislikes)} />
      </div>

      <div className="overflow-hidden border-y border-line bg-panel">
        <div className="divide-y divide-line">
          {items.map((item) => (
            <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 px-1 py-4 sm:px-3">
              <div>
                <div className="text-sm font-medium">
                  {item.label === 1 ? '喜欢' : '不喜欢'}
                  <span className="ml-2 text-muted">· {item.source}</span>
                </div>
                <div className="mt-1 text-xs text-muted">
                  {item.content_id.slice(0, 8)}… · {new Date(item.created_at).toLocaleString()}
                  {item.model_version ? ` · 模型 ${item.model_version}` : ''}
                  {item.probability_at_label != null ? ` · p=${item.probability_at_label.toFixed(3)}` : ''}
                </div>
              </div>
              <button
                type="button"
                className="quiet-button"
                disabled={undo.isPending}
                onClick={() => undo.mutate(item.id)}
              >
                撤销
              </button>
            </div>
          ))}
          {!history.isLoading && items.length === 0 ? (
            <div className="px-4 py-12 text-center text-muted">还没有人工标签。去待筛选页给第一条内容做判断。</div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-4 py-5">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
    </div>
  )
}
