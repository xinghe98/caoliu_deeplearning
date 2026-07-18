import { useCallback, useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Clapperboard } from 'lucide-react'
import { contentApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import type { ContentRead } from '../api/types'
import { ContentArticle } from '../components/content/ContentArticle'
import { useContentLabeling } from '../hooks/useContentLabeling'

function scoreText(item: ContentRead) {
  if (item.probability == null) return '待预测'
  return `${(item.probability * 100).toFixed(1)}%`
}

export function ReviewPage() {
  const queryClient = useQueryClient()
  const [imageIndex, setImageIndex] = useState(0)

  const feedQuery = useQuery({
    queryKey: ['feed', 'mixed'],
    queryFn: () => contentApi.feed('mixed', 20),
  })

  const items = feedQuery.data ?? []
  const current = items[0] ?? null

  useEffect(() => {
    setImageIndex(0)
  }, [current?.id])

  useEffect(() => {
    if (!current) return
    void contentApi.event(current.id, 'view').catch(() => undefined)
  }, [current?.id])

  const invalidate = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['feed'] })
    void queryClient.invalidateQueries({ queryKey: ['labels'] })
    void queryClient.invalidateQueries({ queryKey: ['contents'] })
    void queryClient.invalidateQueries({ queryKey: ['training-status'] })
  }, [queryClient])

  const consumeFeedItem = useCallback(async (contentId: ContentRead['id']) => {
    await queryClient.cancelQueries({ queryKey: ['feed'] })
    queryClient.setQueriesData<ContentRead[]>(
      { queryKey: ['feed'] },
      (cached) => cached?.filter((item) => item.id !== contentId),
    )
    invalidate()
  }, [invalidate, queryClient])

  const { busy, error, like, dislike, skip, markWatched, copyMagnet, openMagnet } = useContentLabeling({
    current,
    setImageIndex,
    onLabeled: consumeFeedItem,
    onSkipped: consumeFeedItem,
    onWatched: consumeFeedItem,
    onUndo: invalidate,
  })

  const remaining = items.length

  if (feedQuery.isLoading) {
    return <div className="page-shell max-w-6xl"><Skeleton /></div>
  }

  if (feedQuery.isError) {
    return (
      <div className="mx-auto max-w-xl border border-like/20 bg-like-soft p-5 text-like">
        <h1 className="page-heading text-xl">无法载入待筛选内容</h1>
        <p className="mt-2 text-sm">{feedQuery.error instanceof HttpError ? feedQuery.error.detail : '请检查连接后刷新页面再试。'}</p>
      </div>
    )
  }

  if (!current) {
    return (
      <div className="mx-auto max-w-xl py-20 text-center">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-teal-soft text-teal"><Clapperboard size={24} /></div>
        <h1 className="page-heading mt-5 text-2xl">这一批已经完成</h1>
        <p className="mx-auto mt-3 max-w-md text-muted">暂时没有未标注内容。可稍后等待新内容入库，或在内容库中浏览全部记录。</p>
        <button
          type="button"
          className="quiet-button mt-6"
          onClick={() => void feedQuery.refetch()}
        >
          刷新队列
        </button>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl pb-40 md:pb-10">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
        <div>
          <div className="eyebrow">今日浏览</div>
          <h1 className="page-heading mt-1 text-3xl">待筛选</h1>
          <p className="mt-2 text-sm text-muted">本批还有 {remaining} 条，模型建议仅作排序参考。</p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1.5 text-sm text-muted">
          <span className={`status-dot ${current.probability == null ? 'text-skip' : 'text-teal'}`} />
          {current.probability == null ? '等待模型预测' : `模型建议 ${scoreText(current)}`}
        </div>
      </div>

      <ContentArticle
        item={current}
        imageIndex={imageIndex}
        onImageIndexChange={setImageIndex}
        busy={busy}
        error={error}
        onLike={like}
        onDislike={dislike}
        onSkip={skip}
        onWatched={markWatched}
        onCopyMagnet={() => void copyMagnet(current)}
        onOpenMagnet={() => void openMagnet(current)}
      />
    </div>
  )
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 w-40 bg-line" />
      <div className="h-[60vh] bg-line" />
      <div className="h-24 bg-line" />
    </div>
  )
}
