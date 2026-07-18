import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { contentApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import type { ContentRead } from '../api/types'
import { ContentArticle } from '../components/content/ContentArticle'
import { useContentLabeling } from '../hooks/useContentLabeling'

function scoreText(item: ContentRead) {
  if (item.probability == null) return '待预测'
  return `${(item.probability * 100).toFixed(1)}%`
}

function labelText(item: ContentRead) {
  if (item.is_watched) return '已看过'
  if (item.current_label === 1) return '喜欢'
  if (item.current_label === 0) return '不喜欢'
  return '未标注'
}

export function ContentDetailPage() {
  const { contentId = '' } = useParams()
  const queryClient = useQueryClient()
  const [imageIndex, setImageIndex] = useState(0)

  const detailQuery = useQuery({
    queryKey: ['content', contentId],
    queryFn: () => contentApi.get(contentId),
    enabled: Boolean(contentId),
  })

  const current = detailQuery.data ?? null

  useEffect(() => {
    setImageIndex(0)
  }, [current?.id])

  useEffect(() => {
    if (!current) return
    void contentApi.event(current.id, 'view').catch(() => undefined)
  }, [current?.id])

  const invalidateAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['content', contentId] })
    void queryClient.invalidateQueries({ queryKey: ['contents'] })
    void queryClient.invalidateQueries({ queryKey: ['feed'] })
    void queryClient.invalidateQueries({ queryKey: ['labels'] })
    void queryClient.invalidateQueries({ queryKey: ['training-status'] })
  }, [contentId, queryClient])

  const consumeLabeledItem = useCallback(async (labeledId: ContentRead['id']) => {
    await queryClient.cancelQueries({ queryKey: ['feed'] })
    queryClient.setQueriesData<ContentRead[]>(
      { queryKey: ['feed'] },
      (cached) => cached?.filter((item) => item.id !== labeledId),
    )
    invalidateAll()
  }, [invalidateAll, queryClient])

  const markWatchedItem = useCallback(async (watchedId: ContentRead['id']) => {
    await queryClient.cancelQueries({ queryKey: ['feed'] })
    queryClient.setQueriesData<ContentRead[]>(
      { queryKey: ['feed'] },
      (cached) => cached?.filter((item) => item.id !== watchedId),
    )
    queryClient.setQueryData<ContentRead>(['content', watchedId], (cached) => (
      cached ? { ...cached, is_watched: true, current_label: null } : cached
    ))
    invalidateAll()
  }, [invalidateAll, queryClient])

  const { busy, error, like, dislike, skip, markWatched, copyMagnet, openMagnet } = useContentLabeling({
    current,
    setImageIndex,
    onLabeled: consumeLabeledItem,
    onSkipped: consumeLabeledItem,
    onWatched: markWatchedItem,
    onUndo: invalidateAll,
  })

  if (detailQuery.isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 w-40 bg-line" />
        <div className="h-[58vh] bg-line" />
        <div className="h-24 bg-line" />
      </div>
    )
  }

  if (detailQuery.isError || !current) {
    return (
      <div className="space-y-4">
        <Link to="/library" className="inline-flex items-center gap-1 text-sm text-muted hover:text-teal">
          <ArrowLeft size={16} /> 返回内容库
        </Link>
        <div role="alert" className="border border-like/20 bg-like-soft p-6 text-like">
          {detailQuery.error instanceof HttpError ? detailQuery.error.detail : '内容不存在或加载失败'}
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl pb-40 md:pb-10">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b border-line pb-5">
        <Link to="/library" className="inline-flex min-h-11 items-center gap-1.5 text-sm text-muted hover:text-teal">
          <ArrowLeft size={16} /> 返回内容库
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-line bg-panel px-3 py-1 text-sm text-muted">人工：{labelText(current)}</span>
          <span className="rounded-full bg-teal-soft px-3 py-1 text-sm text-teal">模型建议：{scoreText(current)}</span>
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
