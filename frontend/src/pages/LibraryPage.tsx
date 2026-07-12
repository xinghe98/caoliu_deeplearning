import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { ArrowUp } from 'lucide-react'
import { contentApi, jobsApi } from '../api/endpoints'
import { AuthImage } from '../components/AuthImage'
import type { ContentRead } from '../api/types'

const PAGE_SIZE = 24

function labelText(item: ContentRead) {
  if (item.current_label === 1) return '喜欢'
  if (item.current_label === 0) return '不喜欢'
  return '未标注'
}

function scoreText(item: ContentRead) {
  if (item.probability == null) return '待预测'
  return `${(item.probability * 100).toFixed(0)}%`
}

export function LibraryPage() {
  const [label, setLabel] = useState<'all' | '1' | '0' | 'unlabeled'>('all')
  const [showTop, setShowTop] = useState(false)
  const sentinelRef = useRef<HTMLDivElement | null>(null)

  const query = useInfiniteQuery({
    queryKey: ['contents', label],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => {
      if (label === '1') return contentApi.list({ label: 1, limit: PAGE_SIZE, cursor: pageParam })
      if (label === '0') return contentApi.list({ label: 0, limit: PAGE_SIZE, cursor: pageParam })
      if (label === 'unlabeled') return contentApi.list({ unlabeled: true, limit: PAGE_SIZE, cursor: pageParam })
      return contentApi.list({ limit: PAGE_SIZE, cursor: pageParam })
    },
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: 15_000,
  })

  const stats = useQuery({
    queryKey: ['job-stats'],
    queryFn: jobsApi.stats,
    refetchInterval: 5000,
  })

  const items = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data])
  const scored = stats.data?.scored_contents ?? 0
  const total = stats.data?.contents_total ?? 0
  const remaining = (stats.data?.predict_pending ?? 0) + (stats.data?.predict_running ?? 0)
  const pct = total > 0 ? Math.min(100, Math.round((scored / total) * 100)) : 0

  useEffect(() => {
    const onScroll = () => setShowTop(window.scrollY > 400)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const node = sentinelRef.current
    if (!node || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return
        if (query.hasNextPage && !query.isFetchingNextPage) {
          void query.fetchNextPage()
        }
      },
      { rootMargin: '400px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [query.hasNextPage, query.isFetchingNextPage, query.fetchNextPage, label, items.length])

  return (
    <div className="space-y-7">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
        <div>
          <div className="eyebrow">全部归档</div>
          <h1 className="page-heading mt-1 text-3xl">内容库</h1>
          <p className="mt-2 text-sm text-muted">按人工标注筛选，模型分数只显示为辅助信息。</p>
        </div>
        <div className="flex flex-wrap gap-1 rounded-xl bg-canvas p-1">
          {[
            ['all', '全部'],
            ['1', '喜欢'],
            ['0', '不喜欢'],
            ['unlabeled', '未标注'],
          ].map(([value, text]) => (
            <button
              key={value}
              type="button"
              onClick={() => setLabel(value as typeof label)}
              className={[
                'min-h-10 rounded-lg px-3.5 py-1.5 text-sm transition-colors',
                label === value ? 'bg-panel font-medium text-ink shadow-sm' : 'text-muted hover:text-ink',
              ].join(' ')}
            >
              {text}
            </button>
          ))}
        </div>
      </div>

      {remaining > 0 || (total > 0 && scored < total) ? (
        <div className="border-y border-line py-3">
          <div className="flex items-center justify-between gap-3 text-xs text-muted">
            <span>模型打分 {scored}/{total}{remaining > 0 ? `，仍有 ${remaining} 条等待处理` : ''}</span>
            <span className="tabular-nums text-teal">{pct}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-canvas">
            <div className="h-full rounded-full bg-teal transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      ) : null}

      {query.isLoading ? <LibrarySkeleton /> : null}
      {query.isError ? <p role="alert" className="border border-like/20 bg-like-soft p-4 text-sm text-like">内容库无法加载，请刷新后重试。</p> : null}

      {!query.isLoading ? <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 sm:gap-x-5 lg:grid-cols-4 xl:grid-cols-5">
        {items.map((item) => {
          const cover = item.media[0]
          return (
            <Link
              key={item.id}
              to={`/library/${item.id}`}
              target="_blank"
              rel="noreferrer"
              className="content-card group block cursor-pointer overflow-hidden"
            >
              <div className="aspect-[4/3] overflow-hidden rounded-lg bg-canvas">
                {cover ? (
                  <AuthImage contentId={item.id} mediaId={cover.id} lazy className="h-full w-full" />
                ) : (
                  <div className="grid h-full place-items-center text-xs text-muted">无图</div>
                )}
              </div>
              <div className="space-y-2 px-0.5 pt-3 pb-1 sm:px-1">
                <h2 className="line-clamp-2 text-sm leading-snug font-medium text-ink group-hover:text-teal">
                  {item.title_clean || '无标题'}
                </h2>
                <div className="flex items-center justify-between gap-2 text-xs text-muted">
                  <span className={`truncate ${item.current_label === 1 ? 'text-like' : item.current_label === 0 ? 'text-dislike' : ''}`}>{labelText(item)}</span>
                  <span className="shrink-0 tabular-nums">{scoreText(item)}</span>
                </div>
              </div>
            </Link>
          )
        })}
      </div> : null}

      <div ref={sentinelRef} className="flex h-10 items-center justify-center">
        {query.isFetchingNextPage ? (
          <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-line border-t-teal" />
        ) : null}
      </div>

      {!query.isLoading && items.length === 0 ? (
        <div className="mx-auto max-w-md py-16 text-center text-muted">
          <p className="text-lg font-medium text-ink">这里还没有符合条件的内容</p>
          <p className="mt-2 text-sm">尝试切换筛选条件，或等待爬虫将新内容导入。</p>
        </div>
      ) : null}

      {showTop ? (
        <button
          type="button"
          aria-label="回到顶部"
          className="fixed right-5 bottom-6 z-40 grid h-11 w-11 cursor-pointer place-items-center rounded-full border border-line bg-panel text-ink shadow-[0_8px_24px_oklch(20%_0.02_270_/_0.12)] transition-colors hover:border-teal hover:bg-teal-soft hover:text-teal md:bottom-8"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        >
          <ArrowUp size={20} />
        </button>
      ) : null}
    </div>
  )
}

function LibrarySkeleton() {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 sm:gap-x-5 lg:grid-cols-4 xl:grid-cols-5" aria-label="正在加载内容库">
      {Array.from({ length: 10 }, (_, index) => <div key={index} className="animate-pulse"><div className="aspect-[4/3] rounded-lg bg-line" /><div className="mt-3 h-4 w-4/5 bg-line" /><div className="mt-2 h-3 w-2/5 bg-line" /></div>)}
    </div>
  )
}
