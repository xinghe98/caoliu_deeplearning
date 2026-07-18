import { CheckCheck, Heart, SkipForward, ThumbsDown } from 'lucide-react'
import type { ContentRead } from '../../api/types'
import { ActionButton } from './ActionButton'
import { MagnetActions } from './MagnetActions'

type ContentBodyProps = {
  item: ContentRead
  busy: boolean
  error: string
  onLike: () => void
  onDislike: () => void
  onSkip: () => void
  onWatched: () => void
  onCopyMagnet: () => void
  onOpenMagnet: () => void
}

export function ContentBody({
  item,
  busy,
  error,
  onLike,
  onDislike,
  onSkip,
  onWatched,
  onCopyMagnet,
  onOpenMagnet,
}: ContentBodyProps) {
  return (
    <>
      <div className="grid gap-7 p-5 md:grid-cols-[minmax(0,1fr)_auto] md:p-7">
        <div>
          <h2 className="page-heading max-w-3xl text-2xl leading-snug break-words">{item.title_clean || '无标题'}</h2>
          <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-sm text-muted">
            <span>模型 {item.model_version || '尚未指定'}</span>
            <span>采集于 {new Date(item.created_at).toLocaleString('zh-CN')}</span>
            {item.is_watched ? <span className="text-teal">已看过</span> : null}
          </div>

          <MagnetActions item={item} onCopy={onCopyMagnet} onOpen={onOpenMagnet} />

          {error ? <p role="alert" className="mt-4 text-sm text-like">{error}</p> : null}
        </div>

        <div className="hidden gap-2 border-l border-line pl-7 md:flex md:w-[270px] md:flex-col">
          <ActionButton
            tone="like"
            disabled={busy}
            onClick={onLike}
            icon={<Heart size={18} />}
            label="喜欢 (1)"
          />
          <ActionButton
            tone="dislike"
            disabled={busy}
            onClick={onDislike}
            icon={<ThumbsDown size={18} />}
            label="不喜欢 (2)"
          />
          <ActionButton
            tone="skip"
            disabled={busy}
            onClick={onSkip}
            icon={<SkipForward size={18} />}
            label="跳过 (3)"
          />
          <ActionButton
            tone="watched"
            disabled={busy}
            onClick={onWatched}
            icon={<CheckCheck size={18} />}
            label="已看过 (4)"
          />
        </div>
        <p className="hidden text-xs text-muted md:block md:pl-7">
          快捷键：1 喜欢 · 2 不喜欢 · 3 跳过 · 4 已看过 · M 复制 · ←/→ 切图 · Z 撤销
        </p>
      </div>

      <MobileLabelBar
        busy={busy}
        onLike={onLike}
        onDislike={onDislike}
        onSkip={onSkip}
        onWatched={onWatched}
      />
    </>
  )
}

function MobileLabelBar({
  busy,
  onLike,
  onDislike,
  onSkip,
  onWatched,
}: {
  busy: boolean
  onLike: () => void
  onDislike: () => void
  onSkip: () => void
  onWatched: () => void
}) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-panel p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] md:hidden">
      <div className="mx-auto grid max-w-lg grid-cols-2 gap-2 sm:grid-cols-4">
        <ActionButton tone="like" disabled={busy} onClick={onLike} icon={<Heart size={18} />} label="喜欢" />
        <ActionButton tone="dislike" disabled={busy} onClick={onDislike} icon={<ThumbsDown size={18} />} label="不喜欢" />
        <ActionButton tone="skip" disabled={busy} onClick={onSkip} icon={<SkipForward size={18} />} label="跳过" />
        <ActionButton tone="watched" disabled={busy} onClick={onWatched} icon={<CheckCheck size={18} />} label="已看过" />
      </div>
    </div>
  )
}
