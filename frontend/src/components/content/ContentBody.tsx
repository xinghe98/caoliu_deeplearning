import { CheckCheck, CircleDashed, Heart, SkipForward, ThumbsDown } from 'lucide-react'
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
  const watched = Boolean(item.is_watched)

  return (
    <>
      <div className="grid gap-7 p-5 md:grid-cols-[minmax(0,1fr)_auto] md:p-7">
        <div>
          <h2 className="page-heading max-w-3xl text-2xl leading-snug break-words">{item.title_clean || '无标题'}</h2>
          <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-sm text-muted">
            <span>模型 {item.model_version || '尚未指定'}</span>
            {item.labeled_at && (item.current_label === 0 || item.current_label === 1) ? (
              <span>
                标注于{' '}
                {(() => {
                  const time = new Date(item.labeled_at)
                  if (Number.isNaN(time.getTime())) return '未知'
                  const y = time.getFullYear()
                  const m = String(time.getMonth() + 1).padStart(2, '0')
                  const d = String(time.getDate()).padStart(2, '0')
                  return `${y}/${m}/${d}`
                })()}
              </span>
            ) : null}
            <span>采集于 {new Date(item.created_at).toLocaleString('zh-CN')}</span>
            <span className={watched ? 'text-teal' : 'text-[oklch(45%_0.12_85)]'}>
              {watched ? '已看过' : '未看过'}
            </span>
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
            tone={watched ? 'watched' : 'unwatched'}
            disabled={busy}
            onClick={onWatched}
            icon={watched ? <CheckCheck size={18} /> : <CircleDashed size={18} />}
            label={watched ? '已看过 (4)' : '未看过 (4)'}
          />
        </div>
        <p className="hidden text-xs text-muted md:block md:pl-7">
          快捷键：1 喜欢 · 2 不喜欢 · 3 跳过 · 4 切换已看过 · M 复制 · ←/→ 切图 · Z 撤销
        </p>
      </div>

      <MobileLabelBar
        busy={busy}
        watched={watched}
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
  watched,
  onLike,
  onDislike,
  onSkip,
  onWatched,
}: {
  busy: boolean
  watched: boolean
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
        <ActionButton
          tone={watched ? 'watched' : 'unwatched'}
          disabled={busy}
          onClick={onWatched}
          icon={watched ? <CheckCheck size={18} /> : <CircleDashed size={18} />}
          label={watched ? '已看过' : '未看过'}
        />
      </div>
    </div>
  )
}
