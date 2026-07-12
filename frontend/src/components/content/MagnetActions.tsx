import { Copy, Link2 } from 'lucide-react'
import { magnetSummary } from '../../api/client'
import type { ContentRead } from '../../api/types'

type MagnetActionsProps = {
  item: ContentRead
  onCopy: () => void
  onOpen: () => void
}

export function MagnetActions({ item, onCopy, onOpen }: MagnetActionsProps) {
  return (
    <div className="mt-6 border-t border-line pt-4">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="min-w-0 flex-1 truncate font-mono text-sm text-muted"
          title={item.magnet_uri || undefined}
        >
          {magnetSummary(item.magnet_uri)}
        </span>
        <button
          type="button"
          className="quiet-button inline-flex shrink-0 items-center gap-1.5 px-3 py-1.5 text-sm"
          onClick={onCopy}
        >
          <Copy size={15} /> 复制链接
        </button>
        <button
          type="button"
          className="quiet-button inline-flex shrink-0 items-center gap-1.5 px-3 py-1.5 text-sm"
          onClick={onOpen}
        >
          <Link2 size={15} /> 下载器打开
        </button>
      </div>
    </div>
  )
}