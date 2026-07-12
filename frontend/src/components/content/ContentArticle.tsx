import type { ContentRead } from '../../api/types'
import { ContentBody } from './ContentBody'
import { ContentImageGallery } from './ContentImageGallery'

type ContentArticleProps = {
  item: ContentRead
  imageIndex: number
  onImageIndexChange: (index: number) => void
  busy: boolean
  error: string
  onLike: () => void
  onDislike: () => void
  onSkip: () => void
  onCopyMagnet: () => void
  onOpenMagnet: () => void
}

export function ContentArticle({
  item,
  imageIndex,
  onImageIndexChange,
  busy,
  error,
  onLike,
  onDislike,
  onSkip,
  onCopyMagnet,
  onOpenMagnet,
}: ContentArticleProps) {
  const media = item.media ?? []

  return (
    <article className="overflow-hidden border border-line bg-panel">
      <ContentImageGallery
        contentId={item.id}
        title={item.title_clean}
        media={media}
        imageIndex={imageIndex}
        onImageIndexChange={onImageIndexChange}
      />
      <ContentBody
        item={item}
        busy={busy}
        error={error}
        onLike={onLike}
        onDislike={onDislike}
        onSkip={onSkip}
        onCopyMagnet={onCopyMagnet}
        onOpenMagnet={onOpenMagnet}
      />
    </article>
  )
}