import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { MediaRead } from '../../api/types'
import { AuthImage } from '../AuthImage'

type ContentImageGalleryProps = {
  contentId: string
  title: string
  media: MediaRead[]
  imageIndex: number
  onImageIndexChange: (index: number) => void
}

export function ContentImageGallery({
  contentId,
  title,
  media,
  imageIndex,
  onImageIndexChange,
}: ContentImageGalleryProps) {
  return (
    <>
      <div className="relative w-full bg-[oklch(94%_0.008_270)]">
        {media.length > 0 ? (
          media.map((item, i) => (
            <div
              key={item.id}
              aria-hidden={i !== imageIndex}
              className={[
                'flex w-full items-center justify-center transition-opacity duration-200',
                i === imageIndex
                  ? 'relative z-10 opacity-100'
                  : 'pointer-events-none absolute inset-0 z-0 opacity-0',
              ].join(' ')}
            >
              <AuthImage
                contentId={contentId}
                mediaId={item.id}
                alt={title}
                lazy={false}
                className="mx-auto max-h-[68vh] w-full"
                imgClassName="mx-auto max-h-[68vh] w-full object-contain"
              />
            </div>
          ))
        ) : (
          <div className="grid h-80 place-items-center text-muted">图片缺失</div>
        )}
        {media.length > 1 ? (
          <>
            <button
              type="button"
              className="absolute top-1/2 left-4 z-20 grid min-h-11 min-w-11 -translate-y-1/2 cursor-pointer place-items-center rounded-full bg-panel text-ink shadow-sm"
              onClick={() => onImageIndexChange(Math.max(0, imageIndex - 1))}
              aria-label="上一张"
            >
              <ChevronLeft className="mx-auto" size={20} />
            </button>
            <button
              type="button"
              className="absolute top-1/2 right-4 z-20 grid min-h-11 min-w-11 -translate-y-1/2 cursor-pointer place-items-center rounded-full bg-panel text-ink shadow-sm"
              onClick={() => onImageIndexChange(Math.min(media.length - 1, imageIndex + 1))}
              aria-label="下一张"
            >
              <ChevronRight className="mx-auto" size={20} />
            </button>
          </>
        ) : null}
      </div>

      {media.length > 1 ? (
        <div className="flex gap-2 overflow-x-auto border-b border-line px-4 py-3">
          {media.map((item, i) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onImageIndexChange(i)}
              className={[
                'h-16 w-16 shrink-0 cursor-pointer overflow-hidden rounded-lg border',
                i === imageIndex ? 'border-teal' : 'border-line',
              ].join(' ')}
            >
              <AuthImage contentId={contentId} mediaId={item.id} lazy={false} className="h-full w-full" />
            </button>
          ))}
        </div>
      ) : null}
    </>
  )
}
