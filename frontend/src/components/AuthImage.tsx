import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { mediaUrl } from '../api/client'

type AuthImageProps = {
  contentId: string
  mediaId: string
  alt?: string
  className?: string
  imgClassName?: string
  /** 进入视口后再请求；内容库网格建议开启 */
  lazy?: boolean
}

const blobCache = new Map<string, string>()

function cacheKey(contentId: string, mediaId: string) {
  return `${contentId}/${mediaId}`
}

export function AuthImage({
  contentId,
  mediaId,
  alt = '',
  className,
  imgClassName = 'h-full w-full object-cover',
  lazy = true,
}: AuthImageProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const contentIdRef = useRef(contentId)
  const [visible, setVisible] = useState(!lazy)
  const [src, setSrc] = useState<string | null>(() => {
    if (!contentId || !mediaId) return null
    return blobCache.get(cacheKey(contentId, mediaId)) ?? null
  })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!lazy) {
      setVisible(true)
      return
    }
    const node = containerRef.current
    if (!node || typeof IntersectionObserver === 'undefined') {
      setVisible(true)
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true)
          observer.disconnect()
        }
      },
      { rootMargin: '200px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [lazy])

  useLayoutEffect(() => {
    if (!contentId || !mediaId) return
    const cached = blobCache.get(cacheKey(contentId, mediaId))
    if (cached) {
      setSrc(cached)
      setError('')
      setLoading(false)
    }
  }, [contentId, mediaId])

  useEffect(() => {
    if (!visible || !contentId || !mediaId) return

    const key = cacheKey(contentId, mediaId)
    const cached = blobCache.get(key)
    if (cached) {
      setSrc(cached)
      setError('')
      setLoading(false)
      return
    }

    if (contentId !== contentIdRef.current) {
      setSrc(null)
      contentIdRef.current = contentId
    }

    let cancelled = false
    setLoading(true)
    setError('')

    const load = async () => {
      try {
        const response = await fetch(mediaUrl(contentId, mediaId), { credentials: 'include' })
        if (!response.ok) {
          const detail =
            response.status === 401
              ? '未登录或会话过期'
              : response.status === 410
                ? '原图已缺失'
                : `加载失败 (${response.status})`
          throw new Error(detail)
        }
        const blob = await response.blob()
        if (cancelled) return
        const objectUrl = URL.createObjectURL(blob)
        blobCache.set(key, objectUrl)
        setSrc(objectUrl)
        setLoading(false)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '无法加载图片')
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [visible, contentId, mediaId])

  return (
    <div ref={containerRef} className={['relative h-full w-full bg-canvas', className].filter(Boolean).join(' ')}>
      {src ? (
        <img src={src} alt={alt} className={imgClassName} draggable={false} />
      ) : (
        <div className="grid h-full place-items-center px-3 text-center text-xs text-muted">
          {error || (visible && loading ? '加载中…' : '')}
        </div>
      )}
    </div>
  )
}