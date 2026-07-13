import { useCallback, useEffect, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { contentApi, labelsApi } from '../api/endpoints'
import { copyText, HttpError } from '../api/client'
import type { ContentRead } from '../api/types'
import { useToast } from '../components/Toast'

type UseContentLabelingOptions = {
  current: ContentRead | null
  setImageIndex: React.Dispatch<React.SetStateAction<number>>
  onLabeled?: () => void
  onSkipped?: () => void
  onUndo?: () => void
}

// crypto.randomUUID() is unavailable in some browsers on an HTTP LAN origin.
// The key only needs to be unique per mutation so the API can safely dedupe a retry.
function createIdempotencyKey() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `label-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`
}

export function useContentLabeling({
  current,
  setImageIndex,
  onLabeled,
  onSkipped,
  onUndo,
}: UseContentLabelingOptions) {
  const { push } = useToast()
  const [error, setError] = useState('')
  const [lastEventId, setLastEventId] = useState<string | null>(null)

  const labelMutation = useMutation({
    mutationFn: async ({ item, label }: { item: ContentRead; label: 0 | 1 }) => {
      const updated = await contentApi.label(item.id, label, createIdempotencyKey())
      return { updated, eventId: updated.label_event_id, label }
    },
    onSuccess: ({ eventId, label }) => {
      setLastEventId(eventId)
      setError('')
      onLabeled?.()
      push({
        message: label === 1 ? '已标记喜欢' : '已标记不喜欢',
        actionLabel: '撤销',
        onAction: () => {
          if (!eventId) return
          void labelsApi.undo(eventId).then(() => {
            onUndo?.()
            push({ message: '已撤销标签' })
          })
        },
      })
    },
    onError: (err) => {
      setError(err instanceof HttpError ? err.detail : '标注失败')
    },
  })

  const skipMutation = useMutation({
    mutationFn: async (item: ContentRead) => {
      await contentApi.event(item.id, 'skip')
      return item.id
    },
    onSuccess: () => {
      setError('')
      onSkipped?.()
      push({ message: '已跳过（7 天冷却，不进入训练）' })
    },
    onError: (err) => {
      setError(err instanceof HttpError ? err.detail : '跳过失败')
    },
  })

  const busy = labelMutation.isPending || skipMutation.isPending

  const copyMagnet = useCallback(async (item: ContentRead) => {
    if (!item.magnet_uri) {
      push({ message: '当前内容没有磁力链接' })
      return
    }
    try {
      await copyText(item.magnet_uri)
      push({ message: '磁力链接已复制' })
      void contentApi.event(item.id, 'copy_magnet').catch(() => undefined)
    } catch {
      push({ message: '复制失败，请手动选择链接' })
    }
  }, [push])

  const openMagnet = useCallback(async (item: ContentRead) => {
    if (!item.magnet_uri) {
      push({ message: '当前内容没有磁力链接' })
      return
    }
    try {
      await contentApi.event(item.id, 'open_magnet')
      window.location.href = item.magnet_uri
    } catch {
      push({ message: '请确认已安装并关联磁力下载器' })
    }
  }, [push])

  const like = useCallback(() => {
    if (!current || busy) return
    labelMutation.mutate({ item: current, label: 1 })
  }, [busy, current, labelMutation])

  const dislike = useCallback(() => {
    if (!current || busy) return
    labelMutation.mutate({ item: current, label: 0 })
  }, [busy, current, labelMutation])

  const skip = useCallback(() => {
    if (!current || busy) return
    skipMutation.mutate(current)
  }, [busy, current, skipMutation])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return
      }
      if (!current || busy) return
      if (event.key === '1') {
        event.preventDefault()
        labelMutation.mutate({ item: current, label: 1 })
      } else if (event.key === '2') {
        event.preventDefault()
        labelMutation.mutate({ item: current, label: 0 })
      } else if (event.key === '3') {
        event.preventDefault()
        skipMutation.mutate(current)
      } else if (event.key.toLowerCase() === 'm') {
        event.preventDefault()
        void copyMagnet(current)
      } else if (event.key.toLowerCase() === 'z' && lastEventId) {
        event.preventDefault()
        void labelsApi.undo(lastEventId).then(() => {
          onUndo?.()
          push({ message: '已撤销标签' })
        })
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        setImageIndex((value) => Math.max(0, value - 1))
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        setImageIndex((value) => Math.min((current.media.length || 1) - 1, value + 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, copyMagnet, current, labelMutation, lastEventId, onUndo, push, setImageIndex, skipMutation])

  return {
    busy,
    error,
    like,
    dislike,
    skip,
    copyMagnet,
    openMagnet,
  }
}
