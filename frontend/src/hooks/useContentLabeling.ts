import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { contentApi, labelsApi } from '../api/endpoints'
import { copyText, HttpError } from '../api/client'
import type { ContentRead } from '../api/types'
import { useToast } from '../components/Toast'

type ItemActionCallback = (contentId: ContentRead['id']) => void | Promise<void>
type WatchedToggleCallback = (updated: ContentRead) => void | Promise<void>

type UseContentLabelingOptions = {
  current: ContentRead | null
  setImageIndex: React.Dispatch<React.SetStateAction<number>>
  onLabeled?: ItemActionCallback
  onSkipped?: ItemActionCallback
  onWatchedToggle?: WatchedToggleCallback
  onUndo?: () => void | Promise<void>
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
  onWatchedToggle,
  onUndo,
}: UseContentLabelingOptions) {
  const { push } = useToast()
  const [error, setError] = useState('')
  const [lastEventId, setLastEventId] = useState<string | null>(null)
  const actionLockRef = useRef(false)
  const undoLockRef = useRef(false)

  const labelMutation = useMutation({
    mutationFn: async ({ item, label }: { item: ContentRead; label: 0 | 1 }) => {
      const updated = await contentApi.label(item.id, {
        label,
        model_version: item.model_version,
        probability_at_label: item.probability,
      }, createIdempotencyKey())
      return { updated, eventId: updated.label_event_id, label }
    },
    onSuccess: async ({ updated, eventId, label }) => {
      setLastEventId(eventId)
      setError('')
      try {
        await onLabeled?.(updated.id)
      } catch {
        // Server already committed; keep UI unlocked and avoid masking success.
      }
      push({
        message: label === 1 ? '已标记喜欢' : '已标记不喜欢',
        actionLabel: '撤销',
        onAction: () => {
          if (!eventId || undoLockRef.current) return
          undoLockRef.current = true
          void labelsApi.undo(eventId).then(() => {
            setLastEventId(null)
            return onUndo?.()
          }).then(() => {
            push({ message: '已撤销标签' })
          }).catch((err) => {
            push({ message: err instanceof HttpError ? err.detail : '撤销失败' })
          }).finally(() => {
            undoLockRef.current = false
          })
        },
      })
    },
    onError: (err) => {
      setError(err instanceof HttpError ? err.detail : '标注失败')
    },
    onSettled: () => {
      actionLockRef.current = false
    },
  })

  const skipMutation = useMutation({
    mutationFn: async (item: ContentRead) => {
      await contentApi.event(item.id, 'skip')
      return item.id
    },
    onSuccess: async (contentId) => {
      setError('')
      try {
        await onSkipped?.(contentId)
      } catch {
        // Keep unlock path in onSettled even if cache updates fail.
      }
      push({ message: '已跳过（7 天冷却，不进入训练）' })
    },
    onError: (err) => {
      setError(err instanceof HttpError ? err.detail : '跳过失败')
    },
    onSettled: () => {
      actionLockRef.current = false
    },
  })

  const watchedMutation = useMutation({
    mutationFn: async (item: ContentRead) => {
      const next = !item.is_watched
      const updated = await contentApi.setWatched(item.id, next)
      return updated
    },
    onSuccess: async (updated) => {
      setError('')
      try {
        await onWatchedToggle?.(updated)
      } catch {
        // Keep unlock path in onSettled even if cache updates fail.
      }
      push({
        message: updated.is_watched
          ? '已标记看过（可再次点击改回未看过；喜欢/不喜欢标签会保留）'
          : '已改回未看过（若有喜欢/不喜欢，会回到对应列表）',
      })
    },
    onError: (err) => {
      setError(err instanceof HttpError ? err.detail : '切换已看过状态失败')
    },
    onSettled: () => {
      actionLockRef.current = false
    },
  })

  const busy = labelMutation.isPending || skipMutation.isPending || watchedMutation.isPending

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

  const submitLabel = useCallback((label: 0 | 1) => {
    if (!current || actionLockRef.current) return
    actionLockRef.current = true
    labelMutation.mutate({ item: current, label })
  }, [current, labelMutation])

  const submitSkip = useCallback(() => {
    if (!current || actionLockRef.current) return
    actionLockRef.current = true
    skipMutation.mutate(current)
  }, [current, skipMutation])

  const submitWatchedToggle = useCallback(() => {
    if (!current || actionLockRef.current) return
    actionLockRef.current = true
    watchedMutation.mutate(current)
  }, [current, watchedMutation])

  const like = useCallback(() => submitLabel(1), [submitLabel])
  const dislike = useCallback(() => submitLabel(0), [submitLabel])
  const skip = useCallback(() => submitSkip(), [submitSkip])
  const toggleWatched = useCallback(() => submitWatchedToggle(), [submitWatchedToggle])

  const undoLast = useCallback(() => {
    if (!lastEventId || undoLockRef.current) return
    undoLockRef.current = true
    const eventId = lastEventId
    setLastEventId(null)
    void labelsApi.undo(eventId).then(() => {
      return onUndo?.()
    }).then(() => {
      push({ message: '已撤销标签' })
    }).catch((err) => {
      setLastEventId(eventId)
      push({ message: err instanceof HttpError ? err.detail : '撤销失败' })
    }).finally(() => {
      undoLockRef.current = false
    })
  }, [lastEventId, onUndo, push])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return
      }
      if (event.repeat) return
      if (event.key === '1') {
        event.preventDefault()
        submitLabel(1)
      } else if (event.key === '2') {
        event.preventDefault()
        submitLabel(0)
      } else if (event.key === '3') {
        event.preventDefault()
        submitSkip()
      } else if (event.key === '4') {
        event.preventDefault()
        submitWatchedToggle()
      } else if (event.key.toLowerCase() === 'm') {
        if (!current) return
        event.preventDefault()
        void copyMagnet(current)
      } else if (event.key.toLowerCase() === 'z') {
        event.preventDefault()
        undoLast()
      } else if (event.key === 'ArrowLeft') {
        if (!current) return
        event.preventDefault()
        setImageIndex((value) => Math.max(0, value - 1))
      } else if (event.key === 'ArrowRight') {
        if (!current) return
        event.preventDefault()
        setImageIndex((value) => Math.min((current.media.length || 1) - 1, value + 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [copyMagnet, current, setImageIndex, submitLabel, submitSkip, submitWatchedToggle, undoLast])

  return {
    busy,
    error,
    like,
    dislike,
    skip,
    toggleWatched,
    copyMagnet,
    openMagnet,
  }
}
