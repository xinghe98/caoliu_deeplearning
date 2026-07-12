import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { X } from 'lucide-react'

type Toast = {
  id: string
  message: string
  actionLabel?: string
  onAction?: () => void
}

type ToastState = {
  push: (toast: Omit<Toast, 'id'>, ttlMs?: number) => void
  dismiss: (id: string) => void
}

const ToastContext = createContext<ToastState | null>(null)

function createToastId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])

  const dismiss = useCallback((id: string) => {
    setItems((current) => current.filter((item) => item.id !== id))
  }, [])

  const push = useCallback((toast: Omit<Toast, 'id'>, ttlMs = 5000) => {
    const id = createToastId()
    setItems((current) => [...current, { ...toast, id }])
    if (ttlMs > 0) {
      window.setTimeout(() => dismiss(id), ttlMs)
    }
  }, [dismiss])

  const value = useMemo(() => ({ push, dismiss }), [push, dismiss])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-20 z-50 flex flex-col items-center gap-2 px-4 md:bottom-6" role="status" aria-live="polite">
        {items.map((item) => (
          <div
            key={item.id}
            className="pointer-events-auto flex max-w-md items-center gap-3 rounded-xl border border-line bg-panel px-4 py-3 text-sm shadow-[0_12px_30px_oklch(20%_0.02_270_/_0.12)]"
          >
            <span>{item.message}</span>
            {item.actionLabel && item.onAction ? (
              <button
                type="button"
                className="min-h-9 font-medium text-teal"
                onClick={() => {
                  item.onAction?.()
                  dismiss(item.id)
                }}
              >
                {item.actionLabel}
              </button>
            ) : null}
            <button type="button" aria-label="关闭通知" className="grid min-h-9 min-w-9 place-items-center rounded-lg hover:bg-canvas" onClick={() => dismiss(item.id)}>
              <X size={16} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}
