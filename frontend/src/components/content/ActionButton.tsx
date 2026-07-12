import type { ReactNode } from 'react'

type ActionButtonProps = {
  tone: 'like' | 'dislike' | 'skip'
  label: string
  icon: ReactNode
  onClick: () => void
  disabled?: boolean
}

export function ActionButton({ tone, label, icon, onClick, disabled }: ActionButtonProps) {
  const styles = {
    like: 'bg-like text-panel hover:bg-[oklch(47%_0.19_28)]',
    dislike: 'bg-dislike text-panel hover:bg-ink',
    skip: 'border border-line bg-panel text-ink hover:bg-canvas',
  }[tone]

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex min-h-12 flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {icon}
      {label}
    </button>
  )
}