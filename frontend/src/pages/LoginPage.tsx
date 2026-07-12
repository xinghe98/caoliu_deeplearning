import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { HttpError } from '../api/client'
import { useAuth } from '../hooks/useAuth'

export function LoginPage() {
  const { user, loading, login, setup } = useAuth()
  const [mode, setMode] = useState<'login' | 'setup'>('login')
  const [username, setUsername] = useState('owner')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  if (!loading && user) return <Navigate to="/review" replace />

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      if (mode === 'setup') await setup(username, password)
      else await login(username, password)
    } catch (err) {
      if (err instanceof HttpError && err.status === 409 && mode === 'setup') {
        setError('管理员已初始化，请直接登录')
        setMode('login')
      } else if (err instanceof HttpError) {
        setError(err.detail)
      } else {
        setError('登录失败')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid min-h-screen place-items-center bg-canvas px-4 py-8">
      <form onSubmit={onSubmit} className="w-full max-w-md border border-line bg-panel p-6 shadow-[0_20px_60px_oklch(20%_0.02_270_/_0.08)] sm:p-9">
        <div className="eyebrow">Preference Archive</div>
        <h1 className="page-heading mt-2 text-3xl">{mode === 'setup' ? '创建管理员' : '进入内容库'}</h1>
        <p className="mt-2 text-sm text-muted">局域网个人内容筛选与标注。密码至少 12 位。</p>
        <label className="mt-6 block text-sm">
          <span className="text-muted">用户名</span>
          <input
            className="mt-1 min-h-12 w-full rounded-xl border border-line bg-panel px-3 py-3 focus:border-teal"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
        </label>
        <label className="mt-4 block text-sm">
          <span className="text-muted">密码</span>
          <input
            type="password"
            className="mt-1 min-h-12 w-full rounded-xl border border-line bg-panel px-3 py-3 focus:border-teal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'setup' ? 'new-password' : 'current-password'}
            minLength={mode === 'setup' ? 12 : 1}
            required
          />
        </label>
        {error ? <p className="mt-4 text-sm text-like">{error}</p> : null}
        <button
          type="submit"
          disabled={submitting}
          className="primary-button mt-6 w-full disabled:opacity-60"
        >
          {submitting ? '提交中…' : mode === 'setup' ? '初始化' : '进入'}
        </button>
        <button
          type="button"
          className="mt-3 w-full text-sm text-muted hover:text-teal"
          onClick={() => setMode((current) => (current === 'login' ? 'setup' : 'login'))}
        >
          {mode === 'login' ? '首次使用？创建管理员' : '已有账户？返回登录'}
        </button>
      </form>
    </div>
  )
}
