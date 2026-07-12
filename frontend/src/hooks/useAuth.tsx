import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { authApi } from '../api/endpoints'
import { HttpError } from '../api/client'
import type { SessionRead } from '../api/types'

type AuthState = {
  user: SessionRead | null
  loading: boolean
  needsSetup: boolean
  refresh: () => Promise<void>
  login: (username: string, password: string) => Promise<void>
  setup: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionRead | null>(null)
  const [loading, setLoading] = useState(true)
  const [needsSetup, setNeedsSetup] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const session = await authApi.session()
      setUser(session)
      setNeedsSetup(false)
    } catch (error) {
      setUser(null)
      if (error instanceof HttpError && error.status === 401) {
        // Unknown whether setup is needed until login/setup fails with 409/other.
        setNeedsSetup(false)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const login = useCallback(async (username: string, password: string) => {
    const session = await authApi.login(username, password)
    setUser(session)
    setNeedsSetup(false)
  }, [])

  const setup = useCallback(async (username: string, password: string) => {
    const session = await authApi.setup(username, password)
    setUser(session)
    setNeedsSetup(false)
  }, [])

  const logout = useCallback(async () => {
    try {
      await authApi.logout()
    } finally {
      setUser(null)
    }
  }, [])

  const value = useMemo(
    () => ({ user, loading, needsSetup, refresh, login, setup, logout }),
    [user, loading, needsSetup, refresh, login, setup, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!loading && !user) navigate('/login', { replace: true })
  }, [loading, user, navigate])

  if (loading) {
    return (
      <div className="grid min-h-screen place-items-center text-muted">
        正在检查登录状态…
      </div>
    )
  }
  if (!user) return null
  return children
}
