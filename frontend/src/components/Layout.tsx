import { NavLink, Outlet } from 'react-router-dom'
import {
  BookmarkCheck,
  Clapperboard,
  Database,
  History,
  Library,
  LogOut,
  Settings,
  Sparkles, Menu, X,
} from 'lucide-react'
import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'

const nav = [
  { to: '/review', label: '待筛选', icon: Clapperboard },
  { to: '/library', label: '内容库', icon: Library },
  { to: '/labels', label: '标签', icon: History },
  { to: '/training', label: '训练', icon: Sparkles },
  { to: '/models', label: '模型', icon: Database },
  { to: '/crawler', label: '任务', icon: BookmarkCheck },
  { to: '/settings', label: '设置', icon: Settings },
]

export function AppLayout() {
  const { user, logout } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="min-h-screen">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-ink focus:px-3 focus:py-2 focus:text-panel">跳到主要内容</a>
      <header className="sticky top-0 z-30 border-b border-line bg-panel/95 backdrop-blur">
        <div className="page-shell flex min-h-16 items-center justify-between gap-3 px-4 lg:px-8">
          <NavLink to="/review" className="shrink-0 leading-none" onClick={() => setMenuOpen(false)}>
            <div className="eyebrow">Preference Archive</div>
            <div className="mt-1 text-base font-semibold tracking-[-0.03em]">偏好筛选</div>
          </NavLink>
          <button type="button" className="quiet-button inline-flex items-center gap-2 md:hidden" aria-expanded={menuOpen} aria-controls="main-nav" onClick={() => setMenuOpen((open) => !open)}>
            {menuOpen ? <X size={17} /> : <Menu size={17} />} 导航
          </button>
          <nav id="main-nav" className={`${menuOpen ? 'flex' : 'hidden'} absolute top-full right-0 left-0 flex-col border-b border-line bg-panel p-3 shadow-sm md:static md:flex md:flex-row md:items-center md:border-0 md:bg-transparent md:p-0 md:shadow-none`}>
          {nav.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  [
                    'inline-flex min-h-11 items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors',
                    isActive ? 'bg-teal-soft font-medium text-teal' : 'text-muted hover:bg-canvas hover:text-ink',
                  ].join(' ')
                }
                onClick={() => setMenuOpen(false)}
              >
                <Icon size={16} />
                {item.label}
              </NavLink>
            )
          })}
          <div className="my-2 border-t border-line md:my-0 md:mx-2 md:h-5 md:border-t-0 md:border-l" />
          <button type="button" onClick={() => void logout()} className="inline-flex min-h-11 items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted hover:bg-canvas hover:text-ink">
            <LogOut size={16} /> <span className="md:hidden">退出 </span>{user?.username}
          </button>
          </nav>
        </div>
      </header>
      <main id="main-content" className="page-shell min-w-0 px-4 py-6 md:px-8 md:py-8">
        <Outlet />
      </main>
    </div>
  )
}
