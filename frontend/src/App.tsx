import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/Layout'
import { ToastProvider } from './components/Toast'
import { AuthProvider, RequireAuth } from './hooks/useAuth'
import { CrawlerPage } from './pages/CrawlerPage'
import { LabelsPage } from './pages/LabelsPage'
import { ContentDetailPage } from './pages/ContentDetailPage'
import { LibraryPage } from './pages/LibraryPage'
import { LoginPage } from './pages/LoginPage'
import { ModelsPage } from './pages/ModelsPage'
import { ReviewPage } from './pages/ReviewPage'
import { SettingsPage } from './pages/SettingsPage'
import { TrainingPage } from './pages/TrainingPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 10_000,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <ToastProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                element={
                  <RequireAuth>
                    <AppLayout />
                  </RequireAuth>
                }
              >
                <Route path="/" element={<Navigate to="/review" replace />} />
                <Route path="/review" element={<ReviewPage />} />
                <Route path="/library" element={<LibraryPage />} />
                <Route path="/library/:contentId" element={<ContentDetailPage />} />
                <Route path="/labels" element={<LabelsPage />} />
                <Route path="/training" element={<TrainingPage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/crawler" element={<CrawlerPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Route>
              <Route path="*" element={<Navigate to="/review" replace />} />
            </Routes>
          </ToastProvider>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
