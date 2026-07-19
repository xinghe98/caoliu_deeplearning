import { api } from './client'
import type {
  CandidateComparison,
  ContentPage,
  ContentRead,
  JobRead,
  JobStats,
  LabelCreate,
  LabelEventRead,
  LabelResultRead,
  ModelRead,
  SessionRead,
  TrainingSnapshotRead,
  TrainingStatusRead,
} from './types'

export const authApi = {
  session: () => api<SessionRead>('/api/v1/auth/session'),
  setup: (username: string, password: string) =>
    api<SessionRead>('/api/v1/auth/setup', { method: 'POST', json: { username, password } }),
  login: (username: string, password: string) =>
    api<SessionRead>('/api/v1/auth/login', { method: 'POST', json: { username, password } }),
  logout: () => api<void>('/api/v1/auth/logout', { method: 'POST' }),
}

export const contentApi = {
  feed: (mode = 'mixed', limit = 20) =>
    api<ContentRead[]>(`/api/v1/feed?mode=${encodeURIComponent(mode)}&limit=${limit}`),
  list: (
    params: {
      label?: number | null
      unlabeled?: boolean
      watched?: boolean
      status?: string
      q?: string
      limit?: number
      cursor?: string | null
    } = {},
  ) => {
    const query = new URLSearchParams()
    if (params.watched) query.set('watched', 'true')
    else if (params.unlabeled) query.set('unlabeled', 'true')
    else if (params.label === 0 || params.label === 1) query.set('label', String(params.label))
    if (params.status) query.set('status', params.status)
    if (params.q?.trim()) query.set('q', params.q.trim())
    if (params.cursor) query.set('cursor', params.cursor)
    query.set('limit', String(params.limit ?? 24))
    return api<ContentPage>(`/api/v1/contents?${query}`)
  },
  get: (id: string) => api<ContentRead>(`/api/v1/contents/${id}`),
  label: (id: string, payload: LabelCreate, idempotencyKey?: string) =>
    api<LabelResultRead>(`/api/v1/contents/${id}/label`, {
      method: 'POST',
      json: payload,
      idempotencyKey,
    }),
  event: (
    id: string,
    event_type: 'view' | 'skip' | 'watched' | 'open_source' | 'copy_magnet' | 'open_magnet',
  ) =>
    api<void>(`/api/v1/contents/${id}/events`, { method: 'POST', json: { event_type } }),
}

export const labelsApi = {
  history: (contentId?: string) =>
    api<LabelEventRead[]>(
      contentId
        ? `/api/v1/labels/history?content_id=${encodeURIComponent(contentId)}&limit=200`
        : '/api/v1/labels/history?limit=200',
    ),
  undo: (eventId: string) => api<ContentRead>(`/api/v1/labels/${eventId}/undo`, { method: 'POST' }),
}

export const trainingApi = {
  status: () => api<TrainingStatusRead>('/api/v1/training/status'),
  snapshots: () => api<TrainingSnapshotRead[]>('/api/v1/training/snapshots'),
  createSnapshot: () => api<TrainingSnapshotRead>('/api/v1/training/snapshots', { method: 'POST' }),
  downloadUrl: (id: string) => `/api/v1/training/snapshots/${id}/download`,
  importCandidate: (file: File, version?: string) => {
    const form = new FormData()
    form.append('archive', file)
    if (version) form.append('version', version)
    return api<ModelRead>('/api/v1/training/candidates/import', { method: 'POST', formData: form })
  },
  comparison: (id: string) => api<CandidateComparison>(`/api/v1/training/candidates/${id}/comparison`),
}

export const modelsApi = {
  list: () => api<ModelRead[]>('/api/v1/models'),
  activate: (id: string, force = false) =>
    api<ModelRead>(`/api/v1/models/${id}/activate?force=${force ? 'true' : 'false'}`, { method: 'POST' }),
  reject: (id: string) => api<ModelRead>(`/api/v1/models/${id}/reject`, { method: 'POST' }),
  rollback: (id: string) => api<ModelRead>(`/api/v1/models/${id}/rollback`, { method: 'POST' }),
}

export const jobsApi = {
  list: () => api<JobRead[]>('/api/v1/jobs?limit=50'),
  stats: () => api<JobStats>('/api/v1/jobs/stats'),
}

export const healthApi = {
  live: () => api<{ status: string }>('/health/live'),
  ready: () => api<{ status: string }>('/health/ready'),
  worker: () => api<{ workers: Array<{ worker_id: string; model_version: string; last_seen_at: string }> }>('/health/worker'),
}
