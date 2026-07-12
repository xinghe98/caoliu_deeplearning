import type { ApiError } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name.replace(/[$()*+.?[\\\]^{|}]/g, '\\$&')}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

export class HttpError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

function formatDetail(payload: ApiError | string | null, fallback: string): string {
  if (!payload) return fallback
  if (typeof payload === 'string') return payload
  if (typeof payload.detail === 'string') return payload.detail
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || JSON.stringify(item)).join('; ')
  }
  return fallback
}

export async function api<T>(
  path: string,
  options: RequestInit & { json?: unknown; formData?: FormData; idempotencyKey?: string } = {},
): Promise<T> {
  const headers = new Headers(options.headers || {})
  const method = (options.method || 'GET').toUpperCase()
  if (options.json !== undefined) {
    headers.set('Content-Type', 'application/json')
  }
  if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') {
    const csrf = getCookie('preference_platform_csrf')
    if (csrf) headers.set('X-CSRF-Token', csrf)
  }
  if (options.idempotencyKey) {
    headers.set('Idempotency-Key', options.idempotencyKey)
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'include',
    body: options.formData ?? (options.json !== undefined ? JSON.stringify(options.json) : options.body),
  })

  if (response.status === 204) {
    return undefined as T
  }

  const contentType = response.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')
  const payload = isJson ? await response.json() : await response.text()

  if (!response.ok) {
    throw new HttpError(response.status, formatDetail(payload as ApiError, `请求失败 (${response.status})`))
  }
  return payload as T
}

export function mediaUrl(contentId: string, mediaId: string): string {
  return `${API_BASE}/api/v1/contents/${contentId}/media/${mediaId}`
}

export function magnetSummary(magnet: string): string {
  if (!magnet) return '无磁力链接'
  const match = magnet.match(/btih:([a-fA-F0-9]{8,40})/i)
  if (match) return `magnet …${match[1].slice(0, 8).toLowerCase()}`
  return magnet.length > 36 ? `${magnet.slice(0, 36)}…` : magnet
}
