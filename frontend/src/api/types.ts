export type MediaRead = {
  id: string
  ordinal: number
  mime_type: string
  width: number | null
  height: number | null
  status: string
}

export type ContentRead = {
  id: string
  content_key: string
  title_clean: string
  source_url: string
  magnet_uri: string
  status: string
  current_label: number | null
  created_at: string
  media: MediaRead[]
  probability: number | null
  decision_threshold: number | null
  model_version: string | null
  dataset_role: string
}

export type ContentPage = {
  items: ContentRead[]
  next_cursor: string | null
}

export type LabelResultRead = ContentRead & {
  label_event_id: string
}

export type SessionRead = {
  username: string
}

export type LabelEventRead = {
  id: string
  content_id: string
  user_id: string | null
  label: number
  source: string
  supersedes_event_id: string | null
  model_version: string | null
  probability_at_label: number | null
  created_at: string
}

export type JobRead = {
  id: string
  job_type: string
  status: string
  attempts: number
  last_error: string
  created_at: string
}

export type JobStats = {
  contents_total: number
  predictions_total: number
  scored_contents: number
  predict_pending: number
  predict_running: number
  predict_succeeded: number
  predict_failed: number
  active_model_version: string | null
}

export type ModelRead = {
  id: string
  version: string
  status: string
  checkpoint_path: string
  decision_threshold: number
  temperature: number
  metrics: Record<string, unknown>
  data_manifest_hash: string
  created_at: string
  activated_at: string | null
}

export type TrainingSnapshotRead = {
  id: string
  status: string
  label_cutoff_at: string
  sample_count: number
  positive_count: number
  negative_count: number
  manifest_hash: string
  created_at: string
}

export type TrainingStatusRead = {
  labels_since_last_snapshot: number
  threshold: number
  ready_for_snapshot: boolean
  latest_snapshot: TrainingSnapshotRead | null
}

export type CandidateComparison = {
  candidate: ModelRead
  active: ModelRead | null
  warnings: string[]
  hard_blocks: string[]
}

export type ApiError = {
  detail?: string | Array<{ msg?: string }>
}
