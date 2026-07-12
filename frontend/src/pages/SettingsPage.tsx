import { useQuery } from '@tanstack/react-query'
import { healthApi } from '../api/endpoints'

export function SettingsPage() {
  const live = useQuery({ queryKey: ['health-live'], queryFn: healthApi.live })
  const ready = useQuery({ queryKey: ['health-ready'], queryFn: healthApi.ready })

  return (
    <div className="space-y-8">
      <div className="border-b border-line pb-5">
        <div className="eyebrow">系统维护</div>
        <h1 className="page-heading mt-1 text-3xl">设置与健康</h1>
        <p className="mt-1 text-sm text-muted">第一版设置以运维说明为主；媒体根与密钥通过后端环境变量配置。</p>
      </div>

      <section className="grid divide-y divide-line border-y border-line sm:grid-cols-2 sm:divide-x sm:divide-y-0">
        <Card title="Live" value={live.data?.status ?? (live.isError ? 'error' : '…')} />
        <Card title="Ready" value={ready.data?.status ?? (ready.isError ? 'error' : '…')} />
      </section>

      <section className="border-y border-line py-6 text-sm leading-7 text-muted">
        <h2 className="text-base font-semibold text-ink">环境变量</h2>
        <ul className="mt-3 list-disc space-y-1 pl-5">
          <li><code>ALLOWED_MEDIA_ROOTS</code>：允许的图片根目录（分号分隔）</li>
          <li><code>INGEST_API_KEY</code>：爬虫入库密钥</li>
          <li><code>TRAINING_LABEL_THRESHOLD</code>：自动训练包阈值（默认 200）</li>
          <li><code>PLATFORM_DATA_DIR</code>：数据库/训练包目录</li>
        </ul>
        <h2 className="mt-6 text-base font-semibold text-ink">本地启动</h2>
        <pre className="mt-3 overflow-x-auto border border-line bg-canvas p-4 text-xs text-ink">
{`# 后端
python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080
python -m platform_app.worker

# 前端开发
cd frontend
npm run dev

# 生产构建（由 FastAPI 托管 dist）
npm run build`}
        </pre>
        <p className="mt-4">媒体文件原地引用，平台备份不等于原图备份。请确保爬虫/数据集目录持续可访问。</p>
      </section>
    </div>
  )
}

function Card({ title, value }: { title: string; value: string }) {
  return (
    <div className="px-4 py-5">
      <div className="text-xs text-muted">{title}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
    </div>
  )
}
