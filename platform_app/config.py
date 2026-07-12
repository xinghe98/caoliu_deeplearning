from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    platform_data_dir: Path = PROJECT_ROOT / 'platform_data'
    database_url: str | None = None
    allowed_media_roots: str = str(PROJECT_ROOT)
    ingest_api_key: str = ''
    skip_cooldown_days: int = 7
    feed_recommendation_ratio: float = 0.8
    training_label_threshold: int = 200
    cookie_secure: bool = False
    csrf_enabled: bool = True
    candidate_max_upload_mb: int = 512
    candidate_max_files: int = 200
    candidate_max_uncompressed_mb: int = 1024
    login_rate_limit_attempts: int = 10
    login_rate_limit_window_seconds: int = 300
    auto_create_tables: bool = True
    # 无 active 模型时自动启用；手动上传并发布后优先生效
    default_model_path: str = str(PROJECT_ROOT / 'best_model.pth')
    default_model_version: str = 'default-env'

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.platform_data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.platform_data_dir / 'platform.db').as_posix()}"

    @property
    def media_roots(self) -> list[Path]:
        return [Path(part).resolve() for part in self.allowed_media_roots.split(';') if part.strip()]

    @property
    def resolved_default_model_path(self) -> Path | None:
        raw = (self.default_model_path or '').strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
