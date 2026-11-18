from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent
# Absolute paths: uvicorn cwd is backend/, but keys may also live in the repo-root .env.
_ENV_FILES = (
    ROOT.parent / ".env",
    ROOT / ".env",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres in docker-compose; SQLite if Docker/Postgres is not installed.
    database_url: str = "sqlite:///./orchestrator.sqlite"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    nvidia_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_api_key: str = ""
    coarena_api_key: str = ""
    cohere_api_key: str = ""
    sambanova_api_key: str = ""
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""
    ollama_base_url: str = "http://localhost:11434"
    allow_mock_fallback: bool = False
    baseline_model: str = "cf-gpt-oss-120b"
    models_config_path: Path = ROOT / "config" / "models.yaml"
    routing_config_path: Path = ROOT / "config" / "routing.yaml"
    eval_fixture_path: Path = ROOT / "fixtures" / "eval_prompts.json"
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174"
    )
    completion_max_tokens: int = 1536
    # When true, merge fixtures/db_seed.json even if request_logs already exist.
    import_local_seed: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
