from typing import Any, List, Tuple, Type

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
from pydantic_settings.sources.providers.env import EnvSettingsSource


class _CommaListEnvSource(EnvSettingsSource):
    COMMA_LIST_FIELDS = {"WATCH_LIST"}

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        if field_name in self.COMMA_LIST_FIELDS and isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _CommaListDotEnvSource(DotEnvSettingsSource):
    COMMA_LIST_FIELDS = {"WATCH_LIST"}

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        if field_name in self.COMMA_LIST_FIELDS and isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./stock_monitor.db"

    ALERT_EMAIL_SENDER: str = ""
    ALERT_EMAIL_PASSWORD: str = ""
    ALERT_EMAIL_RECEIVER: str = ""
    ALERT_EMAIL_SMTP_HOST: str = "smtp.example.com"
    ALERT_EMAIL_SMTP_PORT: int = 465

    ALERT_WEBHOOK_URL: str = ""

    LLM_API_KEY: str = ""
    LLM_API_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    SENTIMENT_MODEL_NAME: str = "yiyanghkust/finbert-tone-chinese"

    PRICE_FETCH_INTERVAL: int = 1
    NEWS_FETCH_INTERVAL: int = 30
    MODEL_RETRAIN_INTERVAL_DAYS: int = 7

    WATCH_LIST: List[str] = ["000001", "600036", "300750"]

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    STREAMLIT_PORT: int = 8501
    API_URL: str = "http://localhost:8000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _CommaListDotEnvSource(settings_cls, env_file=".env", env_file_encoding="utf-8"),
            _CommaListEnvSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()
