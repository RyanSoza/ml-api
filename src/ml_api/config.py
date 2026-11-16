"""Settings from env"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_path: str = "models/model.joblib"
    version: str = "dev"
    log_level: str = "INFO"
