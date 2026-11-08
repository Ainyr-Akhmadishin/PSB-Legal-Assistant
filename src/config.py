from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GIGACHAT_CREDENTIALS: str

    QDRANT_HOST: str
    QDRANT_PORT: int


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()