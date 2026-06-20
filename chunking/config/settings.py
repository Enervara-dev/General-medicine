from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    # Primary extraction model — quality matters for entity recall.
    model_primary: str = "gemini-2.5-flash-lite"
    # Fallback if the primary fails; same family, same dependability.
    model_fallback: str = "gemini-2.5-flash-lite"
    storage_base_path: str = "chunking/output"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
