from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "document_intelligence"

    # Models
    heavy_model: str = "gpt-4o"
    light_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # LLM concurrency
    llm_semaphore: int = 8

    # Token budgets
    max_context_tokens: int = 120_000
    max_segment_tokens: int = 8_000
    max_section_aggregate_tokens: int = 20_000
    max_chapter_aggregate_tokens: int = 30_000
    max_document_aggregate_tokens: int = 40_000
    max_query_context_tokens: int = 20_000
    max_clean_repair_tokens: int = 2_000

    # OCR thresholds
    ocr_dpi: int = 300
    ocr_min_alpha_ratio: float = 0.4
    ocr_min_char_count: int = 50


settings = Settings()
