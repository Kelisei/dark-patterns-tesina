from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_path: str = Field(default="models/dark_pattern_model.joblib", description="Path to the ML model joblib file")
    cache_size: int = Field(default=2000, description="Max size of the LRU cache for predictions")
    active_learning_log_path: str = Field(default="data/active_learning_log.csv", description="Path to save low-confidence predictions")
    active_learning_min_prob: float = Field(default=0.45, description="Min prob to log")
    active_learning_max_prob: float = Field(default=0.55, description="Max prob to log")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
