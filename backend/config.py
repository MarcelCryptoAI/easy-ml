import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    openai_api_key: str = ""
    database_url: str = ""
    redis_url: str = ""
    secret_key: str = ""
    
    take_profit_percentage: float = 2.0
    stop_loss_percentage: float = 1.0
    confidence_threshold: float = 70.0
    max_positions: int = 10
    
    model_config = {"env_file": ".env"}

# Initialize settings with environment variables
settings = Settings(
    bybit_api_key=os.getenv("BYBIT_API_KEY", ""),
    bybit_api_secret=os.getenv("BYBIT_API_SECRET", ""),
    bybit_testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    database_url=(
        os.getenv("DATABASE_URL") or 
        os.getenv("POSTGRES_URL") or 
        os.getenv("RAILWAY_POSTGRES_URL") or 
        ""
    ),
    redis_url=os.getenv("REDIS_URL", ""),
    secret_key=os.getenv("SECRET_KEY", ""),
)