import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    bybit_api_key: str
    bybit_api_secret: str
    bybit_testnet: bool = True
    openai_api_key: str
    database_url: str
    redis_url: str
    secret_key: str
    
    take_profit_percentage: float = 2.0
    stop_loss_percentage: float = 1.0
    confidence_threshold: float = 70.0
    max_positions: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()