from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from config import settings

# Try to get database URL from Railway or environment
database_url = (
    os.getenv("DATABASE_URL") or 
    settings.database_url or
    os.getenv("POSTGRES_URL") or
    os.getenv("RAILWAY_POSTGRES_URL")
)

if not database_url:
    # Log available environment variables for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.error("Available environment variables:")
    for key, value in os.environ.items():
        if any(term in key.upper() for term in ['DATABASE', 'POSTGRES', 'DB']):
            logger.error(f"  {key}: {value[:50]}...")
    raise Exception("DATABASE_URL not found. Check Railway PostgreSQL connection.")

engine = create_engine(
    database_url,
    echo=False,
    pool_size=50,
    max_overflow=100,
    pool_timeout=120,  # Increase timeout
    pool_recycle=3600, # Recycle connections every hour
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Coin(Base):
    __tablename__ = "coins"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    coin_symbol = Column(String, index=True)
    model_type = Column(String)  # lstm, random_forest, svm, neural_network
    confidence = Column(Float)
    prediction = Column(String)  # buy, sell, hold
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class HistoricalData(Base):
    __tablename__ = "historical_data"
    
    id = Column(Integer, primary_key=True, index=True)
    coin_symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String, default="1h")  # 1h, 4h, 1d etc
    created_at = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    coin_symbol = Column(String, index=True)
    order_id = Column(String, unique=True)
    side = Column(String)  # buy, sell
    size = Column(Float)
    price = Column(Float)
    leverage = Column(Integer)
    take_profit = Column(Float)
    stop_loss = Column(Float)
    status = Column(String)  # open, closed, cancelled
    pnl = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    ml_confidence = Column(Float)
    strategy_params = Column(JSON)

class TradingStrategy(Base):
    __tablename__ = "trading_strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    coin_symbol = Column(String, index=True)
    take_profit_percentage = Column(Float, default=2.0)
    stop_loss_percentage = Column(Float, default=1.0)
    leverage = Column(Integer, default=10)
    position_size_percentage = Column(Float, default=5.0)
    confidence_threshold = Column(Float, default=70.0)
    is_active = Column(Boolean, default=True)
    updated_by_ai = Column(Boolean, default=False)
    ai_optimization_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class TradingSignal(Base):
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String, unique=True, index=True)  # Unique identifier for the signal
    coin_symbol = Column(String, index=True, nullable=False)
    signal_type = Column(String, nullable=False)  # LONG, SHORT
    confidence = Column(Float, nullable=False)
    models_agreed = Column(Integer, default=0)
    total_models = Column(Integer, default=0)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    position_size_usdt = Column(Float, default=100.0)
    leverage = Column(Integer, default=10)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    status = Column(String, default="pending")  # pending, executing, executed, failed, closed
    trade_id = Column(Integer, nullable=True)  # Foreign key to trades table
    execution_order_id = Column(String, nullable=True)  # Bybit order ID
    unrealized_pnl_usdt = Column(Float, default=0.0)
    unrealized_pnl_percent = Column(Float, default=0.0)
    realized_pnl_usdt = Column(Float, default=0.0)
    realized_pnl_percent = Column(Float, default=0.0)
    execution_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)