from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from .config import settings

# Try to get database URL from Railway or environment
database_url = settings.database_url or os.getenv("DATABASE_URL")

if not database_url:
    raise Exception("DATABASE_URL not found. Please set DATABASE_URL in Railway variables.")

engine = create_engine(database_url)
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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)