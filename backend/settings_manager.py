from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List
from pydantic import BaseModel

from .database import get_db, TradingStrategy
from .config import settings

router = APIRouter(prefix="/settings", tags=["settings"])

class GlobalSettings(BaseModel):
    max_positions: int
    default_take_profit: float
    default_stop_loss: float
    default_leverage: int
    default_confidence_threshold: float
    trading_enabled: bool

class StrategyUpdate(BaseModel):
    take_profit_percentage: float
    stop_loss_percentage: float
    leverage: int
    position_size_percentage: float
    confidence_threshold: float

@router.get("/global")
async def get_global_settings():
    """Get global platform settings"""
    return {
        "max_positions": settings.max_positions,
        "default_take_profit": settings.take_profit_percentage,
        "default_stop_loss": settings.stop_loss_percentage,
        "default_leverage": 10,
        "default_confidence_threshold": settings.confidence_threshold,
        "bybit_testnet": settings.bybit_testnet,
        "api_configured": bool(settings.bybit_api_key and settings.bybit_api_secret),
        "openai_configured": bool(settings.openai_api_key)
    }

@router.get("/strategies")
async def get_all_strategies(db: Session = Depends(get_db)):
    """Get all coin strategies"""
    strategies = db.query(TradingStrategy).filter(
        TradingStrategy.is_active == True
    ).all()
    
    return [{
        "coin_symbol": strategy.coin_symbol,
        "take_profit_percentage": strategy.take_profit_percentage,
        "stop_loss_percentage": strategy.stop_loss_percentage,
        "leverage": strategy.leverage,
        "position_size_percentage": strategy.position_size_percentage,
        "confidence_threshold": strategy.confidence_threshold,
        "updated_by_ai": strategy.updated_by_ai,
        "ai_optimization_reason": strategy.ai_optimization_reason
    } for strategy in strategies]

@router.put("/strategy/{symbol}")
async def update_coin_strategy(
    symbol: str, 
    strategy_data: StrategyUpdate, 
    db: Session = Depends(get_db)
):
    """Update strategy for specific coin"""
    strategy = db.query(TradingStrategy).filter(
        TradingStrategy.coin_symbol == symbol,
        TradingStrategy.is_active == True
    ).first()
    
    if not strategy:
        # Create new strategy
        strategy = TradingStrategy(coin_symbol=symbol)
        db.add(strategy)
    
    # Validate parameters
    if not (0.5 <= strategy_data.take_profit_percentage <= 10):
        raise HTTPException(status_code=400, detail="Take profit must be between 0.5% and 10%")
    
    if not (0.2 <= strategy_data.stop_loss_percentage <= 5):
        raise HTTPException(status_code=400, detail="Stop loss must be between 0.2% and 5%")
    
    if not (1 <= strategy_data.leverage <= 20):
        raise HTTPException(status_code=400, detail="Leverage must be between 1x and 20x")
    
    if not (60 <= strategy_data.confidence_threshold <= 95):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 60% and 95%")
    
    # Update strategy
    strategy.take_profit_percentage = strategy_data.take_profit_percentage
    strategy.stop_loss_percentage = strategy_data.stop_loss_percentage
    strategy.leverage = strategy_data.leverage
    strategy.position_size_percentage = strategy_data.position_size_percentage
    strategy.confidence_threshold = strategy_data.confidence_threshold
    strategy.updated_by_ai = False  # Mark as manually updated
    
    db.commit()
    
    return {"success": True, "message": f"Strategy updated for {symbol}"}

@router.post("/reset-strategies")
async def reset_all_strategies(db: Session = Depends(get_db)):
    """Reset all strategies to default values"""
    strategies = db.query(TradingStrategy).filter(
        TradingStrategy.is_active == True
    ).all()
    
    for strategy in strategies:
        strategy.take_profit_percentage = settings.take_profit_percentage
        strategy.stop_loss_percentage = settings.stop_loss_percentage
        strategy.leverage = 10
        strategy.position_size_percentage = 5.0
        strategy.confidence_threshold = settings.confidence_threshold
        strategy.updated_by_ai = False
        strategy.ai_optimization_reason = None
    
    db.commit()
    
    return {
        "success": True, 
        "message": f"Reset {len(strategies)} strategies to default values"
    }