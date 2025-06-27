from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime
import logging

from database import get_db, TradingStrategy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimize", tags=["optimization"])

@router.get("/status")
async def get_optimization_status():
    """Get current optimization session status"""
    try:
        return {
            "is_running": False,
            "total_coins": 0,
            "completed_coins": 0,
            "current_coin": "",
            "session_start_time": datetime.utcnow().isoformat(),
            "estimated_completion_time": datetime.utcnow().isoformat(),
            "auto_apply_optimizations": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue")
async def get_optimization_queue():
    """Get current optimization queue"""
    try:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply/{symbol}")
async def apply_optimization(symbol: str, optimization_data: Dict, db: Session = Depends(get_db)):
    """Apply optimized parameters to a trading strategy"""
    try:
        strategy = db.query(TradingStrategy).filter(
            TradingStrategy.coin_symbol == symbol,
            TradingStrategy.is_active == True
        ).first()
        
        if not strategy:
            strategy = TradingStrategy(coin_symbol=symbol, is_active=True)
            db.add(strategy)
        
        # Update strategy with optimized parameters
        strategy.take_profit_percentage = optimization_data.get("take_profit_percentage", 2.0)
        strategy.stop_loss_percentage = optimization_data.get("stop_loss_percentage", 1.0)
        strategy.leverage = optimization_data.get("leverage", 10)
        strategy.updated_by_ai = True
        strategy.ai_optimization_reason = optimization_data.get("reason", "AI optimization applied")
        strategy.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {"success": True, "message": f"Optimization applied to {symbol}"}
    except Exception as e:
        logger.error(f"Error applying optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}")
async def get_optimization_history(symbol: str, db: Session = Depends(get_db)):
    """Get optimization history for a symbol"""
    try:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-all")
async def start_batch_optimization(optimization_data: Dict, db: Session = Depends(get_db)):
    """Start batch optimization for all strategies"""
    try:
        return {"success": True, "message": "Batch optimization started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_optimization():
    """Stop current optimization process"""
    try:
        return {"success": True, "message": "Optimization stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))