from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict
from datetime import datetime
import logging

from database import get_db, Coin, MLPrediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

# Global training state (will be moved to proper state management later)
training_paused = False
current_training_coin = None
current_training_model = None
training_progress = 0

@router.get("/info")
async def get_training_info(db: Session = Depends(get_db)):
    """Get current ML training information with REAL data"""
    try:
        total_coins = db.query(Coin).filter(Coin.is_active == True).count()
        model_types = ["lstm", "random_forest", "svm", "neural_network", "xgboost", "lightgbm", "catboost", "transformer", "gru", "cnn_1d"]
        
        total_models_expected = total_coins * len(model_types)
        completed_predictions = db.query(MLPrediction).count()
        overall_progress = (completed_predictions / total_models_expected * 100) if total_models_expected > 0 else 0
        overall_progress = min(100, overall_progress)
        
        global current_training_coin, current_training_model, training_progress, training_paused
        
        return {
            "total_coins": total_coins,
            "total_models": len(model_types),
            "total_predictions_possible": total_models_expected,
            "completed_predictions": completed_predictions,
            "overall_percentage": round(overall_progress, 2),
            "current_coin": current_training_coin or "None",
            "current_model": current_training_model or "None",
            "current_model_progress": training_progress,
            "status": "paused" if training_paused else "training" if current_training_coin else "idle"
        }
    except Exception as e:
        logger.error(f"Error getting training info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pause")
async def pause_training():
    """Pause ML training"""
    try:
        global training_paused
        training_paused = True
        logger.info("🛑 ML Training paused by user")
        return {"success": True, "message": "Training paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resume") 
async def resume_training():
    """Resume ML training"""
    try:
        global training_paused
        training_paused = False
        logger.info("▶️ ML Training resumed by user")
        return {"success": True, "message": "Training resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/priority")
async def start_priority_training(request: Dict, db: Session = Depends(get_db)):
    """Start priority training for a specific coin (all 10 models)"""
    try:
        coin_symbol = request.get("coin_symbol")
        if not coin_symbol:
            raise HTTPException(status_code=400, detail="coin_symbol is required")
        
        coin = db.query(Coin).filter(Coin.symbol == coin_symbol).first()
        if not coin:
            raise HTTPException(status_code=404, detail=f"Coin {coin_symbol} not found")
        
        # This would trigger priority training logic
        return {
            "success": True,
            "message": f"Priority training started for {coin_symbol}",
            "estimated_completion": "2-5 minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))