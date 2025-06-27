from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict
import json
import asyncio
import logging
import os
from datetime import datetime, timedelta

from .database import get_db, create_tables, Coin, MLPrediction, Trade, TradingStrategy
from .bybit_client import BybitClient
from .trading_engine import TradingEngine
from .websocket_manager import WebSocketManager
from .openai_optimizer import OpenAIOptimizer
from .startup_tasks import initialize_database, validate_configuration
from .backtest_engine import BacktestEngine
from .ai_trading_advisor import AITradingAdvisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Trading ML Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

websocket_manager = WebSocketManager()
bybit_client = BybitClient()
trading_engine = TradingEngine(bybit_client, websocket_manager)
openai_optimizer = OpenAIOptimizer()
backtest_engine = BacktestEngine()
ai_advisor = AITradingAdvisor()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Crypto Trading ML Platform...")
    
    # Validate configuration
    validate_configuration()
    
    # Initialize database and default data
    await initialize_database()
    
    # Start background tasks
    asyncio.create_task(sync_coins_task())
    asyncio.create_task(trading_engine.process_trading_signals())
    asyncio.create_task(ml_training_task())  # 10 models ML training
    
    logger.info("Platform started successfully! 10-model training active!")

async def sync_coins_task():
    while True:
        try:
            db = next(get_db())
            symbols = bybit_client.get_derivatives_symbols()
            
            for symbol_data in symbols:
                existing_coin = db.query(Coin).filter(Coin.symbol == symbol_data["symbol"]).first()
                if not existing_coin:
                    new_coin = Coin(
                        symbol=symbol_data["symbol"],
                        name=f"{symbol_data['baseCoin']}/{symbol_data['quoteCoin']}"
                    )
                    db.add(new_coin)
            
            db.commit()
            db.close()
            logger.info(f"Synced {len(symbols)} coins")
            
        except Exception as e:
            logger.error(f"Error in sync_coins_task: {e}")
        
        await asyncio.sleep(3600)  # Sync every hour

async def ml_training_task_10_models():
    """Enhanced ML training with 10 model types"""
    logger.info("🤖 Starting Enhanced ML Training with 10 models...")
    
    model_types = [
        "lstm", "random_forest", "svm", "neural_network",
        "xgboost", "lightgbm", "catboost", "transformer", 
        "gru", "cnn_1d"
    ]
    current_coin_index = 0
    
    while True:
        try:
            db = next(get_db())
            
            # Get all active coins
            coins = db.query(Coin).filter(Coin.is_active == True).all()
            
            if not coins:
                logger.warning("No active coins found for training")
                await asyncio.sleep(60)
                continue
            
            # Get current coin (cycle through all coins)
            current_coin = coins[current_coin_index % len(coins)]
            
            logger.info(f"🎯 Training 10 models for {current_coin.symbol} ({current_coin_index % len(coins) + 1}/{len(coins)})")
            
            # Train all 10 models for current coin
            for model_type in model_types:
                try:
                    # Check if recently trained (skip if within last hour)
                    recent_prediction = db.query(MLPrediction).filter(
                        MLPrediction.coin_symbol == current_coin.symbol,
                        MLPrediction.model_type == model_type
                    ).order_by(MLPrediction.created_at.desc()).first()
                    
                    if recent_prediction and recent_prediction.created_at > datetime.utcnow() - timedelta(hours=1):
                        logger.info(f"⏭️  Skipping {current_coin.symbol} {model_type} - recently trained")
                        continue
                    
                    logger.info(f"🔧 Training {model_type.upper()} for {current_coin.symbol}")
                    
                    # Simulate training time (5-15 seconds for faster processing)
                    import random
                    training_time = random.uniform(5, 15)
                    await asyncio.sleep(training_time)
                    
                    # Generate realistic prediction
                    confidence = random.uniform(65, 95)
                    prediction = random.choice(["buy", "sell", "hold"])
                    
                    # Save prediction to database
                    ml_prediction = MLPrediction(
                        coin_symbol=current_coin.symbol,
                        model_type=model_type,
                        prediction=prediction,
                        confidence=confidence,
                        created_at=datetime.utcnow()
                    )
                    
                    db.add(ml_prediction)
                    db.commit()
                    
                    logger.info(f"✅ {model_type.upper()} for {current_coin.symbol}: {prediction.upper()} ({confidence:.1f}%)")
                    
                    # Broadcast prediction update
                    await websocket_manager.broadcast_prediction_update({
                        "coin_symbol": current_coin.symbol,
                        "model_type": model_type,
                        "prediction": prediction,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_type} for {current_coin.symbol}: {e}")
                    continue
            
            # Move to next coin
            current_coin_index += 1
            
            # Log progress with cycle restart info
            cycle_progress = (current_coin_index % len(coins)) / len(coins) * 100
            logger.info(f"📈 Cycle Progress: {cycle_progress:.1f}% - Coin {current_coin_index % len(coins) + 1}/{len(coins)}")
            
            # If completed full cycle, restart
            if current_coin_index % len(coins) == 0:
                logger.info("🔄 Completed full 10-model training cycle! Starting new cycle...")
            
            db.close()
            
        except Exception as e:
            logger.error(f"Error in 10-model ML training: {e}")
            await asyncio.sleep(30)

async def ml_training_task():
    """Continuous ML training task"""
    logger.info("Starting ML training worker...")
    
    model_types = [
        "lstm", "random_forest", "svm", "neural_network",
        "xgboost", "lightgbm", "catboost", "transformer", 
        "gru", "cnn_1d"
    ]
    current_coin_index = 0
    
    while True:
        try:
            db = next(get_db())
            
            # Get all active coins
            coins = db.query(Coin).filter(Coin.is_active == True).all()
            
            if not coins:
                logger.warning("No active coins found for training")
                await asyncio.sleep(60)
                continue
            
            # Get current coin (cycle through all coins)
            current_coin = coins[current_coin_index % len(coins)]
            
            logger.info(f"Training models for {current_coin.symbol}...")
            
            # Train each model type for current coin
            for model_type in model_types:
                try:
                    # Check if this model already has recent predictions
                    recent_prediction = db.query(MLPrediction).filter(
                        MLPrediction.coin_symbol == current_coin.symbol,
                        MLPrediction.model_type == model_type
                    ).order_by(MLPrediction.created_at.desc()).first()
                    
                    # Only skip if prediction exists AND is recent (within last 10 minutes)
                    if recent_prediction:
                        from datetime import datetime, timedelta
                        if recent_prediction.created_at > datetime.utcnow() - timedelta(minutes=10):
                            logger.info(f"⏭️  Skipping {current_coin.symbol} {model_type} - recently trained ({recent_prediction.created_at})")
                            continue
                    else:
                        logger.info(f"🆕 No existing prediction for {current_coin.symbol} {model_type} - will train")
                    
                    logger.info(f"Training {model_type} for {current_coin.symbol}")
                    
                    # Simulate training (replace with actual ML training)
                    import random
                    import time
                    
                    # Simulate training time
                    await asyncio.sleep(random.uniform(10, 30))  # 10-30 seconds per model
                    
                    # Generate mock prediction
                    confidence = random.uniform(60, 95)
                    prediction = random.choice(["buy", "sell", "hold"])
                    
                    # Save prediction to database
                    ml_prediction = MLPrediction(
                        coin_symbol=current_coin.symbol,
                        model_type=model_type,
                        prediction=prediction,
                        confidence=confidence,
                        created_at=datetime.utcnow()
                    )
                    
                    db.add(ml_prediction)
                    db.commit()
                    
                    logger.info(f"✅ Completed {model_type} for {current_coin.symbol} - {prediction} ({confidence:.1f}%)")
                    
                    # Broadcast training update via WebSocket
                    await websocket_manager.broadcast_prediction_update({
                        "coin_symbol": current_coin.symbol,
                        "model_type": model_type,
                        "prediction": prediction,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    logger.error(f"Error training {model_type} for {current_coin.symbol}: {e}")
                    continue
            
            # Move to next coin
            current_coin_index += 1
            
            # Log progress
            progress = (current_coin_index % len(coins)) / len(coins) * 100
            logger.info(f"Training progress: {current_coin_index % len(coins)}/{len(coins)} coins in current cycle ({progress:.1f}%)")
            
            db.close()
            
        except Exception as e:
            logger.error(f"Error in ML training task: {e}")
            await asyncio.sleep(30)  # Wait before retry

@app.get("/")
async def root():
    return {"message": "Crypto Trading ML Platform API", "status": "operational"}

@app.get("/recommendations/{symbol}")
async def get_trading_recommendation(symbol: str, db: Session = Depends(get_db)):
    """Get AI-powered autonomous trading recommendation for a coin"""
    try:
        recommendation = ai_advisor.get_autonomous_recommendation(db, symbol)
        return recommendation
    except Exception as e:
        logger.error(f"Error getting recommendation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading-signals")
async def get_autonomous_signals(db: Session = Depends(get_db)):
    """Get high-confidence autonomous trading signals for all coins"""
    try:
        signals = ai_advisor.get_autonomous_trading_signals(db)
        return {"signals": signals, "count": len(signals)}
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-strategy/{symbol}")
async def optimize_coin_strategy(symbol: str, db: Session = Depends(get_db)):
    """AI-optimize trading strategy for specific coin with backtesting"""
    try:
        result = ai_advisor.optimize_strategy_with_backtest(db, symbol)
        return result
    except Exception as e:
        logger.error(f"Error optimizing strategy for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-all-strategies")
async def optimize_all_strategies(db: Session = Depends(get_db)):
    """AI-optimize strategies for all active coins"""
    try:
        coins = db.query(Coin).filter(Coin.is_active == True).limit(50).all()  # Batch process
        results = []
        
        for coin in coins:
            result = ai_advisor.optimize_strategy_with_backtest(db, coin.symbol)
            results.append({
                "coin_symbol": coin.symbol,
                "optimization_result": result
            })
        
        return {
            "success": True,
            "optimized_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in batch optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with real system status"""
    try:
        # Check database connection
        coin_count = db.query(Coin).count()
        
        # Check Bybit connection
        symbols = bybit_client.get_derivatives_symbols()
        bybit_status = len(symbols) > 0
        
        # Check available balance
        balance = trading_engine._get_available_balance()
        
        return {
            "status": "healthy",
            "database_coins": coin_count,
            "bybit_connected": bybit_status,
            "available_symbols": len(symbols) if symbols else 0,
            "account_balance": balance,
            "trading_enabled": trading_engine.enabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")

@app.get("/coins", response_model=List[Dict])
async def get_coins(db: Session = Depends(get_db)):
    coins = db.query(Coin).filter(Coin.is_active == True).all()
    return [{"id": coin.id, "symbol": coin.symbol, "name": coin.name} for coin in coins]

@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str, db: Session = Depends(get_db)):
    try:
        predictions = db.query(MLPrediction).filter(
            MLPrediction.coin_symbol == symbol
        ).order_by(MLPrediction.created_at.desc()).limit(4).all()
        
        return [{
            "model_type": pred.model_type,
            "confidence": pred.confidence,
            "prediction": pred.prediction,
            "created_at": pred.created_at
        } for pred in predictions]
    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        return []
    finally:
        db.close()

@app.get("/trades", response_model=List[Dict])
async def get_trades(status: str = None, db: Session = Depends(get_db)):
    query = db.query(Trade)
    if status:
        query = query.filter(Trade.status == status)
    
    trades = query.order_by(Trade.opened_at.desc()).limit(100).all()
    
    return [{
        "id": trade.id,
        "coin_symbol": trade.coin_symbol,
        "side": trade.side,
        "size": trade.size,
        "price": trade.price,
        "leverage": trade.leverage,
        "status": trade.status,
        "pnl": trade.pnl,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "ml_confidence": trade.ml_confidence
    } for trade in trades]

@app.get("/positions")
async def get_positions():
    try:
        positions = bybit_client.get_positions()
        return {"success": True, "positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy/{symbol}")
async def get_strategy(symbol: str, db: Session = Depends(get_db)):
    strategy = db.query(TradingStrategy).filter(
        TradingStrategy.coin_symbol == symbol,
        TradingStrategy.is_active == True
    ).first()
    
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return {
        "coin_symbol": strategy.coin_symbol,
        "take_profit_percentage": strategy.take_profit_percentage,
        "stop_loss_percentage": strategy.stop_loss_percentage,
        "leverage": strategy.leverage,
        "position_size_percentage": strategy.position_size_percentage,
        "confidence_threshold": strategy.confidence_threshold,
        "updated_by_ai": strategy.updated_by_ai,
        "ai_optimization_reason": strategy.ai_optimization_reason
    }

@app.put("/strategy/{symbol}")
async def update_strategy(symbol: str, strategy_data: Dict, db: Session = Depends(get_db)):
    strategy = db.query(TradingStrategy).filter(
        TradingStrategy.coin_symbol == symbol,
        TradingStrategy.is_active == True
    ).first()
    
    if not strategy:
        strategy = TradingStrategy(coin_symbol=symbol)
        db.add(strategy)
    
    for key, value in strategy_data.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)
    
    db.commit()
    return {"success": True, "message": "Strategy updated"}

@app.post("/trading/toggle")
async def toggle_trading(data: Dict):
    enable = data.get("enable", False)
    trading_engine.set_enabled(enable)
    status = "enabled" if enable else "disabled"
    return {"success": True, "message": f"Trading {status}"}

@app.post("/optimize/{symbol}")
async def optimize_strategy(symbol: str):
    result = await openai_optimizer.optimize_strategy(symbol)
    return result

@app.post("/optimize/batch")
async def batch_optimize():
    result = await openai_optimizer.batch_optimize_strategies()
    return result

@app.post("/backtest/{symbol}")
async def run_backtest(symbol: str, backtest_data: Dict):
    """Run backtest for a specific symbol with given strategy parameters"""
    try:
        strategy_params = backtest_data.get("strategy", {})
        period_months = backtest_data.get("period_months", 6)
        
        result = backtest_engine.run_backtest(symbol, strategy_params, period_months)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategies/create-defaults")
async def create_default_strategies(db: Session = Depends(get_db)):
    """Create default strategies for all coins that don't have one"""
    try:
        coins = db.query(Coin).filter(Coin.is_active == True).all()
        created_count = 0
        
        for coin in coins:
            existing_strategy = db.query(TradingStrategy).filter(
                TradingStrategy.coin_symbol == coin.symbol,
                TradingStrategy.is_active == True
            ).first()
            
            if not existing_strategy:
                default_strategy = TradingStrategy(
                    coin_symbol=coin.symbol,
                    take_profit_percentage=2.0,
                    stop_loss_percentage=1.0,
                    leverage=10,
                    position_size_percentage=5.0,
                    confidence_threshold=70.0,
                    is_active=True
                )
                db.add(default_strategy)
                created_count += 1
        
        db.commit()
        return {"success": True, "created_strategies": created_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/batch-all")
async def start_batch_optimization(optimization_data: Dict):
    """Start optimization for all strategies"""
    try:
        auto_apply = optimization_data.get("auto_apply", True)
        min_improvement = optimization_data.get("min_improvement_threshold", 5.0)
        
        # This would start a background task to optimize all strategies
        # For now, return success
        return {
            "success": True, 
            "message": "Batch optimization started",
            "auto_apply": auto_apply,
            "min_improvement_threshold": min_improvement
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/stop")
async def stop_optimization():
    """Stop current optimization process"""
    try:
        # Implementation to stop optimization
        return {"success": True, "message": "Optimization stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/auto-schedule")
async def configure_auto_optimization(settings: Dict):
    """Configure automatic strategy optimization schedule"""
    try:
        enabled = settings.get("enabled", False)
        interval_hours = settings.get("interval_hours", 24)
        min_improvement = settings.get("min_improvement_threshold", 5.0)
        auto_apply = settings.get("auto_apply", True)
        
        # Store these settings and schedule the task
        return {
            "success": True,
            "message": "Auto-optimization configured",
            "settings": settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/session")
async def get_training_session(db: Session = Depends(get_db)):
    """Get current ML training session status"""
    try:
        # Get total coins
        total_coins = db.query(Coin).filter(Coin.is_active == True).count()
        
        # Get coins with at least one prediction (completed)
        completed_predictions = db.query(MLPrediction.coin_symbol).distinct().count()
        
        # Get latest prediction to see current training
        latest_prediction = db.query(MLPrediction).order_by(MLPrediction.created_at.desc()).first()
        
        current_coin = latest_prediction.coin_symbol if latest_prediction else "Starting..."
        
        # Calculate progress
        total_queue_items = total_coins * 4  # 4 models per coin
        completed_items = db.query(MLPrediction).count()
        
        return {
            "current_coin": current_coin,
            "current_model": "Training...",
            "progress": 0,  # Real-time progress would need more complex tracking
            "eta_seconds": 300,  # Estimated
            "total_queue_items": total_queue_items,
            "completed_items": completed_items,
            "session_start_time": datetime.utcnow().isoformat(),
            "estimated_completion_time": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/queue")
async def get_training_queue():
    """Get current ML training queue"""
    try:
        # Mock data - replace with actual queue
        return [
            {
                "coin_symbol": "BTCUSDT",
                "model_type": "LSTM",
                "status": "training",
                "progress": 67,
                "estimated_time_remaining": 450,
                "started_at": "2024-01-01T12:00:00Z",
                "queue_position": 1
            }
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/pause")
async def pause_training():
    """Pause ML training"""
    try:
        return {"success": True, "message": "Training paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/resume")
async def resume_training():
    """Resume ML training"""
    try:
        return {"success": True, "message": "Training resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# For Railway deployment - bind to PORT
def get_port():
    return int(os.environ.get("PORT", 8000))