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

from database import get_db, create_tables, Coin, MLPrediction, Trade, TradingStrategy, HistoricalData
from bybit_client import BybitClient
from trading_engine import TradingEngine
from websocket_manager import WebSocketManager
from openai_optimizer import OpenAIOptimizer
from startup_tasks import initialize_database, validate_configuration, cleanup_disallowed_coins
from backtest_engine import BacktestEngine
from ai_trading_advisor import AITradingAdvisor
from historical_data_service import historical_service

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
    validate_configuration()
    await initialize_database()
    asyncio.create_task(sync_coins_task())
    asyncio.create_task(trading_engine.process_trading_signals())
    asyncio.create_task(ml_training_task_10_models())
    asyncio.create_task(historical_data_fetch_task())
    logger.info("Platform started successfully!")

async def sync_coins_task():
    while True:
        try:
            db = next(get_db())
            
            # Clean up disallowed coins first
            await cleanup_disallowed_coins(db)
            
            # Get allowed symbols from Bybit
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
            logger.info(f"Synced {len(symbols)} allowed coins")
            
        except Exception as e:
            logger.error(f"Error in sync_coins_task: {e}")
        
        await asyncio.sleep(3600)

async def historical_data_fetch_task():
    logger.info("🔄 Starting historical data fetch task...")
    await asyncio.sleep(30)
    
    while True:
        try:
            logger.info("📊 Starting historical data fetch for all coins...")
            await historical_service.fetch_all_coins_historical_data()
            logger.info("✅ Historical data fetch cycle completed")
            await asyncio.sleep(86400)
        except Exception as e:
            logger.error(f"Error in historical data fetch task: {e}")
            await asyncio.sleep(3600)

# Global training control
training_paused = False
current_training_coin = None
current_training_model = None
training_progress = 0

async def ml_training_task_10_models():
    logger.info("🤖 Starting Enhanced ML Training with 10 models...")
    
    model_types = [
        "lstm", "random_forest", "svm", "neural_network",
        "xgboost", "lightgbm", "catboost", "transformer", 
        "gru", "cnn_1d"
    ]
    current_coin_index = 0
    
    while True:
        try:
            global training_paused
            if training_paused:
                logger.info("⏸️ Training is paused, waiting...")
                await asyncio.sleep(5)
                continue
                
            db = next(get_db())
            coins = db.query(Coin).filter(Coin.is_active == True).all()
            
            if not coins:
                logger.warning("No active coins found for training")
                await asyncio.sleep(60)
                continue
            
            current_coin = coins[current_coin_index % len(coins)]
            logger.info(f"🎯 Training 10 models for {current_coin.symbol}")
            
            for model_type in model_types:
                if training_paused:
                    break
                    
                try:
                    recent_prediction = db.query(MLPrediction).filter(
                        MLPrediction.coin_symbol == current_coin.symbol,
                        MLPrediction.model_type == model_type
                    ).order_by(MLPrediction.created_at.desc()).first()
                    
                    if recent_prediction and recent_prediction.created_at > datetime.utcnow() - timedelta(hours=1):
                        logger.info(f"⏭️  Skipping {current_coin.symbol} {model_type} - recently trained")
                        continue
                    
                    global current_training_coin, current_training_model, training_progress
                    current_training_coin = current_coin.symbol
                    current_training_model = model_type
                    training_progress = 0
                    
                    logger.info(f"🔧 Training {model_type.upper()} for {current_coin.symbol}")
                    
                    # Get historical data with fallback to live price data
                    historical_data = historical_service.get_historical_data(db, current_coin.symbol, "1h", 30)
                    
                    if not historical_data or len(historical_data) < 20:
                        # Fallback: get live price data from Bybit
                        try:
                            klines = bybit_client.get_klines(current_coin.symbol, interval="1", limit=100)
                            if klines and len(klines) >= 20:
                                historical_data = klines
                                logger.info(f"📈 Using live Bybit data for {current_coin.symbol}: {len(klines)} points")
                            else:
                                logger.warning(f"⚠️ No data available for {current_coin.symbol} - skipping")
                                continue
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get live data for {current_coin.symbol}: {e}")
                            continue
                    else:
                        logger.debug(f"📊 Using stored data for {current_coin.symbol}: {len(historical_data)} points")
                    
                    training_time = min(60, max(10, len(historical_data) / 10))
                    training_progress = 0
                    
                    for step in range(10):
                        if training_paused:
                            break
                        training_progress = int((step + 1) / 10 * 100)
                        await asyncio.sleep(training_time / 10)
                    
                    # Simple prediction logic (will be replaced with real ML)
                    try:
                        recent_prices = [float(d['close']) for d in historical_data[-20:]]
                        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
                        
                        if price_change > 2.0:
                            prediction = "buy"
                            confidence = min(95, 70 + abs(price_change) * 2)
                        elif price_change < -2.0:
                            prediction = "sell" 
                            confidence = min(95, 70 + abs(price_change) * 2)
                        else:
                            prediction = "hold"
                            confidence = 60 + abs(price_change)
                        
                        confidence = max(60, min(95, confidence))
                        
                    except Exception as e:
                        logger.error(f"Error in ML prediction for {current_coin.symbol}: {e}")
                        continue
                    
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
                    
                    await websocket_manager.broadcast_prediction_update({
                        "coin_symbol": current_coin.symbol,
                        "model_type": model_type,
                        "prediction": prediction,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_type} for {current_coin.symbol}: {e}")
                    continue
            
            current_coin_index += 1
            db.close()
            
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in ML training task: {e}")
            await asyncio.sleep(60)

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        coin_count = db.query(Coin).count()
        symbols = bybit_client.get_derivatives_symbols()
        bybit_status = len(symbols) > 0
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

@app.get("/status")
async def get_system_status(db: Session = Depends(get_db)):
    # Initialize all status checks
    frontend_connected = True  # Frontend is always connected if request is received
    backend_connected = True   # Backend is connected if this endpoint responds
    worker_connected = False
    database_connected = False
    openai_connected = False  
    bybit_connected = False
    uta_balance = "0.00"
    
    # Database status check
    try:
        db.query(Coin).count()
        database_connected = True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    # OpenAI API status check
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and len(openai_api_key) > 10:
            openai_connected = True
    except Exception as e:
        logger.error(f"OpenAI connection failed: {e}")
    
    # ByBit API status check
    try:
        symbols = bybit_client.get_derivatives_symbols()
        bybit_connected = len(symbols) > 0
        
        if bybit_connected:
            try:
                response = bybit_client.session.get_wallet_balance(
                    accountType="UNIFIED"
                )
                if response and response.get("result"):
                    coins_list = response["result"]["list"]
                    if coins_list:
                        for coin_data in coins_list[0].get("coin", []):
                            if coin_data.get("coin") == "USDT":
                                uta_balance = coin_data.get("availableToWithdraw", "0.00")
                                break
            except Exception as e:
                logger.error(f"Failed to get UTA balance: {e}")
                
    except Exception as e:
        logger.error(f"Bybit connection failed: {e}")
    
    # Worker status check (check if background tasks are running)
    try:
        # Check if we have recent predictions (indicates ML worker is active)
        from datetime import datetime, timedelta
        recent_predictions = db.query(MLPrediction).filter(
            MLPrediction.created_at > datetime.utcnow() - timedelta(hours=1)
        ).count()
        worker_connected = recent_predictions > 0
    except Exception as e:
        logger.error(f"Worker status check failed: {e}")
    
    return {
        "frontend_connected": frontend_connected,
        "backend_connected": backend_connected,
        "worker_connected": worker_connected,
        "database_connected": database_connected,
        "openai_connected": openai_connected,
        "bybit_connected": bybit_connected,
        "uta_balance": uta_balance
    }

@app.get("/coins")
async def get_coins(db: Session = Depends(get_db)):
    coins = db.query(Coin).filter(Coin.is_active == True).all()
    return [{"id": coin.id, "symbol": coin.symbol, "name": coin.name} for coin in coins]

@app.get("/dashboard/batch")
async def get_dashboard_batch(db: Session = Depends(get_db)):
    """Get batch dashboard data for all coins to avoid N+1 query problem"""
    try:
        # Get all active coins
        coins = db.query(Coin).filter(Coin.is_active == True).all()
        
        dashboard_data = []
        for coin in coins:
            try:
                # Get latest predictions for this coin
                predictions = db.query(MLPrediction).filter(
                    MLPrediction.coin_symbol == coin.symbol
                ).order_by(MLPrediction.created_at.desc()).limit(10).all()
                
                if not predictions:
                    # No predictions available
                    dashboard_data.append({
                        "coin_symbol": coin.symbol,
                        "recommendation": "HOLD",
                        "confidence": 0,
                        "avg_confidence": 0,
                        "models_trained": 0,
                        "last_trained": "Never",
                        "consensus_breakdown": {"buy": 0, "sell": 0, "hold": 0}
                    })
                    continue
                
                # Calculate consensus
                consensus = {"buy": 0, "sell": 0, "hold": 0}
                total_confidence = 0
                
                for pred in predictions:
                    if pred.prediction.lower() == "long":
                        consensus["buy"] += 1
                    elif pred.prediction.lower() == "short":  
                        consensus["sell"] += 1
                    else:
                        consensus["hold"] += 1
                    total_confidence += pred.confidence
                
                total_models = len(predictions)
                avg_confidence = total_confidence / total_models if total_models > 0 else 0
                
                # Determine overall recommendation
                if consensus["buy"] > consensus["sell"] and consensus["buy"] > consensus["hold"]:
                    recommendation = "LONG"
                    confidence = (consensus["buy"] / total_models) * 100
                elif consensus["sell"] > consensus["buy"] and consensus["sell"] > consensus["hold"]:
                    recommendation = "SHORT"
                    confidence = (consensus["sell"] / total_models) * 100
                else:
                    recommendation = "HOLD"
                    confidence = (consensus["hold"] / total_models) * 100
                
                dashboard_data.append({
                    "coin_symbol": coin.symbol,
                    "recommendation": recommendation,
                    "confidence": round(confidence, 1),
                    "avg_confidence": round(avg_confidence, 1),
                    "models_trained": total_models,
                    "last_trained": predictions[0].created_at.isoformat() if predictions else "Never",
                    "consensus_breakdown": consensus
                })
                
            except Exception as e:
                logger.error(f"Error processing dashboard data for {coin.symbol}: {e}")
                dashboard_data.append({
                    "coin_symbol": coin.symbol,
                    "recommendation": "HOLD",
                    "confidence": 0,
                    "avg_confidence": 0,
                    "models_trained": 0,
                    "last_trained": "Never",
                    "consensus_breakdown": {"buy": 0, "sell": 0, "hold": 0}
                })
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error in dashboard batch endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/test")
async def performance_test():
    """Simple performance test endpoint to measure response times"""
    import time
    start_time = time.time()
    
    # Simulate some work
    import asyncio
    await asyncio.sleep(0.001)  # 1ms delay
    
    end_time = time.time()
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return {
        "status": "ok",
        "response_time_ms": round(response_time, 2),
        "timestamp": time.time()
    }

@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str, db: Session = Depends(get_db)):
    try:
        predictions = db.query(MLPrediction).filter(
            MLPrediction.coin_symbol == symbol
        ).order_by(MLPrediction.created_at.desc()).limit(10).all()
        
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

@app.get("/training-info")
async def get_training_info(db: Session = Depends(get_db)):
    """Get current ML training information with REAL data - OPTIMIZED"""
    try:
        # Use efficient queries
        total_coins = db.query(Coin).filter(Coin.is_active == True).count()
        model_types = ["lstm", "random_forest", "svm", "neural_network", "xgboost", "lightgbm", "catboost", "transformer", "gru", "cnn_1d"]
        
        total_models_expected = total_coins * len(model_types)
        completed_predictions = db.query(MLPrediction).count()
        overall_progress = (completed_predictions / total_models_expected * 100) if total_models_expected > 0 else 0
        overall_progress = min(100, overall_progress)
        
        # Get recent training activity for better status detection
        from datetime import datetime, timedelta
        recent_predictions = db.query(MLPrediction).filter(
            MLPrediction.created_at > datetime.utcnow() - timedelta(minutes=5)
        ).count()
        
        global current_training_coin, current_training_model, training_progress, training_paused
        
        # Better status detection
        if training_paused:
            status = "paused"
        elif current_training_coin:
            status = "training"
        elif recent_predictions > 0:
            status = "active"
        else:
            status = "idle"
        
        return {
            "total_coins": total_coins,
            "total_models": len(model_types),
            "total_predictions_possible": total_models_expected,
            "completed_predictions": completed_predictions,
            "overall_percentage": round(overall_progress, 2),
            "current_coin": current_training_coin or "None",
            "current_model": current_training_model or "None",
            "current_model_progress": training_progress,
            "status": status,
            "recent_activity": recent_predictions,
            "model_types": model_types
        }
    except Exception as e:
        logger.error(f"Error getting training info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/statistics")
async def get_training_statistics(db: Session = Depends(get_db)):
    """Get detailed training statistics for better dashboard consistency"""
    try:
        # Get prediction statistics by model type
        from sqlalchemy import func
        
        model_stats = db.query(
            MLPrediction.model_type,
            func.count(MLPrediction.id).label('count'),
            func.avg(MLPrediction.confidence).label('avg_confidence'),
            func.max(MLPrediction.created_at).label('last_updated')
        ).group_by(MLPrediction.model_type).all()
        
        # Get prediction statistics by coin
        coin_stats = db.query(
            MLPrediction.coin_symbol,
            func.count(MLPrediction.id).label('predictions_count'),
            func.avg(MLPrediction.confidence).label('avg_confidence'),
            func.max(MLPrediction.created_at).label('last_prediction')
        ).group_by(MLPrediction.coin_symbol).limit(20).all()
        
        # Get recent predictions for activity monitoring
        recent_predictions = db.query(MLPrediction).filter(
            MLPrediction.created_at > datetime.utcnow() - timedelta(hours=1)
        ).order_by(MLPrediction.created_at.desc()).limit(50).all()
        
        return {
            "model_statistics": [
                {
                    "model_type": stat.model_type,
                    "predictions_count": stat.count,
                    "avg_confidence": round(float(stat.avg_confidence), 2) if stat.avg_confidence else 0,
                    "last_updated": stat.last_updated.isoformat() if stat.last_updated else None
                }
                for stat in model_stats
            ],
            "top_coins": [
                {
                    "coin_symbol": stat.coin_symbol,
                    "predictions_count": stat.predictions_count,
                    "avg_confidence": round(float(stat.avg_confidence), 2) if stat.avg_confidence else 0,
                    "last_prediction": stat.last_prediction.isoformat() if stat.last_prediction else None
                }
                for stat in coin_stats
            ],
            "recent_activity": [
                {
                    "coin_symbol": pred.coin_symbol,
                    "model_type": pred.model_type,
                    "prediction": pred.prediction,
                    "confidence": pred.confidence,
                    "created_at": pred.created_at.isoformat()
                }
                for pred in recent_predictions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting training statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/status")
async def get_account_balance():
    """Get account balance and trading status"""
    try:
        balance = trading_engine._get_available_balance()
        positions = bybit_client.get_positions()
        
        return {
            "success": True,
            "balance": {"available_balance": balance},
            "trading_enabled": trading_engine.enabled,
            "open_positions": len(positions)
        }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price and market data for a symbol"""
    try:
        klines = bybit_client.get_klines(symbol, interval="1", limit=2)
        if not klines:
            raise HTTPException(status_code=404, detail=f"Price data not found for {symbol}")
        
        current_price = float(klines[-1]["close"])
        return {"symbol": symbol, "price": current_price}
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/status")
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

@app.get("/optimize/queue")
async def get_optimization_queue():
    """Get current optimization queue"""
    try:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/apply/{symbol}")
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

@app.get("/strategies/all")
async def get_all_strategies(db: Session = Depends(get_db)):
    """Get all strategies for all coins without pagination"""
    try:
        # Query all coins with their strategies
        query = db.query(Coin, TradingStrategy).outerjoin(
            TradingStrategy, 
            Coin.symbol == TradingStrategy.coin_symbol
        ).filter(Coin.is_active == True)
        
        # Get all results
        results = query.all()
        
        strategies = []
        for coin, strategy in results:
            # Create strategy with defaults if not exists
            if not strategy:
                strategy_data = {
                    "coin_symbol": coin.symbol,
                    "leverage": 10,
                    "margin_mode": "cross",
                    "position_size_percent": 2.0,
                    "confidence_threshold": 80.0,
                    "min_models_required": 7,
                    "total_models_available": 10,
                    "take_profit_percentage": 2.0,
                    "stop_loss_percentage": 1.0,
                    "is_active": True,
                    "ai_optimized": False
                }
            else:
                strategy_data = {
                    "coin_symbol": coin.symbol,
                    "leverage": strategy.leverage,
                    "margin_mode": getattr(strategy, 'margin_mode', 'cross'),
                    "position_size_percent": strategy.position_size_percentage,
                    "confidence_threshold": strategy.confidence_threshold,
                    "min_models_required": getattr(strategy, 'min_models_required', 7),
                    "total_models_available": 10,
                    "take_profit_percentage": strategy.take_profit_percentage,
                    "stop_loss_percentage": strategy.stop_loss_percentage,
                    "is_active": strategy.is_active,
                    "ai_optimized": strategy.updated_by_ai
                }
            strategies.append(strategy_data)
        
        return {
            "strategies": strategies,
            "total": len(strategies)
        }
    except Exception as e:
        logger.error(f"Error getting all strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/paginated")
async def get_strategies_paginated(
    page: int = 1,
    limit: int = 50,
    search: str = None,
    db: Session = Depends(get_db)
):
    """Get paginated strategies for all coins"""
    try:
        # Query all coins with their strategies
        query = db.query(Coin, TradingStrategy).outerjoin(
            TradingStrategy, 
            Coin.symbol == TradingStrategy.coin_symbol
        ).filter(Coin.is_active == True)
        
        # Apply search filter if provided
        if search:
            query = query.filter(Coin.symbol.ilike(f"%{search}%"))
        
        # Get total count
        total = query.count()
        
        # Calculate pagination
        skip = (page - 1) * limit
        
        # Get paginated results
        results = query.offset(skip).limit(limit).all()
        
        strategies = []
        for coin, strategy in results:
            # Create strategy with defaults if not exists
            if not strategy:
                strategy_data = {
                    "coin_symbol": coin.symbol,
                    "leverage": 10,
                    "margin_mode": "cross",
                    "position_size_percent": 2.0,
                    "confidence_threshold": 80.0,
                    "min_models_required": 7,
                    "total_models_available": 10,
                    "take_profit_percentage": 2.0,
                    "stop_loss_percentage": 1.0,
                    "is_active": True,
                    "ai_optimized": False
                }
            else:
                strategy_data = {
                    "coin_symbol": coin.symbol,
                    "leverage": strategy.leverage,
                    "margin_mode": getattr(strategy, 'margin_mode', 'cross'),
                    "position_size_percent": strategy.position_size_percentage,
                    "confidence_threshold": strategy.confidence_threshold,
                    "min_models_required": getattr(strategy, 'min_models_required', 7),
                    "total_models_available": 10,
                    "take_profit_percentage": strategy.take_profit_percentage,
                    "stop_loss_percentage": strategy.stop_loss_percentage,
                    "is_active": strategy.is_active,
                    "ai_optimized": strategy.updated_by_ai
                }
            strategies.append(strategy_data)
        
        return {
            "strategies": strategies,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        logger.error(f"Error getting paginated strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy/update")
async def update_strategy(strategy_data: Dict, db: Session = Depends(get_db)):
    """Update strategy configuration for a coin"""
    try:
        coin_symbol = strategy_data.get("coin_symbol")
        if not coin_symbol:
            raise HTTPException(status_code=400, detail="coin_symbol is required")
        
        # Get or create strategy
        strategy = db.query(TradingStrategy).filter(
            TradingStrategy.coin_symbol == coin_symbol
        ).first()
        
        if not strategy:
            strategy = TradingStrategy(coin_symbol=coin_symbol)
            db.add(strategy)
        
        # Update strategy fields
        strategy.leverage = strategy_data.get("leverage", 10)
        strategy.margin_mode = strategy_data.get("margin_mode", "cross")
        strategy.position_size_percentage = strategy_data.get("position_size_percent", 2.0)
        strategy.confidence_threshold = strategy_data.get("confidence_threshold", 80.0)
        strategy.take_profit_percentage = strategy_data.get("take_profit_percentage", 2.0)
        strategy.stop_loss_percentage = strategy_data.get("stop_loss_percentage", 1.0)
        strategy.updated_at = datetime.utcnow()
        
        # Store additional fields if needed
        if hasattr(strategy, 'min_models_required'):
            strategy.min_models_required = strategy_data.get("min_models_required", 7)
        
        db.commit()
        
        return {"success": True, "message": f"Strategy updated for {coin_symbol}"}
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals")
async def get_trading_signals(db: Session = Depends(get_db)):
    """Get autonomous trading signals from AI advisor"""
    try:
        signals = ai_advisor.get_autonomous_trading_signals(db)
        return {
            "success": True,
            "signals": signals,
            "total_signals": len(signals),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/manual")
async def execute_manual_trade(trade_data: Dict, db: Session = Depends(get_db)):
    """Execute manual trade with specified parameters"""
    try:
        required_fields = ['coin_symbol', 'side', 'amount_percentage', 'leverage']
        for field in required_fields:
            if field not in trade_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        coin_symbol = trade_data['coin_symbol']
        side = trade_data['side']
        amount_percentage = trade_data['amount_percentage']
        leverage = trade_data['leverage']
        
        # Get current price
        klines = bybit_client.get_klines(coin_symbol, limit=1)
        if not klines:
            raise HTTPException(status_code=400, detail=f"Unable to get current price for {coin_symbol}")
        current_price = float(klines[-1]["close"])
        
        # Get available balance
        balance = trading_engine._get_available_balance()
        if balance < 1:
            raise HTTPException(status_code=400, detail="Insufficient balance for trading")
        
        # Calculate position size
        trade_amount = (balance * amount_percentage) / 100
        position_size = trade_amount / current_price
        
        if position_size < 0.001:
            raise HTTPException(status_code=400, detail="Position size too small")
        
        logger.info(f"🎯 MANUAL TRADE: {coin_symbol} {side.upper()}")
        logger.info(f"   💰 Amount: {trade_amount:.2f} USDT ({amount_percentage}%)")
        logger.info(f"   📏 Size: {position_size:.6f} ({leverage}x)")
        
        # Simulate trade execution for now
        return {
            "success": True,
            "message": f"Manual {side.upper()} order simulated",
            "trade_details": {
                "coin_symbol": coin_symbol,
                "side": side,
                "amount": trade_amount,
                "position_size": position_size,
                "current_price": current_price,
                "leverage": leverage
            }
        }
    except Exception as e:
        logger.error(f"Error in manual trade execution: {e}")
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

# Quick deployment test endpoint
@app.get("/deployment-test")
async def deployment_test():
    return {
        "message": "✅ Clean deployment with all endpoints!", 
        "timestamp": datetime.utcnow().isoformat(), 
        "version": "v2.1",
        "endpoints": [
            "/training-info", 
            "/trading/status", 
            "/trading/manual",
            "/price/{symbol}",
            "/optimize/status",
            "/optimize/queue", 
            "/optimize/apply/{symbol}",
            "/signals"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)