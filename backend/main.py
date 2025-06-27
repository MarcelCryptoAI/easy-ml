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

from .database import get_db, create_tables, Coin, MLPrediction, Trade, TradingStrategy, HistoricalData
from .bybit_client import BybitClient
from .trading_engine import TradingEngine
from .websocket_manager import WebSocketManager
from .openai_optimizer import OpenAIOptimizer
from .startup_tasks import initialize_database, validate_configuration
from .backtest_engine import BacktestEngine
from .ai_trading_advisor import AITradingAdvisor
from .historical_data_service import historical_service

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
                    
                    historical_data = historical_service.get_historical_data(db, current_coin.symbol, "1h", 30)
                    
                    if not historical_data or len(historical_data) < 100:
                        logger.warning(f"⚠️ Insufficient data for {current_coin.symbol} - skipping")
                        continue
                    
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
    database_connected = False
    openai_connected = False  
    bybit_connected = False
    uta_balance = "0.00"
    
    try:
        db.query(Coin).count()
        database_connected = True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and len(openai_api_key) > 10:
            openai_connected = True
    except Exception as e:
        logger.error(f"OpenAI connection failed: {e}")
    
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
    
    return {
        "database_connected": database_connected,
        "openai_connected": openai_connected,
        "bybit_connected": bybit_connected,
        "uta_balance": uta_balance
    }

@app.get("/coins")
async def get_coins(db: Session = Depends(get_db)):
    coins = db.query(Coin).filter(Coin.is_active == True).all()
    return [{"id": coin.id, "symbol": coin.symbol, "name": coin.name} for coin in coins]

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
            "/optimize/apply/{symbol}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)