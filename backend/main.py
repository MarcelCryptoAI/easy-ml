from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict
import json
import asyncio
import logging
import os

from .database import get_db, create_tables, Coin, MLPrediction, Trade, TradingStrategy
from .bybit_client import BybitClient
from .trading_engine import TradingEngine
from .websocket_manager import WebSocketManager
from .openai_optimizer import OpenAIOptimizer
from .startup_tasks import initialize_database, validate_configuration

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
        
        await asyncio.sleep(3600)  # Sync every hour

@app.get("/")
async def root():
    return {"message": "Crypto Trading ML Platform API", "status": "operational"}

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
    predictions = db.query(MLPrediction).filter(
        MLPrediction.coin_symbol == symbol
    ).order_by(MLPrediction.created_at.desc()).limit(4).all()
    
    return [{
        "model_type": pred.model_type,
        "confidence": pred.confidence,
        "prediction": pred.prediction,
        "created_at": pred.created_at
    } for pred in predictions]

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