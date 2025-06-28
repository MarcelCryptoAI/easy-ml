from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

app = FastAPI(title="Test Trading Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "✅ Backend is ONLINE!", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "backend"}

@app.get("/status")
async def status():
    return {
        "database_connected": True,
        "openai_connected": True,
        "bybit_connected": True,
        "uta_balance": "100.00"
    }

@app.get("/deployment-test")
async def deployment_test():
    return {
        "message": "✅ Test deployment successful!", 
        "timestamp": datetime.utcnow().isoformat(),
        "version": "test-v1.0"
    }

@app.get("/training-info")
async def training_info():
    return {
        "total_coins": 502,
        "total_models": 10,
        "completed_predictions": 3520,
        "overall_percentage": 70.2,
        "status": "training"
    }

@app.get("/trading/status") 
async def trading_status():
    return {
        "success": True,
        "balance": {"available_balance": 100.00},
        "trading_enabled": True,
        "open_positions": 0
    }

@app.get("/signals")
async def get_trading_signals():
    return {
        "success": True,
        "signals": [
            {
                "coin_symbol": "BTCUSDT",
                "signal": "LONG",
                "confidence": 85.2,
                "avg_model_confidence": 82.5,
                "models_consensus": {"buy": 7, "sell": 1, "hold": 2},
                "reasoning": "Strong bullish consensus across 7 models with high confidence levels"
            },
            {
                "coin_symbol": "ETHUSDT", 
                "signal": "SHORT",
                "confidence": 78.9,
                "avg_model_confidence": 76.2,
                "models_consensus": {"buy": 2, "sell": 6, "hold": 2},
                "reasoning": "Bearish trend detected with 6 models agreeing on sell signal"
            }
        ],
        "total_signals": 2,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/price/{symbol}")
async def get_price(symbol: str):
    return {
        "symbol": symbol,
        "price": 45250.75 if "BTC" in symbol else 2850.25
    }

@app.post("/trading/manual")
async def manual_trade(trade_data: dict):
    return {
        "success": True,
        "message": f"Manual {trade_data.get('side', 'BUY')} order simulated",
        "trade_details": {
            "coin_symbol": trade_data.get("coin_symbol", "BTCUSDT"),
            "side": trade_data.get("side", "buy"),
            "amount": 100.0,
            "position_size": 0.002,
            "current_price": 45250.75,
            "leverage": trade_data.get("leverage", 10)
        }
    }

@app.get("/strategies/paginated")
async def get_strategies_paginated(page: int = 1, limit: int = 50, search: str = None):
    # Generate sample strategies
    all_coins = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
        "AVAXUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT", "ATOMUSDT", "LTCUSDT", "UNIUSDT",
        "NEARUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "LDOUSDT", "STXUSDT", "FILUSDT",
        "IMXUSDT", "HBARUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "KASUSDT"
    ]
    
    # Filter by search if provided
    filtered_coins = all_coins
    if search:
        filtered_coins = [c for c in all_coins if search.upper() in c]
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    paginated_coins = filtered_coins[start:end]
    
    strategies = []
    for coin in paginated_coins:
        strategies.append({
            "coin_symbol": coin,
            "leverage": 10,
            "margin_mode": "cross",
            "position_size_percent": 2.0,
            "confidence_threshold": 80.0,
            "min_models_required": 7,
            "total_models_available": 10,
            "take_profit_percentage": 2.0,
            "stop_loss_percentage": 1.0,
            "is_active": True,
            "ai_optimized": coin in ["BTCUSDT", "ETHUSDT"]
        })
    
    return {
        "strategies": strategies,
        "total": len(filtered_coins),
        "page": page,
        "pages": (len(filtered_coins) + limit - 1) // limit
    }

@app.post("/strategy/update")
async def update_strategy(strategy_data: dict):
    return {
        "success": True,
        "message": f"Strategy updated for {strategy_data.get('coin_symbol', 'UNKNOWN')}"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)