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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)