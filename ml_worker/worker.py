import asyncio
import logging
import sys
import os
from datetime import datetime
from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, Coin, MLPrediction
from backend.bybit_client import BybitClient
from feature_engineering import FeatureEngineer
from models.lstm_model import LSTMModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from models.neural_network_model import NeuralNetworkModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLWorker:
    def __init__(self):
        self.bybit_client = BybitClient()
        self.feature_engineer = FeatureEngineer()
        
        self.models = {
            "lstm": LSTMModel(),
            "random_forest": RandomForestModel(),
            "svm": SVMModel(),
            "neural_network": NeuralNetworkModel()
        }
        
        self.current_coin_index = 0
        self.coins = []
        
        self.model_save_dir = "ml_worker/saved_models"
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    async def load_coins(self):
        db = SessionLocal()
        try:
            self.coins = db.query(Coin).filter(Coin.is_active == True).all()
            logger.info(f"Loaded {len(self.coins)} active coins")
        finally:
            db.close()
    
    def load_saved_models(self, symbol: str):
        for model_name, model in self.models.items():
            model_path = os.path.join(self.model_save_dir, f"{symbol}_{model_name}")
            if model.load_model(model_path):
                logger.info(f"Loaded saved model {model_name} for {symbol}")
    
    def save_models(self, symbol: str):
        for model_name, model in self.models.items():
            if model.is_trained:
                model_path = os.path.join(self.model_save_dir, f"{symbol}_{model_name}")
                model.save_model(model_path)
                logger.info(f"Saved model {model_name} for {symbol}")
    
    async def train_coin(self, coin: Coin):
        logger.info(f"Training models for {coin.symbol}")
        
        try:
            klines = self.bybit_client.get_klines(coin.symbol, interval="1", limit=1000)
            
            if len(klines) < 200:
                logger.warning(f"Insufficient data for {coin.symbol}: {len(klines)} klines")
                return
            
            features = self.feature_engineer.create_features(klines)
            
            if features is None or len(features) < 100:
                logger.warning(f"Could not create sufficient features for {coin.symbol}")
                return
            
            self.load_saved_models(coin.symbol)
            
            db = SessionLocal()
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Training {model_name} for {coin.symbol}")
                    
                    training_result = model.train(features)
                    
                    if training_result["success"]:
                        prediction_result = model.predict(features)
                        
                        if prediction_result["success"]:
                            prediction = MLPrediction(
                                coin_symbol=coin.symbol,
                                model_type=model_name,
                                confidence=prediction_result["confidence"],
                                prediction=prediction_result["prediction"],
                                features={
                                    "feature_count": len(features[0]) if len(features) > 0 else 0,
                                    "training_accuracy": training_result.get("accuracy", 0),
                                    "probabilities": prediction_result.get("probabilities", [])
                                }
                            )
                            
                            existing_prediction = db.query(MLPrediction).filter(
                                MLPrediction.coin_symbol == coin.symbol,
                                MLPrediction.model_type == model_name
                            ).order_by(MLPrediction.created_at.desc()).first()
                            
                            if existing_prediction:
                                db.delete(existing_prediction)
                            
                            db.add(prediction)
                            db.commit()
                            
                            logger.info(f"Saved prediction for {coin.symbol} {model_name}: "
                                      f"{prediction_result['prediction']} "
                                      f"({prediction_result['confidence']:.2f}%)")
                        
                        else:
                            logger.error(f"Prediction failed for {coin.symbol} {model_name}: "
                                       f"{prediction_result['error']}")
                    
                    else:
                        logger.error(f"Training failed for {coin.symbol} {model_name}: "
                                   f"{training_result['error']}")
                
                except Exception as e:
                    logger.error(f"Exception training {model_name} for {coin.symbol}: {e}")
            
            self.save_models(coin.symbol)
            db.close()
            
            coin.last_updated = datetime.utcnow()
            db = SessionLocal()
            db.merge(coin)
            db.commit()
            db.close()
            
            logger.info(f"Completed training for {coin.symbol}")
        
        except Exception as e:
            logger.error(f"Error training {coin.symbol}: {e}")
    
    async def continuous_training_loop(self):
        logger.info("Starting continuous training loop")
        
        while True:
            try:
                await self.load_coins()
                
                if not self.coins:
                    logger.warning("No active coins found, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                if self.current_coin_index >= len(self.coins):
                    self.current_coin_index = 0
                    logger.info("Completed full training cycle, starting over")
                
                current_coin = self.coins[self.current_coin_index]
                await self.train_coin(current_coin)
                
                self.current_coin_index += 1
                
                await asyncio.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(30)
    
    async def run(self):
        logger.info("ML Worker starting...")
        await self.continuous_training_loop()

async def main():
    worker = MLWorker()
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())