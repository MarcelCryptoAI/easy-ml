from openai import OpenAI
import logging
import json
from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal, Trade, TradingStrategy, MLPrediction

logger = logging.getLogger(__name__)

class OpenAIOptimizer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    def analyze_performance(self, symbol: str, days: int = 7) -> Dict:
        db = SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            trades = db.query(Trade).filter(
                Trade.coin_symbol == symbol,
                Trade.opened_at >= cutoff_date,
                Trade.status == "closed"
            ).all()
            
            if not trades:
                return {"success": False, "error": "No recent trades found"}
            
            total_pnl = sum(trade.pnl for trade in trades)
            win_count = sum(1 for trade in trades if trade.pnl > 0)
            loss_count = len(trades) - win_count
            win_rate = win_count / len(trades) if trades else 0
            
            avg_win = sum(trade.pnl for trade in trades if trade.pnl > 0) / win_count if win_count > 0 else 0
            avg_loss = sum(trade.pnl for trade in trades if trade.pnl < 0) / loss_count if loss_count > 0 else 0
            
            recent_predictions = db.query(MLPrediction).filter(
                MLPrediction.coin_symbol == symbol,
                MLPrediction.created_at >= cutoff_date
            ).all()
            
            ml_accuracy_by_model = {}
            for pred in recent_predictions:
                if pred.model_type not in ml_accuracy_by_model:
                    ml_accuracy_by_model[pred.model_type] = []
                ml_accuracy_by_model[pred.model_type].append(pred.confidence)
            
            avg_ml_confidence = {
                model: sum(confidences) / len(confidences) 
                for model, confidences in ml_accuracy_by_model.items()
            }
            
            return {
                "success": True,
                "total_pnl": total_pnl,
                "trade_count": len(trades),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_ml_confidence": avg_ml_confidence,
                "trades_data": [
                    {
                        "pnl": trade.pnl,
                        "confidence": trade.ml_confidence,
                        "strategy_params": trade.strategy_params
                    } for trade in trades
                ]
            }
        
        finally:
            db.close()
    
    def get_optimization_prompt(self, symbol: str, performance_data: Dict, current_strategy: Dict) -> str:
        return f"""
Analyze the trading performance for cryptocurrency {symbol} and suggest optimizations for the trading strategy.

Current Strategy Parameters:
- Take Profit: {current_strategy['take_profit_percentage']}%
- Stop Loss: {current_strategy['stop_loss_percentage']}%
- Leverage: {current_strategy['leverage']}x
- Position Size: {current_strategy['position_size_percentage']}%
- Confidence Threshold: {current_strategy['confidence_threshold']}%

Performance Data (Last 7 days):
- Total P&L: {performance_data['total_pnl']:.2f}%
- Number of Trades: {performance_data['trade_count']}
- Win Rate: {performance_data['win_rate']:.2f}%
- Average Win: {performance_data['avg_win']:.2f}%
- Average Loss: {performance_data['avg_loss']:.2f}%
- ML Model Confidence Averages: {performance_data['avg_ml_confidence']}

Individual Trade Data: {performance_data['trades_data']}

Please analyze this data and provide optimized strategy parameters. Consider:
1. Risk-reward ratio optimization
2. Win rate vs average profit per trade
3. ML model confidence patterns
4. Volatility and market conditions for this specific coin

Respond with a JSON object containing:
{{
    "optimized_params": {{
        "take_profit_percentage": float,
        "stop_loss_percentage": float,
        "leverage": int,
        "position_size_percentage": float,
        "confidence_threshold": float
    }},
    "reasoning": "Detailed explanation of why these changes are recommended",
    "expected_improvement": "Description of expected performance improvement",
    "risk_assessment": "Assessment of increased/decreased risk with these changes"
}}

Only suggest conservative optimizations. Never suggest leverage higher than 20x or confidence threshold lower than 60%.
"""
    
    async def optimize_strategy(self, symbol: str) -> Dict:
        try:
            performance_data = self.analyze_performance(symbol)
            
            if not performance_data["success"]:
                return performance_data
            
            db = SessionLocal()
            current_strategy = db.query(TradingStrategy).filter(
                TradingStrategy.coin_symbol == symbol,
                TradingStrategy.is_active == True
            ).first()
            
            if not current_strategy:
                return {"success": False, "error": "No strategy found for symbol"}
            
            strategy_dict = {
                "take_profit_percentage": current_strategy.take_profit_percentage,
                "stop_loss_percentage": current_strategy.stop_loss_percentage,
                "leverage": current_strategy.leverage,
                "position_size_percentage": current_strategy.position_size_percentage,
                "confidence_threshold": current_strategy.confidence_threshold
            }
            
            prompt = self.get_optimization_prompt(symbol, performance_data, strategy_dict)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantitative trading analyst specializing in cryptocurrency trading strategy optimization. Provide conservative, data-driven recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            
            try:
                optimization_result = json.loads(ai_response)
                
                optimized_params = optimization_result["optimized_params"]
                
                if self.validate_parameters(optimized_params):
                    current_strategy.take_profit_percentage = optimized_params["take_profit_percentage"]
                    current_strategy.stop_loss_percentage = optimized_params["stop_loss_percentage"]
                    current_strategy.leverage = optimized_params["leverage"]
                    current_strategy.position_size_percentage = optimized_params["position_size_percentage"]
                    current_strategy.confidence_threshold = optimized_params["confidence_threshold"]
                    current_strategy.updated_by_ai = True
                    current_strategy.ai_optimization_reason = optimization_result["reasoning"]
                    current_strategy.updated_at = datetime.utcnow()
                    
                    db.commit()
                    
                    logger.info(f"Optimized strategy for {symbol}: {optimized_params}")
                    
                    return {
                        "success": True,
                        "optimized_params": optimized_params,
                        "reasoning": optimization_result["reasoning"],
                        "expected_improvement": optimization_result["expected_improvement"],
                        "risk_assessment": optimization_result["risk_assessment"]
                    }
                
                else:
                    return {"success": False, "error": "AI suggested invalid parameters"}
            
            except json.JSONDecodeError:
                logger.error(f"Could not parse AI response: {ai_response}")
                return {"success": False, "error": "Invalid AI response format"}
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_parameters(self, params: Dict) -> bool:
        required_keys = [
            "take_profit_percentage", "stop_loss_percentage", 
            "leverage", "position_size_percentage", "confidence_threshold"
        ]
        
        if not all(key in params for key in required_keys):
            return False
        
        if not (0.5 <= params["take_profit_percentage"] <= 10):
            return False
        if not (0.2 <= params["stop_loss_percentage"] <= 5):
            return False
        if not (1 <= params["leverage"] <= 20):
            return False
        if not (1 <= params["position_size_percentage"] <= 20):
            return False
        if not (60 <= params["confidence_threshold"] <= 95):
            return False
        
        return True
    
    async def batch_optimize_strategies(self) -> Dict:
        db = SessionLocal()
        try:
            strategies = db.query(TradingStrategy).filter(
                TradingStrategy.is_active == True
            ).all()
            
            results = {}
            
            for strategy in strategies:
                result = await self.optimize_strategy(strategy.coin_symbol)
                results[strategy.coin_symbol] = result
                
                if result["success"]:
                    logger.info(f"Successfully optimized strategy for {strategy.coin_symbol}")
                else:
                    logger.warning(f"Failed to optimize strategy for {strategy.coin_symbol}: {result.get('error')}")
            
            return {"success": True, "results": results}
        
        except Exception as e:
            logger.error(f"Error in batch optimization: {e}")
            return {"success": False, "error": str(e)}
        
        finally:
            db.close()