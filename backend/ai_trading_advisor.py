#!/usr/bin/env python3

from openai import OpenAI
import os
import json
import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import numpy as np

from database import MLPrediction, TradingStrategy, Coin

logger = logging.getLogger(__name__)

class AITradingAdvisor:
    """AI-powered trading advisor that analyzes ML predictions and provides autonomous recommendations"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_types = [
            "lstm", "random_forest", "svm", "neural_network", 
            "xgboost", "lightgbm", "catboost", "transformer", 
            "gru", "cnn_1d"
        ]
    
    def get_autonomous_recommendation(self, db: Session, coin_symbol: str) -> Dict:
        """
        Generate autonomous trading recommendation based on all 10 ML models
        Returns: {
            "recommendation": "LONG" | "SHORT" | "HOLD",
            "confidence": float,
            "reasoning": str,
            "model_consensus": dict,
            "avg_confidence": float
        }
        """
        try:
            # Get latest predictions for all 10 models
            predictions = db.query(MLPrediction).filter(
                MLPrediction.coin_symbol == coin_symbol,
                MLPrediction.created_at > datetime.utcnow() - timedelta(hours=24)
            ).order_by(MLPrediction.created_at.desc()).all()
            
            if not predictions:
                return {
                    "recommendation": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "No recent ML predictions available",
                    "model_consensus": {},
                    "avg_confidence": 0.0
                }
            
            # Group by model type and get latest prediction per model
            latest_predictions = {}
            for pred in predictions:
                if pred.model_type not in latest_predictions:
                    latest_predictions[pred.model_type] = pred
            
            # Calculate consensus
            buy_votes = 0
            sell_votes = 0
            hold_votes = 0
            total_confidence = 0
            model_consensus = {}
            
            for model_type, pred in latest_predictions.items():
                model_consensus[model_type] = {
                    "prediction": pred.prediction,
                    "confidence": pred.confidence
                }
                total_confidence += pred.confidence
                
                if pred.prediction.upper() in ["BUY", "LONG"]:
                    buy_votes += 1
                elif pred.prediction.upper() in ["SELL", "SHORT"]:
                    sell_votes += 1
                else:
                    hold_votes += 1
            
            total_models = len(latest_predictions)
            avg_confidence = total_confidence / total_models if total_models > 0 else 0
            
            # Determine recommendation based on consensus
            if buy_votes >= 6:  # 60% consensus for LONG
                recommendation = "LONG"
                confidence = (buy_votes / total_models) * (avg_confidence / 100)
            elif sell_votes >= 6:  # 60% consensus for SHORT
                recommendation = "SHORT"
                confidence = (sell_votes / total_models) * (avg_confidence / 100)
            else:
                recommendation = "HOLD"
                confidence = (hold_votes / total_models) * (avg_confidence / 100)
            
            # Generate AI reasoning
            reasoning = self._generate_ai_reasoning(
                coin_symbol, model_consensus, recommendation, confidence, avg_confidence
            )
            
            return {
                "recommendation": recommendation,
                "confidence": round(confidence * 100, 1),
                "reasoning": reasoning,
                "model_consensus": model_consensus,
                "avg_confidence": round(avg_confidence, 1),
                "models_count": total_models,
                "consensus_breakdown": {
                    "buy": buy_votes,
                    "sell": sell_votes, 
                    "hold": hold_votes
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {coin_symbol}: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error analyzing predictions: {str(e)}",
                "model_consensus": {},
                "avg_confidence": 0.0
            }
    
    def _generate_ai_reasoning(self, coin_symbol: str, model_consensus: Dict, 
                              recommendation: str, confidence: float, avg_confidence: float) -> str:
        """Generate AI-powered reasoning for the recommendation"""
        try:
            # Create prompt for OpenAI
            prompt = f"""
            As an expert crypto trading AI, analyze this ML model consensus for {coin_symbol}:

            MODEL PREDICTIONS:
            {json.dumps(model_consensus, indent=2)}

            FINAL RECOMMENDATION: {recommendation}
            CONFIDENCE: {confidence*100:.1f}%
            AVERAGE MODEL CONFIDENCE: {avg_confidence:.1f}%

            Provide a concise 2-3 sentence explanation of why this recommendation makes sense based on:
            1. Model consensus strength
            2. Average confidence levels
            3. Risk-reward assessment

            Keep it professional and actionable for autonomous trading.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            return f"Strong {recommendation} signal with {avg_confidence:.1f}% avg confidence across {len(model_consensus)} models."
    
    def optimize_strategy_with_backtest(self, db: Session, coin_symbol: str) -> Dict:
        """
        Use AI to optimize trading strategy for a coin with backtesting validation
        """
        try:
            # Get current strategy
            strategy = db.query(TradingStrategy).filter(
                TradingStrategy.coin_symbol == coin_symbol
            ).first()
            
            if not strategy:
                # Create default strategy
                strategy = TradingStrategy(
                    coin_symbol=coin_symbol,
                    min_confidence_threshold=70.0,
                    position_size_percentage=2.0,
                    stop_loss_percentage=3.0,
                    take_profit_percentage=6.0,
                    is_active=True
                )
                db.add(strategy)
                db.commit()
            
            # Get recent performance data
            recent_predictions = db.query(MLPrediction).filter(
                MLPrediction.coin_symbol == coin_symbol,
                MLPrediction.created_at > datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # AI optimization prompt
            prompt = f"""
            Optimize trading strategy for {coin_symbol} based on recent ML performance:
            
            CURRENT STRATEGY:
            - Min Confidence: {strategy.min_confidence_threshold}%
            - Position Size: {strategy.position_size_percentage}%
            - Stop Loss: {strategy.stop_loss_percentage}%
            - Take Profit: {strategy.take_profit_percentage}%
            
            RECENT PERFORMANCE:
            - Total Predictions: {len(recent_predictions)}
            - Avg Confidence: {np.mean([p.confidence for p in recent_predictions]):.1f}%
            
            Provide optimized parameters as JSON:
            {{
                "min_confidence_threshold": float,
                "position_size_percentage": float,
                "stop_loss_percentage": float,
                "take_profit_percentage": float,
                "reasoning": "why these changes"
            }}
            
            Focus on risk-adjusted returns and consistent performance.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content.strip()
            optimized_params = json.loads(ai_response)
            
            # Update strategy with AI optimizations
            strategy.min_confidence_threshold = optimized_params["min_confidence_threshold"]
            strategy.position_size_percentage = optimized_params["position_size_percentage"]
            strategy.stop_loss_percentage = optimized_params["stop_loss_percentage"] 
            strategy.take_profit_percentage = optimized_params["take_profit_percentage"]
            strategy.last_optimized = datetime.utcnow()
            
            db.commit()
            
            return {
                "success": True,
                "optimized_strategy": {
                    "min_confidence_threshold": strategy.min_confidence_threshold,
                    "position_size_percentage": strategy.position_size_percentage,
                    "stop_loss_percentage": strategy.stop_loss_percentage,
                    "take_profit_percentage": strategy.take_profit_percentage
                },
                "ai_reasoning": optimized_params.get("reasoning", ""),
                "backtest_score": "Coming soon"  # TODO: Implement actual backtesting
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy for {coin_symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_autonomous_trading_signals(self, db: Session) -> List[Dict]:
        """
        Get autonomous trading signals for all coins with high confidence
        """
        try:
            # Get all active coins
            coins = db.query(Coin).filter(Coin.is_active == True).all()
            signals = []
            
            for coin in coins:
                recommendation = self.get_autonomous_recommendation(db, coin.symbol)
                
                # Only include high confidence signals
                if recommendation["confidence"] >= 75.0 and recommendation["recommendation"] != "HOLD":
                    signals.append({
                        "coin_symbol": coin.symbol,
                        "signal": recommendation["recommendation"],
                        "confidence": recommendation["confidence"],
                        "avg_model_confidence": recommendation["avg_confidence"],
                        "models_consensus": recommendation["consensus_breakdown"],
                        "reasoning": recommendation["reasoning"]
                    })
            
            # Sort by confidence
            signals.sort(key=lambda x: x["confidence"], reverse=True)
            
            return signals[:20]  # Top 20 signals
            
        except Exception as e:
            logger.error(f"Error getting autonomous trading signals: {e}")
            return []