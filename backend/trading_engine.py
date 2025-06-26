import asyncio
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, List

from .database import SessionLocal, MLPrediction, Trade, TradingStrategy
from .bybit_client import BybitClient
from .websocket_manager import WebSocketManager
from .config import settings

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, bybit_client: BybitClient, websocket_manager: WebSocketManager):
        self.bybit_client = bybit_client
        self.websocket_manager = websocket_manager
        self.enabled = False
        self.max_positions = settings.max_positions
        
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        logger.info(f"Trading engine {'enabled' if enabled else 'disabled'}")
    
    async def process_trading_signals(self):
        while True:
            try:
                if not self.enabled:
                    await asyncio.sleep(10)
                    continue
                
                db = SessionLocal()
                
                current_positions = self.bybit_client.get_positions()
                if len(current_positions) >= self.max_positions:
                    logger.info(f"Max positions ({self.max_positions}) reached")
                    db.close()
                    await asyncio.sleep(30)
                    continue
                
                predictions = self._get_latest_predictions(db)
                
                for symbol, symbol_predictions in predictions.items():
                    if self._has_open_position(symbol, current_positions):
                        continue
                    
                    strategy = self._get_strategy(db, symbol)
                    if not strategy:
                        continue
                    
                    signal = self._evaluate_predictions(symbol_predictions, strategy)
                    
                    if signal:
                        await self._execute_trade(db, symbol, signal, strategy)
                
                await self._monitor_open_trades(db)
                
                db.close()
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in trading engine: {e}")
                await asyncio.sleep(10)
    
    def _get_latest_predictions(self, db: Session) -> Dict[str, List[MLPrediction]]:
        predictions_by_symbol = {}
        
        latest_predictions = db.query(MLPrediction).order_by(
            MLPrediction.coin_symbol,
            MLPrediction.created_at.desc()
        ).all()
        
        for prediction in latest_predictions:
            symbol = prediction.coin_symbol
            if symbol not in predictions_by_symbol:
                predictions_by_symbol[symbol] = []
            
            if len(predictions_by_symbol[symbol]) < 4:
                predictions_by_symbol[symbol].append(prediction)
        
        return {k: v for k, v in predictions_by_symbol.items() if len(v) == 4}
    
    def _get_strategy(self, db: Session, symbol: str) -> TradingStrategy:
        strategy = db.query(TradingStrategy).filter(
            TradingStrategy.coin_symbol == symbol,
            TradingStrategy.is_active == True
        ).first()
        
        if not strategy:
            strategy = TradingStrategy(
                coin_symbol=symbol,
                take_profit_percentage=settings.take_profit_percentage,
                stop_loss_percentage=settings.stop_loss_percentage
            )
            db.add(strategy)
            db.commit()
        
        return strategy
    
    def _has_open_position(self, symbol: str, positions: List[Dict]) -> bool:
        return any(pos["symbol"] == symbol for pos in positions)
    
    def _evaluate_predictions(self, predictions: List[MLPrediction], strategy: TradingStrategy) -> Dict:
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        
        for prediction in predictions:
            if prediction.confidence >= strategy.confidence_threshold:
                if prediction.prediction == "buy":
                    buy_signals += 1
                elif prediction.prediction == "sell":
                    sell_signals += 1
                total_confidence += prediction.confidence
        
        if buy_signals >= 3:
            return {
                "side": "buy",
                "confidence": total_confidence / len(predictions)
            }
        elif sell_signals >= 3:
            return {
                "side": "sell", 
                "confidence": total_confidence / len(predictions)
            }
        
        return None
    
    async def _execute_trade(self, db: Session, symbol: str, signal: Dict, strategy: TradingStrategy):
        try:
            klines = self.bybit_client.get_klines(symbol, limit=1)
            if not klines:
                logger.error(f"No price data for {symbol}")
                return
            
            current_price = klines[-1]["close"]
            
            if signal["side"] == "buy":
                take_profit_price = current_price * (1 + strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 - strategy.stop_loss_percentage / 100)
            else:
                take_profit_price = current_price * (1 - strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 + strategy.stop_loss_percentage / 100)
            
            position_size = self._calculate_position_size(current_price, strategy)
            
            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=signal["side"],
                qty=position_size,
                leverage=strategy.leverage,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price
            )
            
            if order_result["success"]:
                trade = Trade(
                    coin_symbol=symbol,
                    order_id=order_result["order_id"],
                    side=signal["side"],
                    size=position_size,
                    price=current_price,
                    leverage=strategy.leverage,
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    status="open",
                    ml_confidence=signal["confidence"],
                    strategy_params={
                        "take_profit_percentage": strategy.take_profit_percentage,
                        "stop_loss_percentage": strategy.stop_loss_percentage,
                        "leverage": strategy.leverage
                    }
                )
                
                db.add(trade)
                db.commit()
                
                await self.websocket_manager.broadcast_trade_update({
                    "action": "opened",
                    "symbol": symbol,
                    "side": signal["side"],
                    "price": current_price,
                    "confidence": signal["confidence"]
                })
                
                logger.info(f"Trade executed: {symbol} {signal['side']} at {current_price}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def _get_available_balance(self) -> float:
        """Get available USDT balance from Bybit Unified Trading Account (UTA)"""
        try:
            response = self.bybit_client.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if response["retCode"] == 0:
                # Get the first account list (should be UTA)
                account_list = response["result"]["list"][0]
                # Find USDT coin in the account
                for coin_info in account_list["coin"]:
                    if coin_info["coin"] == "USDT":
                        # Use availableBalance for UTA trading
                        return float(coin_info["availableBalance"])
                return 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def _calculate_position_size(self, price: float, strategy: TradingStrategy) -> float:
        """Calculate position size based on real account balance"""
        available_balance = self._get_available_balance()
        
        if available_balance <= 0:
            logger.error("No available balance for trading")
            return 0.0
        
        position_value = available_balance * (strategy.position_size_percentage / 100)
        position_size = position_value / price
        
        # Ensure minimum position size requirements
        if position_size < 0.001:
            logger.warning(f"Position size {position_size} too small, skipping trade")
            return 0.0
        
        return position_size
    
    async def _monitor_open_trades(self, db: Session):
        open_trades = db.query(Trade).filter(Trade.status == "open").all()
        current_positions = self.bybit_client.get_positions()
        
        for trade in open_trades:
            position = next((p for p in current_positions if p["symbol"] == trade.coin_symbol), None)
            
            if not position:
                trade.status = "closed"
                trade.closed_at = datetime.utcnow()
                
                klines = self.bybit_client.get_klines(trade.coin_symbol, limit=1)
                if klines:
                    current_price = klines[-1]["close"]
                    if trade.side == "buy":
                        trade.pnl = (current_price - trade.price) / trade.price * 100
                    else:
                        trade.pnl = (trade.price - current_price) / trade.price * 100
                
                db.commit()
                
                await self.websocket_manager.broadcast_trade_update({
                    "action": "closed",
                    "symbol": trade.coin_symbol,
                    "pnl": trade.pnl
                })
                
                logger.info(f"Trade closed: {trade.coin_symbol} PnL: {trade.pnl:.2f}%")