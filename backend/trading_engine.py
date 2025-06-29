import asyncio
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, List

from database import SessionLocal, MLPrediction, Trade, TradingStrategy
from bybit_client import BybitClient
from websocket_manager import WebSocketManager
from config import settings

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, bybit_client: BybitClient, websocket_manager: WebSocketManager):
        self.bybit_client = bybit_client
        self.websocket_manager = websocket_manager
        self.enabled = False
        self.max_positions = settings.max_positions
        self.auto_start_trading = True  # Auto-enable trading on startup
        self.min_balance_required = 10.0  # Minimum USDT balance to start trading
        self.max_daily_loss_percentage = 5.0  # Stop trading if daily loss exceeds 5%
        self.daily_loss_tracker = 0.0
        self.trade_count_today = 0
        self.last_reset_date = datetime.utcnow().date()
        
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        logger.info(f"Trading engine {'enabled' if enabled else 'disabled'}")
    
    async def auto_startup_check(self):
        """Automatically enable trading if conditions are met"""
        if not self.auto_start_trading:
            return
            
        try:
            # Check if we have sufficient balance
            balance = self._get_available_balance()
            if balance < self.min_balance_required:
                logger.warning(f"Insufficient balance for auto-trading: {balance} USDT (min: {self.min_balance_required})")
                return
            
            # Check if Bybit connection is working
            symbols = self.bybit_client.get_derivatives_symbols()
            if not symbols or len(symbols) == 0:
                logger.warning("Bybit connection failed, cannot auto-start trading")
                return
            
            # Check if we haven't exceeded daily loss limit
            if self.daily_loss_tracker >= self.max_daily_loss_percentage:
                logger.warning(f"Daily loss limit exceeded: {self.daily_loss_tracker}%, trading disabled")
                return
            
            # All checks passed - enable trading
            self.enabled = True
            logger.info(f"🚀 AUTO-TRADING ENABLED! Balance: {balance} USDT, Symbols: {len(symbols)}")
            
        except Exception as e:
            logger.error(f"Error in auto-startup check: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics if it's a new day"""
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_loss_tracker = 0.0
            self.trade_count_today = 0
            self.last_reset_date = current_date
            logger.info(f"📅 Daily stats reset for {current_date}")
    
    def check_risk_limits(self) -> bool:
        """Check if trading should continue based on risk limits"""
        self.reset_daily_stats()
        
        # Check daily loss limit
        if self.daily_loss_tracker >= self.max_daily_loss_percentage:
            logger.warning(f"🛑 Daily loss limit exceeded: {self.daily_loss_tracker}%")
            self.enabled = False
            return False
        
        # Check balance
        balance = self._get_available_balance()
        if balance < self.min_balance_required:
            logger.warning(f"🛑 Balance too low: {balance} USDT (min: {self.min_balance_required})")
            self.enabled = False
            return False
        
        return True
    
    async def process_trading_signals(self):
        # Auto-startup check on first run
        await self.auto_startup_check()
        
        while True:
            try:
                # Check risk limits before processing
                if not self.check_risk_limits():
                    await asyncio.sleep(30)
                    continue
                    
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
        """Enhanced prediction evaluation with configurable model agreement"""
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        total_confidence = 0
        valid_predictions = 0
        
        # Get strategy parameters with defaults
        min_models_required = getattr(strategy, 'min_models_required', 7)
        confidence_threshold = getattr(strategy, 'confidence_threshold', 80.0)
        
        for prediction in predictions:
            if prediction.confidence >= confidence_threshold:
                valid_predictions += 1
                if prediction.prediction == "buy":
                    buy_signals += 1
                elif prediction.prediction == "sell":
                    sell_signals += 1
                else:
                    hold_signals += 1
                total_confidence += prediction.confidence
        
        # Check if we have enough total models and enough agreeing models
        total_models = len(predictions)
        if total_models < min_models_required:
            logger.debug(f"Not enough models: {total_models} < {min_models_required}")
            return None
        
        avg_confidence = total_confidence / valid_predictions if valid_predictions > 0 else 0
        
        # Require majority agreement (at least 60% of valid predictions)
        required_agreement = max(3, int(valid_predictions * 0.6))
        
        if buy_signals >= required_agreement and buy_signals >= min_models_required:
            return {
                "side": "buy",
                "confidence": avg_confidence,
                "models_agreed": buy_signals,
                "total_models": total_models,
                "valid_models": valid_predictions
            }
        elif sell_signals >= required_agreement and sell_signals >= min_models_required:
            return {
                "side": "sell", 
                "confidence": avg_confidence,
                "models_agreed": sell_signals,
                "total_models": total_models,
                "valid_models": valid_predictions
            }
        
        return None
    
    async def _execute_trade(self, db: Session, symbol: str, signal: Dict, strategy: TradingStrategy):
        """Enhanced trade execution with detailed logging and error handling"""
        try:
            # Get current price
            klines = self.bybit_client.get_klines(symbol, limit=1)
            if not klines:
                logger.error(f"❌ No price data for {symbol}")
                return
            
            current_price = float(klines[-1]["close"])
            
            # Calculate TP/SL prices
            if signal["side"] == "buy":
                take_profit_price = current_price * (1 + strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 - strategy.stop_loss_percentage / 100)
            else:
                take_profit_price = current_price * (1 - strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 + strategy.stop_loss_percentage / 100)
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price, strategy)
            if position_size <= 0:
                logger.warning(f"⚠️ Invalid position size for {symbol}: {position_size}")
                return
            
            # Log trade attempt
            logger.info(f"🎯 TRADE SIGNAL: {symbol} {signal['side'].upper()}")
            logger.info(f"   💰 Price: ${current_price:.4f}")
            logger.info(f"   📊 Models: {signal.get('models_agreed', '?')}/{signal.get('total_models', '?')} ({signal['confidence']:.1f}%)")
            logger.info(f"   📏 Size: {position_size:.6f} ({strategy.leverage}x leverage)")
            logger.info(f"   🎯 TP: ${take_profit_price:.4f} | SL: ${stop_loss_price:.4f}")
            
            # Execute the order
            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=signal["side"],
                qty=position_size,
                leverage=strategy.leverage,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price
            )
            
            if order_result and order_result.get("success"):
                # Create trade record
                trade = Trade(
                    coin_symbol=symbol,
                    order_id=order_result.get("order_id", ""),
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
                        "leverage": strategy.leverage,
                        "models_agreed": signal.get("models_agreed", 0),
                        "total_models": signal.get("total_models", 0)
                    }
                )
                
                db.add(trade)
                db.commit()
                
                # Update trade counter
                self.trade_count_today += 1
                
                # Broadcast update
                await self.websocket_manager.broadcast_trade_update({
                    "action": "opened",
                    "symbol": symbol,
                    "side": signal["side"],
                    "price": current_price,
                    "confidence": signal["confidence"],
                    "models_agreed": signal.get("models_agreed", 0),
                    "total_models": signal.get("total_models", 0),
                    "order_id": order_result.get("order_id", "")
                })
                
                logger.info(f"✅ TRADE EXECUTED: {symbol} {signal['side'].upper()} | Order ID: {order_result.get('order_id', 'Unknown')}")
                
            else:
                error_msg = order_result.get("error", "Unknown error") if order_result else "No response from exchange"
                logger.error(f"❌ TRADE FAILED: {symbol} - {error_msg}")
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
    
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
                        # Use availableBalance for UTA trading (with fallback)
                        balance = coin_info.get("availableBalance") or coin_info.get("availableToWithdraw") or coin_info.get("walletBalance") or "0"
                        return float(balance)
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
    
    async def execute_signal_trade(self, coin_symbol: str, signal_side: str, confidence: float, strategy):
        """Execute a trade based on a signal (called by SignalExecutionEngine)"""
        try:
            # Get current price
            klines = self.bybit_client.get_klines(coin_symbol, limit=1)
            if not klines:
                logger.error(f"❌ No price data for {coin_symbol}")
                return False
            
            current_price = float(klines[-1]["close"])
            
            # Check if we already have an open position for this symbol
            current_positions = self.bybit_client.get_positions()
            if self._has_open_position(coin_symbol, current_positions):
                logger.info(f"⏭️ Skipping {coin_symbol} - already has open position")
                return False
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price, strategy)
            if position_size <= 0:
                logger.warning(f"⚠️ Invalid position size for {coin_symbol}: {position_size}")
                return False
            
            # Convert signal side to order side
            side = "buy" if signal_side == "LONG" else "sell"
            
            # Calculate TP/SL prices
            if side == "buy":
                take_profit_price = current_price * (1 + strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 - strategy.stop_loss_percentage / 100)
            else:
                take_profit_price = current_price * (1 - strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 + strategy.stop_loss_percentage / 100)
            
            logger.info(f"🎯 SIGNAL TRADE: {coin_symbol} {side.upper()}")
            logger.info(f"   💰 Price: ${current_price:.4f}")
            logger.info(f"   📊 Confidence: {confidence:.1f}%")
            logger.info(f"   📏 Size: {position_size:.6f} ({strategy.leverage}x leverage)")
            logger.info(f"   🎯 TP: ${take_profit_price:.4f} | SL: ${stop_loss_price:.4f}")
            
            # Execute the order
            order_result = self.bybit_client.place_order(
                symbol=coin_symbol,
                side=side,
                qty=position_size,
                leverage=strategy.leverage,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price
            )
            
            if order_result and order_result.get("success"):
                # Create trade record
                db = SessionLocal()
                trade = Trade(
                    coin_symbol=coin_symbol,
                    order_id=order_result.get("order_id", ""),
                    side=side,
                    size=position_size,
                    price=current_price,
                    leverage=strategy.leverage,
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    status="open",
                    ml_confidence=confidence,
                    strategy_params={
                        "signal_execution": True,
                        "take_profit_percentage": strategy.take_profit_percentage,
                        "stop_loss_percentage": strategy.stop_loss_percentage,
                        "leverage": strategy.leverage
                    }
                )
                
                db.add(trade)
                db.commit()
                db.close()
                
                # Broadcast update
                await self.websocket_manager.broadcast_trade_update({
                    "action": "signal_trade_opened",
                    "symbol": coin_symbol,
                    "side": side,
                    "price": current_price,
                    "confidence": confidence,
                    "order_id": order_result.get("order_id", "")
                })
                
                logger.info(f"✅ SIGNAL TRADE EXECUTED: {coin_symbol} {side.upper()} | Order ID: {order_result.get('order_id', 'Unknown')}")
                return True
            
            else:
                error_msg = order_result.get("error", "Unknown error") if order_result else "No response from exchange"
                logger.error(f"❌ SIGNAL TRADE FAILED: {coin_symbol} - {error_msg}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Error executing signal trade for {coin_symbol}: {e}")
            return False