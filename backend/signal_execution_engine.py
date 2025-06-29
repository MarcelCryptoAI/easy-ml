import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import json

from database import SessionLocal, TradingSignal, Trade, TradingStrategy, MLPrediction, Coin
from bybit_client import BybitClient
from websocket_manager import WebSocketManager
from config import settings

logger = logging.getLogger(__name__)

class SignalExecutionEngine:
    def __init__(self, bybit_client: BybitClient, websocket_manager: WebSocketManager):
        self.bybit_client = bybit_client
        self.websocket_manager = websocket_manager
        self.enabled = True
        self.max_concurrent_signals = 50  # Maximum number of signals to process simultaneously
        self.min_balance_required = 50.0  # Minimum USDT balance for signal execution
        self.max_position_size_usdt = 500.0  # Maximum position size per signal
        self.signal_expiry_hours = 2  # Hours after which unexecuted signals expire
        
        # Risk management parameters
        self.max_daily_trades = 100
        self.max_daily_loss_usdt = 200.0
        self.daily_trades_count = 0
        self.daily_loss_usdt = 0.0
        self.last_reset_date = datetime.utcnow().date()
        
        logger.info("🚀 SignalExecutionEngine initialized")
    
    def reset_daily_limits(self):
        """Reset daily limits if it's a new day"""
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_loss_usdt = 0.0
            self.last_reset_date = current_date
            logger.info(f"📅 Daily limits reset for {current_date}")
    
    def check_risk_limits(self) -> bool:
        """Check if signal execution should continue based on risk limits"""
        self.reset_daily_limits()
        
        # Check daily trade limit
        if self.daily_trades_count >= self.max_daily_trades:
            logger.warning(f"🛑 Daily trade limit reached: {self.daily_trades_count}")
            return False
        
        # Check daily loss limit
        if self.daily_loss_usdt >= self.max_daily_loss_usdt:
            logger.warning(f"🛑 Daily loss limit reached: ${self.daily_loss_usdt:.2f}")
            return False
        
        # Check account balance
        balance = self._get_available_balance()
        if balance < self.min_balance_required:
            logger.warning(f"🛑 Insufficient balance: ${balance:.2f} (min: ${self.min_balance_required:.2f})")
            return False
        
        return True
    
    def _get_available_balance(self) -> float:
        """Get available USDT balance from Bybit"""
        try:
            response = self.bybit_client.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if response["retCode"] == 0:
                account_list = response["result"]["list"][0]
                for coin_info in account_list["coin"]:
                    if coin_info["coin"] == "USDT":
                        balance = coin_info.get("availableBalance") or coin_info.get("availableToWithdraw") or "0"
                        return float(balance)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    async def generate_and_store_signals(self, db: Session) -> List[TradingSignal]:
        """Generate trading signals from ML predictions and store them in database"""
        try:
            # Get all active coins
            active_coins = db.query(Coin).filter(Coin.is_active == True).all()
            new_signals = []
            
            for coin in active_coins:
                try:
                    # Get latest predictions for this coin (last 2 hours)
                    predictions = db.query(MLPrediction).filter(
                        MLPrediction.coin_symbol == coin.symbol,
                        MLPrediction.created_at >= datetime.utcnow() - timedelta(hours=2)
                    ).order_by(MLPrediction.created_at.desc()).limit(10).all()
                    
                    if len(predictions) < 2:  # Need at least 2 model predictions
                        continue
                    
                    # Group by latest prediction per model type
                    latest_by_model = {}
                    for pred in predictions:
                        if pred.model_type not in latest_by_model:
                            latest_by_model[pred.model_type] = pred
                    
                    # Apply weighted consensus algorithm
                    model_weights = {
                        'transformer': 1.25,
                        'lstm': 1.20,
                        'xgboost': 1.15,
                        'lightgbm': 1.10,
                        'catboost': 1.05,
                        'random_forest': 1.00,
                        'neural_network': 0.95,
                        'svm': 0.90,
                        'gru': 1.10,
                        'cnn_1d': 0.85
                    }
                    
                    long_score = 0
                    short_score = 0
                    total_weight = 0
                    buy_count = 0
                    sell_count = 0
                    total_confidence = 0
                    
                    for model_type, pred in latest_by_model.items():
                        weight = model_weights.get(pred.model_type.lower(), 1.0)
                        total_weight += weight
                        total_confidence += pred.confidence
                        
                        if pred.prediction.upper() in ["BUY", "LONG"]:
                            long_score += weight * (pred.confidence / 100)
                            buy_count += 1
                        elif pred.prediction.upper() in ["SELL", "SHORT"]:
                            short_score += weight * (pred.confidence / 100)
                            sell_count += 1
                    
                    if total_weight == 0:
                        continue
                    
                    # Normalize scores
                    long_confidence = long_score / total_weight
                    short_confidence = short_score / total_weight
                    avg_confidence = total_confidence / len(latest_by_model)
                    
                    # Determine signal strength
                    max_confidence = max(long_confidence, short_confidence)
                    decision_margin = abs(long_confidence - short_confidence)
                    
                    # Signal criteria: confidence >= 25% AND margin >= 0.05 (VERY LOW THRESHOLDS)
                    signal_threshold = 0.25
                    margin_threshold = 0.05
                    
                    if max_confidence >= signal_threshold and decision_margin >= margin_threshold:
                        signal_type = 'LONG' if long_confidence > short_confidence else 'SHORT'
                        models_agreed = buy_count if signal_type == 'LONG' else sell_count
                        
                        # Check if we already have a pending/executing signal for this coin
                        existing_signal = db.query(TradingSignal).filter(
                            TradingSignal.coin_symbol == coin.symbol,
                            TradingSignal.status.in_(['pending', 'executing']),
                            TradingSignal.created_at >= datetime.utcnow() - timedelta(hours=1)
                        ).first()
                        
                        if existing_signal:
                            continue  # Skip if we already have a recent signal
                        
                        # Get current price
                        current_price = 0.0
                        try:
                            klines = self.bybit_client.get_klines(coin.symbol, interval="1", limit=1)
                            if klines:
                                current_price = float(klines[-1]["close"])
                        except Exception as price_error:
                            logger.error(f"Failed to get price for {coin.symbol}: {price_error}")
                            continue
                        
                        # Generate unique signal ID
                        signal_id = f"{coin.symbol}_{int(datetime.utcnow().timestamp())}"
                        
                        # Create signal record
                        signal = TradingSignal(
                            signal_id=signal_id,
                            coin_symbol=coin.symbol,
                            signal_type=signal_type,
                            confidence=max_confidence * 100,  # Convert to percentage
                            models_agreed=models_agreed,
                            total_models=len(latest_by_model),
                            entry_price=current_price,
                            current_price=current_price,
                            position_size_usdt=min(100.0, self.max_position_size_usdt),  # Default position size
                            status='pending'
                        )
                        
                        db.add(signal)
                        new_signals.append(signal)
                        
                        logger.info(f"🎯 NEW SIGNAL: {coin.symbol} - {signal_type} "
                                  f"(Confidence: {max_confidence:.1%}, Models: {models_agreed}/{len(latest_by_model)})")
                
                except Exception as coin_error:
                    logger.error(f"Error processing signals for {coin.symbol}: {coin_error}")
                    continue
            
            if new_signals:
                db.commit()
                logger.info(f"📊 Generated {len(new_signals)} new trading signals")
            
            return new_signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def execute_signal(self, db: Session, signal: TradingSignal) -> bool:
        """Execute a trading signal by placing a real order"""
        try:
            # Update signal status to executing
            signal.status = 'executing'
            db.commit()
            
            logger.info(f"🔄 EXECUTING SIGNAL: {signal.signal_id} - {signal.coin_symbol} {signal.signal_type}")
            
            # Get strategy for this coin
            strategy = db.query(TradingStrategy).filter(
                TradingStrategy.coin_symbol == signal.coin_symbol,
                TradingStrategy.is_active == True
            ).first()
            
            if not strategy:
                # Create default strategy
                strategy = TradingStrategy(
                    coin_symbol=signal.coin_symbol,
                    take_profit_percentage=2.0,
                    stop_loss_percentage=1.0,
                    leverage=10,
                    position_size_percentage=2.0,
                    is_active=True
                )
                db.add(strategy)
                db.commit()
            
            # Get current price
            klines = self.bybit_client.get_klines(signal.coin_symbol, limit=1)
            if not klines:
                raise Exception(f"Unable to get current price for {signal.coin_symbol}")
            
            current_price = float(klines[-1]["close"])
            signal.current_price = current_price
            
            # Calculate position size based on available balance
            available_balance = self._get_available_balance()
            position_value = min(signal.position_size_usdt, available_balance * 0.02)  # Max 2% of balance per trade
            position_size = position_value / current_price
            
            if position_size < 0.001:  # Minimum position size
                raise Exception(f"Position size too small: {position_size}")
            
            # Calculate TP/SL prices
            if signal.signal_type == 'LONG':
                side = 'Buy'
                take_profit_price = current_price * (1 + strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 - strategy.stop_loss_percentage / 100)
            else:
                side = 'Sell'
                take_profit_price = current_price * (1 - strategy.take_profit_percentage / 100)
                stop_loss_price = current_price * (1 + strategy.stop_loss_percentage / 100)
            
            # Store TP/SL in signal
            signal.take_profit_price = take_profit_price
            signal.stop_loss_price = stop_loss_price
            signal.leverage = strategy.leverage
            
            logger.info(f"📈 TRADE DETAILS: {signal.coin_symbol}")
            logger.info(f"   💰 Entry: ${current_price:.4f}")
            logger.info(f"   📏 Size: {position_size:.6f} (${position_value:.2f})")
            logger.info(f"   🎯 TP: ${take_profit_price:.4f} | SL: ${stop_loss_price:.4f}")
            logger.info(f"   ⚡ Leverage: {strategy.leverage}x")
            
            # Execute the order
            order_result = self.bybit_client.place_order(
                symbol=signal.coin_symbol,
                side=side,
                qty=position_size,
                leverage=strategy.leverage,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                order_type="market"
            )
            
            if order_result and order_result.get("success"):
                # Create trade record
                trade = Trade(
                    coin_symbol=signal.coin_symbol,
                    order_id=order_result.get("order_id", ""),
                    side=signal.signal_type.lower(),
                    size=position_size,
                    price=current_price,
                    leverage=strategy.leverage,
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    status="open",
                    ml_confidence=signal.confidence,
                    strategy_params={
                        "signal_id": signal.signal_id,
                        "models_agreed": signal.models_agreed,
                        "total_models": signal.total_models,
                        "execution_engine": "SignalExecutionEngine"
                    }
                )
                
                db.add(trade)
                db.commit()
                
                # Update signal status
                signal.status = 'executed'
                signal.executed_at = datetime.utcnow()
                signal.trade_id = trade.id
                signal.execution_order_id = order_result.get("order_id", "")
                signal.position_size_usdt = position_value
                db.commit()
                
                # Update daily counters
                self.daily_trades_count += 1
                
                # Broadcast update
                await self.websocket_manager.broadcast_trade_update({
                    "action": "signal_executed",
                    "signal_id": signal.signal_id,
                    "symbol": signal.coin_symbol,
                    "side": signal.signal_type,
                    "price": current_price,
                    "confidence": signal.confidence,
                    "order_id": order_result.get("order_id", ""),
                    "trade_id": trade.id
                })
                
                logger.info(f"✅ SIGNAL EXECUTED: {signal.signal_id} - Order ID: {order_result.get('order_id', 'Unknown')}")
                return True
            
            else:
                # Execution failed
                error_msg = order_result.get("error", "Unknown error") if order_result else "No response from exchange"
                signal.status = 'failed'
                signal.execution_error = error_msg
                db.commit()
                
                logger.error(f"❌ SIGNAL EXECUTION FAILED: {signal.signal_id} - {error_msg}")
                return False
        
        except Exception as e:
            # Execution failed
            signal.status = 'failed'
            signal.execution_error = str(e)
            db.commit()
            
            logger.error(f"❌ SIGNAL EXECUTION ERROR: {signal.signal_id} - {e}")
            return False
    
    async def update_signal_pnl(self, db: Session, signal: TradingSignal):
        """Update P&L for an executed signal"""
        try:
            if signal.status != 'executed' or not signal.trade_id:
                return
            
            # Get current price
            klines = self.bybit_client.get_klines(signal.coin_symbol, limit=1)
            if not klines:
                return
            
            current_price = float(klines[-1]["close"])
            signal.current_price = current_price
            
            # Calculate unrealized P&L
            entry_price = signal.entry_price
            position_size_usdt = signal.position_size_usdt
            
            if signal.signal_type == 'LONG':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Apply leverage
            pnl_percent *= signal.leverage
            pnl_usdt = (position_size_usdt * pnl_percent) / 100
            
            signal.unrealized_pnl_percent = pnl_percent
            signal.unrealized_pnl_usdt = pnl_usdt
            
            # Check if position is closed
            trade = db.query(Trade).filter(Trade.id == signal.trade_id).first()
            if trade and trade.status == 'closed':
                signal.status = 'closed'
                signal.closed_at = datetime.utcnow()
                signal.realized_pnl_usdt = trade.pnl
                signal.realized_pnl_percent = pnl_percent
                
                # Update daily loss tracker
                if trade.pnl < 0:
                    self.daily_loss_usdt += abs(trade.pnl)
            
            db.commit()
        
        except Exception as e:
            logger.error(f"Error updating P&L for signal {signal.signal_id}: {e}")
    
    async def cleanup_expired_signals(self, db: Session):
        """Remove expired signals that haven't been executed"""
        try:
            expiry_time = datetime.utcnow() - timedelta(hours=self.signal_expiry_hours)
            
            expired_signals = db.query(TradingSignal).filter(
                TradingSignal.status == 'pending',
                TradingSignal.created_at < expiry_time
            ).all()
            
            for signal in expired_signals:
                signal.status = 'expired'
                signal.execution_error = f"Signal expired after {self.signal_expiry_hours} hours"
            
            if expired_signals:
                db.commit()
                logger.info(f"🗑️ Cleaned up {len(expired_signals)} expired signals")
        
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")
    
    async def process_signals_continuously(self):
        """Main loop for continuous signal processing"""
        logger.info("🔄 Starting continuous signal processing...")
        
        while True:
            try:
                if not self.enabled:
                    await asyncio.sleep(10)
                    continue
                
                if not self.check_risk_limits():
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                db = SessionLocal()
                
                # Generate new signals
                await self.generate_and_store_signals(db)
                
                # Execute pending signals
                pending_signals = db.query(TradingSignal).filter(
                    TradingSignal.status == 'pending'
                ).order_by(TradingSignal.confidence.desc()).limit(self.max_concurrent_signals).all()
                
                executed_count = 0
                for signal in pending_signals:
                    if await self.execute_signal(db, signal):
                        executed_count += 1
                    
                    # Small delay between executions
                    await asyncio.sleep(0.5)
                
                # Update P&L for executed signals
                executed_signals = db.query(TradingSignal).filter(
                    TradingSignal.status.in_(['executed'])
                ).all()
                
                for signal in executed_signals:
                    await self.update_signal_pnl(db, signal)
                
                # Cleanup expired signals
                await self.cleanup_expired_signals(db)
                
                db.close()
                
                if executed_count > 0:
                    logger.info(f"🎯 Processed {len(pending_signals)} signals, executed {executed_count} trades")
                
            except Exception as e:
                logger.error(f"Error in signal processing loop: {e}")
            
            # Wait before next cycle
            await asyncio.sleep(30)