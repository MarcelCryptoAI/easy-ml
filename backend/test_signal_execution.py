#!/usr/bin/env python3
"""
Test script for the automatic signal execution system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from database import SessionLocal, create_tables, TradingSignal, Coin, MLPrediction, TradingStrategy
from signal_execution_engine import SignalExecutionEngine
from bybit_client import BybitClient
from websocket_manager import WebSocketManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_signal_execution():
    """Test the signal execution system"""
    try:
        logger.info("🧪 Testing Automatic Signal Execution System")
        
        # Initialize components
        bybit_client = BybitClient()
        websocket_manager = WebSocketManager()
        signal_engine = SignalExecutionEngine(bybit_client, websocket_manager)
        
        # Test 1: Check database connection
        logger.info("📊 Test 1: Database Connection")
        db = SessionLocal()
        coin_count = db.query(Coin).count()
        logger.info(f"   ✅ Connected to database with {coin_count} coins")
        
        # Test 2: Check ByBit connection
        logger.info("🔗 Test 2: ByBit API Connection")
        balance = signal_engine._get_available_balance()
        logger.info(f"   ✅ ByBit connected, balance: ${balance:.2f} USDT")
        
        # Test 3: Check if we have ML predictions
        logger.info("🤖 Test 3: ML Predictions Available")
        recent_predictions = db.query(MLPrediction).filter(
            MLPrediction.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        logger.info(f"   ✅ Found {recent_predictions} ML predictions in last 24h")
        
        # Test 4: Generate test signals
        logger.info("🎯 Test 4: Signal Generation")
        signals = await signal_engine.generate_and_store_signals(db)
        logger.info(f"   ✅ Generated {len(signals)} new signals")
        
        # Test 5: Check signal status
        logger.info("📋 Test 5: Signal Status Check")
        all_signals = db.query(TradingSignal).all()
        status_breakdown = {}
        for signal in all_signals:
            status_breakdown[signal.status] = status_breakdown.get(signal.status, 0) + 1
        
        logger.info(f"   📊 Signal status breakdown: {status_breakdown}")
        
        # Test 6: Risk management check
        logger.info("🛡️ Test 6: Risk Management")
        can_trade = signal_engine.check_risk_limits()
        logger.info(f"   ✅ Risk limits check: {'PASS' if can_trade else 'FAIL'}")
        logger.info(f"   📈 Daily trades: {signal_engine.daily_trades_count}/{signal_engine.max_daily_trades}")
        logger.info(f"   💸 Daily loss: ${signal_engine.daily_loss_usdt:.2f}/${signal_engine.max_daily_loss_usdt:.2f}")
        
        # Test 7: Check pending signals
        logger.info("⏳ Test 7: Pending Signals")
        pending_signals = db.query(TradingSignal).filter(TradingSignal.status == 'pending').all()
        logger.info(f"   📋 Found {len(pending_signals)} pending signals")
        
        for signal in pending_signals[:3]:  # Show first 3
            logger.info(f"      • {signal.coin_symbol} {signal.signal_type} (confidence: {signal.confidence:.1f}%)")
        
        # Test 8: Simulate signal execution (DRY RUN)
        logger.info("🔄 Test 8: Signal Execution Simulation")
        if pending_signals and balance > 50:
            test_signal = pending_signals[0]
            logger.info(f"   🎯 Would execute: {test_signal.coin_symbol} {test_signal.signal_type}")
            logger.info(f"   💰 Entry price: ${test_signal.entry_price:.4f}")
            logger.info(f"   📊 Confidence: {test_signal.confidence:.1f}%")
            logger.info("   ⚠️  DRY RUN - Not executing real trade")
        else:
            logger.info("   ⚠️  No signals to execute or insufficient balance")
        
        # Test 9: Check strategies
        logger.info("⚙️ Test 9: Trading Strategies")
        strategies = db.query(TradingStrategy).filter(TradingStrategy.is_active == True).count()
        logger.info(f"   ✅ Found {strategies} active trading strategies")
        
        # Test 10: Engine status
        logger.info("🚀 Test 10: Engine Status")
        logger.info(f"   🔧 Signal engine enabled: {signal_engine.enabled}")
        logger.info(f"   📊 Max concurrent signals: {signal_engine.max_concurrent_signals}")
        logger.info(f"   💼 Min balance required: ${signal_engine.min_balance_required:.2f}")
        logger.info(f"   📈 Max position size: ${signal_engine.max_position_size_usdt:.2f}")
        
        db.close()
        
        logger.info("✅ All tests completed successfully!")
        logger.info("🚀 Automatic Signal Execution System is ready!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def run_continuous_test(duration_minutes=5):
    """Run the signal execution engine for a few minutes to test continuous operation"""
    logger.info(f"🔄 Running continuous test for {duration_minutes} minutes...")
    
    bybit_client = BybitClient()
    websocket_manager = WebSocketManager()
    signal_engine = SignalExecutionEngine(bybit_client, websocket_manager)
    
    # Disable actual execution for testing
    signal_engine.enabled = False
    logger.info("⚠️ Signal execution disabled for testing")
    
    # Run for specified duration
    end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
    cycle_count = 0
    
    while datetime.utcnow() < end_time:
        cycle_count += 1
        logger.info(f"🔄 Test cycle {cycle_count}")
        
        try:
            db = SessionLocal()
            
            # Generate signals
            signals = await signal_engine.generate_and_store_signals(db)
            logger.info(f"   📊 Generated {len(signals)} signals")
            
            # Check pending signals
            pending = db.query(TradingSignal).filter(TradingSignal.status == 'pending').count()
            logger.info(f"   ⏳ Pending signals: {pending}")
            
            # Cleanup expired signals
            await signal_engine.cleanup_expired_signals(db)
            
            db.close()
            
        except Exception as e:
            logger.error(f"   ❌ Cycle error: {e}")
        
        # Wait 30 seconds before next cycle
        await asyncio.sleep(30)
    
    logger.info(f"✅ Continuous test completed after {cycle_count} cycles")

if __name__ == "__main__":
    print("🧪 Automatic Signal Execution Test Suite")
    print("=" * 50)
    
    # Run basic tests
    asyncio.run(test_signal_execution())
    
    print("\n" + "=" * 50)
    response = input("Run continuous test? (y/N): ")
    
    if response.lower() == 'y':
        duration = input("Test duration in minutes (default 2): ")
        try:
            duration = int(duration) if duration else 2
        except:
            duration = 2
        
        asyncio.run(run_continuous_test(duration))
    
    print("🎉 Testing complete!")