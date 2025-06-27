import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from bybit_client import BybitClient
from config import settings

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        self.bybit_client = BybitClient()
    
    def run_backtest(self, symbol: str, strategy_params: Dict, period_months: int = 6) -> Dict:
        """
        Run backtest for a specific symbol and strategy over historical data
        """
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_months * 30)
            
            # Get historical klines (you might need to implement this in BybitClient)
            historical_data = self._get_historical_data(symbol, start_date, end_date)
            
            if not historical_data:
                return {"error": "No historical data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Simulate ML predictions (replace with actual historical predictions if available)
            df = self._add_simulated_ml_signals(df, strategy_params)
            
            # Run backtest simulation
            results = self._simulate_trading(df, strategy_params)
            
            return {
                "success": True,
                "results": results,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "data_points": len(df)
            }
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return {"error": str(e)}
    
    def _get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Get historical price data from Bybit
        This is a simplified version - you might need to implement pagination for large datasets
        """
        try:
            # For now, get recent data (Bybit API limits historical data)
            klines = self.bybit_client.get_klines(symbol, interval="1", limit=1000)
            
            if not klines:
                return []
            
            # Filter by date range if possible
            filtered_klines = []
            for kline in klines:
                kline_date = datetime.fromtimestamp(kline['timestamp'] / 1000)
                if start_date <= kline_date <= end_date:
                    filtered_klines.append(kline)
            
            return filtered_klines or klines  # Return all if no date filtering worked
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def _add_simulated_ml_signals(self, df: pd.DataFrame, strategy_params: Dict) -> pd.DataFrame:
        """
        Add simulated ML signals based on technical indicators
        In production, this would use actual historical ML predictions
        """
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Generate simulated ML signals
        signals = []
        confidences = []
        
        for i in range(len(df)):
            # Simulate ML model predictions based on technical indicators
            signal = "hold"
            confidence = 50.0
            
            if i >= 50:  # Need enough data for indicators
                price = df.iloc[i]['close']
                sma_20 = df.iloc[i]['sma_20']
                sma_50 = df.iloc[i]['sma_50']
                rsi = df.iloc[i]['rsi']
                
                # Simple strategy logic for simulation
                if price > sma_20 > sma_50 and rsi < 70:
                    signal = "buy"
                    confidence = min(85, 60 + (price - sma_20) / price * 100)
                elif price < sma_20 < sma_50 and rsi > 30:
                    signal = "sell"
                    confidence = min(85, 60 + (sma_20 - price) / price * 100)
                else:
                    confidence = np.random.uniform(30, 60)
            
            signals.append(signal)
            confidences.append(confidence)
        
        df['ml_signal'] = signals
        df['ml_confidence'] = confidences
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _simulate_trading(self, df: pd.DataFrame, strategy_params: Dict) -> Dict:
        """
        Simulate trading based on ML signals and strategy parameters
        """
        initial_balance = 10000  # $10,000 starting balance
        balance = initial_balance
        position = None
        trades = []
        equity_curve = []
        
        take_profit = strategy_params.get('take_profit_percentage', 2.0) / 100
        stop_loss = strategy_params.get('stop_loss_percentage', 1.0) / 100
        confidence_threshold = strategy_params.get('confidence_threshold', 70.0)
        position_size = strategy_params.get('position_size_percentage', 5.0) / 100
        leverage = strategy_params.get('leverage', 10)
        
        for i, row in df.iterrows():
            current_price = row['close']
            ml_signal = row['ml_signal']
            ml_confidence = row['ml_confidence']
            
            # Check if we should close existing position
            if position:
                entry_price = position['entry_price']
                side = position['side']
                
                # Calculate P&L
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Apply leverage
                pnl_pct *= leverage
                
                # Check stop loss or take profit
                should_close = False
                if pnl_pct <= -stop_loss:  # Stop loss
                    should_close = True
                    exit_reason = "stop_loss"
                elif pnl_pct >= take_profit:  # Take profit
                    should_close = True
                    exit_reason = "take_profit"
                
                if should_close:
                    # Close position
                    trade_value = balance * position_size
                    profit = trade_value * pnl_pct
                    balance += profit
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': i,
                        'side': side,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'profit': profit,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Check if we should open new position
            if not position and ml_confidence >= confidence_threshold:
                if ml_signal in ['buy', 'sell']:
                    position = {
                        'side': ml_signal,
                        'entry_price': current_price,
                        'entry_time': i,
                        'confidence': ml_confidence
                    }
            
            # Track equity curve
            current_equity = balance
            if position:
                entry_price = position['entry_price']
                side = position['side']
                if side == 'buy':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price
                
                unrealized_pnl *= leverage * balance * position_size
                current_equity += unrealized_pnl
            
            equity_curve.append({
                'timestamp': i,
                'equity': current_equity,
                'price': current_price
            })
        
        # Calculate performance metrics
        if trades:
            returns = [t['pnl_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            
            total_return = (balance - initial_balance) / initial_balance * 100
            win_rate = len(winning_trades) / len(trades) * 100
            
            # Calculate max drawdown
            equity_values = [e['equity'] for e in equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Monthly returns for chart
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            return {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "avg_trade_duration": np.mean([1 for _ in trades]),  # Simplified
                "profit_factor": sum([t['profit'] for t in winning_trades]) / abs(sum([t['profit'] for t in trades if t['profit'] < 0])) if any(t['profit'] < 0 for t in trades) else float('inf'),
                "monthly_returns": monthly_returns,
                "trade_distribution": [
                    {"type": "Winning", "count": len(winning_trades)},
                    {"type": "Losing", "count": len(trades) - len(winning_trades)}
                ],
                "equity_curve": equity_curve[-100:],  # Last 100 points for chart
                "trades": trades[-20:]  # Last 20 trades
            }
        else:
            return {
                "total_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "avg_trade_duration": 0,
                "profit_factor": 0,
                "monthly_returns": [],
                "trade_distribution": [],
                "equity_curve": equity_curve[-100:],
                "trades": []
            }
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict]) -> List[Dict]:
        """Calculate monthly returns from equity curve"""
        if not equity_curve:
            return []
        
        # Simplified monthly calculation
        monthly_data = []
        for i in range(0, len(equity_curve), 30):  # Rough monthly grouping
            if i + 30 < len(equity_curve):
                start_equity = equity_curve[i]['equity']
                end_equity = equity_curve[i + 30]['equity']
                monthly_return = (end_equity - start_equity) / start_equity * 100
                monthly_data.append({
                    "month": f"Month {i//30 + 1}",
                    "return": monthly_return
                })
        
        return monthly_data