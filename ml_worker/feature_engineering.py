import pandas as pd
import numpy as np
import ta
from typing import Dict, List

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, klines: List[Dict]) -> np.ndarray:
        df = pd.DataFrame(klines)
        
        if len(df) < 50:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        features = pd.DataFrame(index=df.index)
        
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'volatility_{period}'] = df['close'].rolling(period).std()
            features[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = (
            ta.volatility.BollingerBands(df['close']).bollinger_hband(),
            ta.volatility.BollingerBands(df['close']).bollinger_mavg(),
            ta.volatility.BollingerBands(df['close']).bollinger_lband()
        )
        
        features['macd'] = ta.trend.MACD(df['close']).macd()
        features['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        features['macd_histogram'] = ta.trend.MACD(df['close']).macd_diff()
        
        features['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        features['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        
        features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        features['volume_sma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma']
        
        features['price_change_1'] = df['close'].pct_change(1)
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        for lag in [1, 2, 3, 5]:
            features[f'close_lag_{lag}'] = df['close'].shift(lag)
            features[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        features = features.dropna()
        
        if len(features) == 0:
            return None
        
        self.feature_names = features.columns.tolist()
        
        return features.values
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names