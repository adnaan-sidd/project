# bollinger_bands.py
import pandas as pd
from indicators.indicators import TechnicalIndicators

class BollingerBandsStrategy:
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, df):
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = TechnicalIndicators.bollinger_bands(
            df['Close'], self.window, self.num_std
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['Close'] < df['BB_lower'], 'signal'] = 1  # Buy signal
        df.loc[df['Close'] > df['BB_upper'], 'signal'] = -1  # Sell signal
        
        # Calculate strategy returns
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        return df