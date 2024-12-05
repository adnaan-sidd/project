# moving_average.py
import pandas as pd
from indicators.indicators import TechnicalIndicators

class MovingAverageStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, df):
        df = df.copy()
        
        # Calculate moving averages
        df['SMA_short'] = TechnicalIndicators.sma(df['Close'], self.short_window)
        df['SMA_long'] = TechnicalIndicators.sma(df['Close'], self.long_window)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1  # Buy signal
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1  # Sell signal
        
        # Calculate strategy returns
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        return df