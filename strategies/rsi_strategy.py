# rsi_strategy.py
import pandas as pd
from indicators.indicators import TechnicalIndicators

class RSIStrategy:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def generate_signals(self, df):
        df = df.copy()
        
        # Calculate RSI
        df['RSI'] = TechnicalIndicators.rsi(df['Close'], self.period)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['RSI'] < self.oversold, 'signal'] = 1  # Buy signal
        df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell signal
        
        # Calculate strategy returns
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        return df