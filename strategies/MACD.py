import pandas as pd
from indicators.indicators import TechnicalIndicators

class MACDStrategy:
    @staticmethod
    def generate_signals(data, fast=12, slow=26, signal=9):
        """
        Generates buy/sell signals based on MACD line and signal line crossovers.
        Args:
            data (pd.Series): Price data (typically 'close').
            fast (int): Fast EMA period.
            slow (int): Slow EMA period.
            signal (int): Signal line EMA period.
        Returns:
            pd.Series: Signals where 1 = Buy, -1 = Sell, and 0 = Hold.
        """
        from indicators import TechnicalIndicators  # Ensure indicators.py is in the path
        
        macd_line, signal_line, _ = TechnicalIndicators.macd(data, fast, slow, signal)
        signals = pd.Series(index=data.index, data=0)
        
        # Buy signal when MACD crosses above the signal line
        signals[macd_line > signal_line] = 1
        
        # Sell signal when MACD crosses below the signal line
        signals[macd_line < signal_line] = -1
        
        return signals