import pandas as pd
from indicators.indicators import TechnicalIndicators

class ATRStrategy:
    @staticmethod
    def trailing_stop_loss(data, atr_multiplier=2, period=14):
        """
        Sets trailing stop-loss levels based on ATR.
        Args:
            data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            atr_multiplier (int): Multiplier for ATR.
            period (int): ATR calculation period.
        Returns:
            pd.Series: Trailing stop-loss levels.
        """
        from indicators import TechnicalIndicators  # Ensure indicators.py is in the path
        
        atr = TechnicalIndicators.atr(data['high'], data['low'], data['close'], period)
        trailing_stop_loss = data['close'] - (atr * atr_multiplier)
        return trailing_stop_loss