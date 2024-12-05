import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        fast_ema = data.ewm(span=fast, adjust=False).mean()
        slow_ema = data.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high, low, close, period=14):
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].rolling(period).mean()

    @staticmethod
    def stochastic_oscillator(high, low, close, period=14):
        """Stochastic Oscillator (%K and %D lines)."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=3).mean()  # 3-period SMA of %K
        return k_line, d_line

    @staticmethod
    def vwap(high, low, close, volume):
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @staticmethod
    def parabolic_sar(high, low, acceleration=0.02, max_acceleration=0.2):
        """Parabolic SAR."""
        sar = pd.Series(index=high.index, dtype='float64')
        trend = pd.Series(index=high.index, dtype='int8')  # 1 for uptrend, -1 for downtrend
        
        # Initial SAR and trend direction
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        
        af = acceleration  # Initial acceleration factor
        ep = high.iloc[0]  # Initial extreme price (high)
        
        for i in range(1, len(high)):
            prev_sar = sar.iloc[i - 1]
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            
            if trend.iloc[i - 1] == 1:  # Uptrend
                if low.iloc[i] < sar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = -1
                    sar.iloc[i] = high.iloc[i]
                    ep = low.iloc[i]
                    af = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + acceleration, max_acceleration)
            else:  # Downtrend
                if high.iloc[i] > sar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = 1
                    sar.iloc[i] = low.iloc[i]
                    ep = high.iloc[i]
                    af = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + acceleration, max_acceleration)
        
        return sar

    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index (CCI)."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
        cci = (typical_price - sma) / (0.015 * mad)
        return cci