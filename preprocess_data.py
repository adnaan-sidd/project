import os
import pickle
import logging
import ast
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- Add this line

# Constants
CLOSE_COLUMN = 'Close'
TIMESTAMP_COLUMN = 'timestamp'
SENTIMENT_SCORES_COLUMN = 'sentiment_scores'

# Custom Exceptions
class DataLoaderError(Exception):
    pass

class ConfigError(Exception):
    pass

# Setup logging with both file and console handlers
def setup_logging() -> logging.Logger:
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    def __init__(self, config_path: str = 'config/config.yaml'):
        logger.info(f"Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            raise ConfigError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.validate_config()
        self.setup_directories()
        self.load_config_attributes()
        logger.info("Configuration loaded successfully")

    def validate_config(self) -> None:
        required_fields = ['assets', 'model_params', 'preprocessing', 'system']
        for field in required_fields:
            if field not in self.config:
                raise ConfigError(f"Missing required configuration field: {field}")

    def setup_directories(self) -> None:
        os.makedirs(self.config['system']['data_directory'], exist_ok=True)
        os.makedirs(self.config['system']['log_directory'], exist_ok=True)

    def load_config_attributes(self) -> None:
        self.assets = self.config['assets']
        self.sequence_length = self.config['model_params']['sequence_length']
        self.feature_columns = self.config['preprocessing']['feature_columns']

class DataLoader:
    def __init__(self, symbol: str, config: Config):
        self.symbol = symbol
        self.config = config
        logger.info(f"Initializing DataLoader for {symbol}")

    def load_and_merge_data(self, sentiment_file: str) -> pd.DataFrame:
        try:
            data_dir = self.config.config['system']['data_directory']
            yfinance_file = self.get_yfinance_file(data_dir)
            logger.info(f"Loading yfinance data from: {yfinance_file}")
            yfinance_data = self.load_yfinance_data(yfinance_file)
            candles_data = self.load_candles_data()
            market_data = pd.concat([yfinance_data, candles_data]).drop_duplicates().sort_index()

            logger.info(f"Loading sentiment data from: {sentiment_file}")
            sentiment_data = self.load_sentiment_data(sentiment_file)
            merged_data = pd.merge(market_data, sentiment_data, left_index=True, right_index=True, how='outer')

            logger.info(f"Successfully merged data for {self.symbol}. Shape: {merged_data.shape}")
            return merged_data

        except Exception as e:
            logger.error(f"Error while loading and merging data: {e}")
            raise DataLoaderError(f"Error while loading and merging data: {e}")

    def get_yfinance_file(self, data_dir: str) -> str:
        if self.symbol == "XAUUSD":
            return os.path.join(data_dir, 'yfinance', 'gold_yf.csv')
        return os.path.join(data_dir, 'yfinance', f'{self.symbol}_yf.csv')

    def load_yfinance_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YFinance data file not found: {file_path}")

        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()].copy()  # Avoid SettingWithCopyWarning
        df.index = df.index.tz_localize(None)  # Convert to tz-naive
        df.rename(columns=str.capitalize, inplace=True)
        return df

    def load_candles_data(self) -> pd.DataFrame:
        candle_timeframes = ['D1', 'H1', 'H4', 'M15']
        data_dir = self.config.config['system']['data_directory']
        candles_data = []

        for timeframe in candle_timeframes:
            candle_file = os.path.join(data_dir, 'candles', f'{self.symbol}_{timeframe}.csv')
            if os.path.exists(candle_file):
                logger.info(f"Loading candles data from: {candle_file}")
                df = self.load_yfinance_data(candle_file)
                candles_data.append(df)

        return pd.concat(candles_data) if candles_data else pd.DataFrame()

    def load_sentiment_data(self, sentiment_file: str) -> pd.DataFrame:
        if not os.path.exists(sentiment_file):
            raise FileNotFoundError(f"Sentiment data file not found: {sentiment_file}")

        sentiment_data = pd.read_csv(sentiment_file)
        sentiment_data['Date'] = pd.to_datetime(sentiment_data[TIMESTAMP_COLUMN], errors='coerce')
        sentiment_data = sentiment_data[sentiment_data['Date'].notnull()].copy()  # Avoid SettingWithCopyWarning
        sentiment_data.set_index('Date', inplace=True)
        sentiment_data.index = sentiment_data.index.tz_localize(None)  # Convert to tz-naive
        sentiment_data = self.process_sentiment_scores(sentiment_data)
        return sentiment_data

    def process_sentiment_scores(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        sentiment_data[SENTIMENT_SCORES_COLUMN] = sentiment_data[SENTIMENT_SCORES_COLUMN].apply(ast.literal_eval)
        sentiment_data['positive'] = sentiment_data[SENTIMENT_SCORES_COLUMN].apply(lambda x: x['score'] if x['label'] == 'POSITIVE' else 0)
        sentiment_data['negative'] = sentiment_data[SENTIMENT_SCORES_COLUMN].apply(lambda x: x['score'] if x['label'] == 'NEGATIVE' else 0)
        sentiment_data['neutral'] = sentiment_data[SENTIMENT_SCORES_COLUMN].apply(lambda x: x['score'] if x['label'] == 'NEUTRAL' else 0)
        sentiment_data.drop(columns=[SENTIMENT_SCORES_COLUMN], inplace=True)
        return sentiment_data

class FeatureEngine:
    def __init__(self, config: Config):
        self.config = config

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.validate_dataframe(df)

            logger.info("Calculating technical indicators")
            df['Returns'] = df[CLOSE_COLUMN].pct_change(fill_method=None)

            self.calculate_moving_averages(df)
            self.calculate_rsi(df)
            self.calculate_bollinger_bands(df)
            self.calculate_macd(df)
            self.calculate_atr(df)

            df['Volatility'] = df[CLOSE_COLUMN].rolling(window=20).std()
            df['Momentum'] = df[CLOSE_COLUMN] - df[CLOSE_COLUMN].shift(4)
            df['ROC'] = (df[CLOSE_COLUMN] - df[CLOSE_COLUMN].shift(12)) / df[CLOSE_COLUMN].shift(12) * 100

            # Calculate MA Crossover
            short_window = self.config.config['strategies']['moving_average_crossover']['short_window']
            long_window = self.config.config['strategies']['moving_average_crossover']['long_window']
            df[f'SMA_{short_window}'] = df[CLOSE_COLUMN].rolling(window=short_window).mean()
            df[f'SMA_{long_window}'] = df[CLOSE_COLUMN].rolling(window=long_window).mean()
            df['ma_crossover'] = np.where(df[f'SMA_{short_window}'] > df[f'SMA_{long_window}'], 1, 0)
            logger.info(f"MA Crossover column created with values: {df['ma_crossover'].head()}")

            logger.info("Technical indicators calculated successfully")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        if CLOSE_COLUMN not in df.columns:
            raise ValueError("DataFrame does not contain 'Close' column")

    def calculate_moving_averages(self, df: pd.DataFrame) -> None:
        short_window = self.config.config['strategies']['moving_average_crossover']['short_window']
        long_window = self.config.config['strategies']['moving_average_crossover']['long_window']
        df[f'SMA_{short_window}'] = df[CLOSE_COLUMN].rolling(window=short_window).mean()
        df[f'SMA_{long_window}'] = df[CLOSE_COLUMN].rolling(window=long_window).mean()

    def calculate_rsi(self, df: pd.DataFrame) -> None:
        rsi_period = self.config.config['strategies']['rsi']['period']
        delta = df[CLOSE_COLUMN].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> None:
        bb_window = self.config.config['strategies']['bollinger_bands']['window']
        bb_std = self.config.config['strategies']['bollinger_bands']['num_std']
        df['BB_middle'] = df[CLOSE_COLUMN].rolling(window=bb_window).mean()
        df['BB_upper'] = df['BB_middle'] + bb_std * df[CLOSE_COLUMN].rolling(window=bb_window).std()
        df['BB_lower'] = df['BB_middle'] - bb_std * df[CLOSE_COLUMN].rolling(window=bb_window).std()

    def calculate_macd(self, df: pd.DataFrame) -> None:
        macd_fast = self.config.config['strategies']['macd']['fast']
        macd_slow = self.config.config['strategies']['macd']['slow']
        macd_signal = self.config.config['strategies']['macd']['signal']
        ema_fast = df[CLOSE_COLUMN].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df[CLOSE_COLUMN].ewm(span=macd_slow, adjust=False).mean()
        df['MACD_line'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD_line'].ewm(span=macd_signal, adjust=False).mean()
        df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']

    def calculate_atr(self, df: pd.DataFrame) -> None:
        atr_period = self.config.config['strategies']['atr']['period']
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df[CLOSE_COLUMN].shift()).abs()
        low_close = (df['Low'] - df[CLOSE_COLUMN].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=atr_period).mean()

class SequencePreparation:
    def __init__(self, config: Config):
        self.config = config
        self.sequence_length = config.sequence_length

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler]]:
        try:
            logger.info("Preparing sequences for LSTM training")
            feature_columns = self.config.feature_columns
            
            self.check_missing_columns(data, feature_columns)
            scalers, scaled_data = self.scale_data(data, feature_columns)

            X, y = self.create_sequences(scaled_data, feature_columns)
            logger.info(f"Sequences prepared. X shape: {X.shape}, y shape: {y.shape}")

            return X, y, scalers

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

    def check_missing_columns(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

    def scale_data(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[Dict[str, MinMaxScaler], pd.DataFrame]:
        scalers = {}
        scaled_data = pd.DataFrame(index=data.index)

        for column in feature_columns:
            scaler = MinMaxScaler()
            scaled_data[column] = scaler.fit_transform(data[[column]])
            scalers[column] = scaler

        return scalers, scaled_data

    def create_sequences(self, scaled_data: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Creating sequences and labels")
        X, y = [], []
        data_values = scaled_data[feature_columns].values
        close_values = scaled_data[CLOSE_COLUMN].values

        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i:(i + self.sequence_length)])
            
            # Determine labels: Buy (1), Sell (0), or Hold (2)
            if close_values[i + self.sequence_length] > close_values[i + self.sequence_length - 1]:
                y.append(1)  # Buy
            elif close_values[i + self.sequence_length] < close_values[i + self.sequence_length - 1]:
                y.append(0)  # Sell
            else:
                y.append(2)  # Hold

        return np.array(X), np.array(y)

class PreprocessingPipeline:
    def __init__(self, config: Config):
        self.config = config

    def process_symbol(self, symbol: str, sentiment_file: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler]]:
        try:
            logger.info(f"Starting preprocessing for {symbol}")

            data_loader = DataLoader(symbol, self.config)
            feature_engine = FeatureEngine(self.config)
            sequence_prep = SequencePreparation(self.config)

            # Check for existing processed data
            existing_X_file = f'data/{symbol}_X.npy'
            existing_y_file = f'data/{symbol}_y.npy'

            new_data_available = self.check_for_new_data(symbol)

            if os.path.exists(existing_X_file) and os.path.exists(existing_y_file) and not new_data_available:
                logger.info(f"Processed data already exists for {symbol}. Skipping preprocessing.")
                return np.load(existing_X_file), np.load(existing_y_file), self.load_scalers(symbol)

            # Load data
            merged_data = data_loader.load_and_merge_data(sentiment_file)
            # Calculate technical indicators
            data_with_indicators = feature_engine.calculate_technical_indicators(merged_data)
            # Fill missing values
            data_filled = data_with_indicators.ffill().bfill()

            # Prepare sequences for LSTM
            X, y, scalers = sequence_prep.prepare_sequences(data_filled)

            # Save processed data
            self.save_processed_data(symbol, X, y, scalers)

            logger.info(f"Preprocessing completed for {symbol}. Shape: X: {X.shape}, y: {y.shape}")
            return X, y, scalers

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            raise

    def preprocess_new_data(self, symbol: str, new_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler]]:
        """Process new data for a specific symbol and prepare for model updating."""
        try:
            logger.info(f"Processing new data for {symbol}...")
            feature_engine = FeatureEngine(self.config)
            sequence_prep = SequencePreparation(self.config)

            # Calculate technical indicators on the new data
            data_with_indicators = feature_engine.calculate_technical_indicators(new_data)
            data_filled = data_with_indicators.ffill().bfill()

            # Prepare sequences from the filled data
            X, y, scalers = sequence_prep.prepare_sequences(data_filled)

            logger.info(f"New data processed for {symbol}. X shape: {X.shape}, y shape: {y.shape}")

            return X, y, scalers

        except Exception as e:
            logger.error(f"Error processing new data for {symbol}: {str(e)}")
            raise

    def check_for_new_data(self, symbol: str) -> bool:
        # Check if any newer raw files exist in the expected directories
        data_dir = self.config.config['system']['data_directory']
        yfinance_file = self.get_yfinance_file(data_dir, symbol)

        if os.path.exists(yfinance_file):
            new_data_time = os.path.getmtime(yfinance_file)

            existing_X_file = f'data/{symbol}_X.npy'
            existing_y_file = f'data/{symbol}_y.npy'

            if os.path.exists(existing_X_file):
                existing_X_time = os.path.getmtime(existing_X_file)
                return new_data_time > existing_X_time  # Newer raw data found

        return False

    def get_yfinance_file(self, data_dir: str, symbol: str) -> str:
        if symbol == "XAUUSD":
            return os.path.join(data_dir, 'yfinance', 'gold_yf.csv')
        return os.path.join(data_dir, 'yfinance', f'{symbol}_yf.csv')

    def load_scalers(self, symbol: str) -> Dict[str, MinMaxScaler]:
        scaler_path = f'data/{symbol}_scalers.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_processed_data(self, symbol: str, X: np.ndarray, y: np.ndarray, scalers: Dict[str, MinMaxScaler]) -> None:
        np.save(f'data/{symbol}_X.npy', X)
        np.save(f'data/{symbol}_y.npy', y)

        scaler_path = f'data/{symbol}_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info(f"Saved scalers to: {scaler_path}")

    def process_all_symbols(self, sentiment_file: str) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_symbol, symbol, sentiment_file) for symbol in self.config.assets]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing: {e}")

def main() -> None:
    try:
        logger.info("Starting preprocessing pipeline...")
        config = Config('config/config.yaml')
        sentiment_file = os.path.join(config.config['system']['data_directory'], 'sentiment_data.csv')
        
        if not os.path.exists(sentiment_file):
            logger.error(f"Sentiment data file not found: {sentiment_file}")
            raise FileNotFoundError(sentiment_file)

        pipeline = PreprocessingPipeline(config)
        pipeline.process_all_symbols(sentiment_file)
        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()