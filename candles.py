import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MT5Credentials:
    """Store MT5 connection credentials."""
    login: int
    password: str
    server: str

class MT5Handler:
    """Handles MetaTrader 5 initialization and operations."""
    TIMEFRAME_MAP: Dict[str, int] = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }

    def __init__(self, credentials: MT5Credentials):
        self.credentials = credentials
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def initialize(self) -> bool:
        """Initialize MT5 connection."""
        if self._initialized:
            return True

        success = mt5.initialize(
            login=self.credentials.login,
            password=self.credentials.password,
            server=self.credentials.server
        )
        if not success:
            error_code, error_desc = mt5.last_error()
            logger.error(f"Failed to initialize MT5: {error_code} - {error_desc}")
            return False

        self._initialized = True
        logger.info("MT5 connection initialized successfully.")
        return True

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 connection closed.")

    @classmethod
    def get_mt5_timeframe(cls, timeframe: str) -> Optional[int]:
        """Convert string timeframe to MT5 timeframe constant."""
        return cls.TIMEFRAME_MAP.get(timeframe)

class DataManager:
    """Manages candle data storage and updates."""
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = base_dir
        self.candles_dir = os.path.join(base_dir, 'candles')

        # Ensure the `data` and `candles` directories exist
        if not os.path.exists(self.base_dir):
            logger.info(f"Creating 'data' directory at {self.base_dir}")
            os.makedirs(self.base_dir)

        if not os.path.exists(self.candles_dir):
            logger.info(f"Creating 'candles' directory at {self.candles_dir}")
            os.makedirs(self.candles_dir)

        logger.info(f"Data will be stored at: {self.candles_dir}")

    def get_file_path(self, symbol: str, timeframe: str) -> str:
        """Generate the file path for a specific symbol and timeframe."""
        return os.path.join(self.candles_dir, f"{symbol}_{timeframe}.csv")

    def get_last_candle_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Retrieve the timestamp of the last candle in the file."""
        file_path = self.get_file_path(symbol, timeframe)
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            return pd.to_datetime(df['time'].iloc[-1])
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def update_data(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> bool:
        """Update the existing candle data file with new data."""
        file_path = self.get_file_path(symbol, timeframe)
        try:
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path)
                existing_data['time'] = pd.to_datetime(existing_data['time'])

                # Combine and deduplicate data
                combined_data = pd.concat([existing_data, new_data]).drop_duplicates(
                    subset=['time'], keep='last'
                ).sort_values('time')

                # Save updated data
                combined_data.to_csv(file_path, index=False)
                logger.info(f"Updated {symbol} {timeframe} with {len(new_data)} new candles.")
            else:
                new_data.to_csv(file_path, index=False)
                logger.info(f"Created new file for {symbol} {timeframe} with {len(new_data)} candles.")
            return True
        except Exception as e:
            logger.error(f"Error updating data for {symbol} {timeframe}: {e}")
            return False

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators like RSI, MA, and Bollinger Bands."""
    # RSI Calculation
    delta = df['close'].diff()  # Diff of closing prices
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the 14-day period
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Averages
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_50'] + (df['close'].rolling(window=50).std() * 2)
    df['Bollinger_Lower'] = df['SMA_50'] - (df['close'].rolling(window=50).std() * 2)

    return df

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def fetch_data_range(
    handler: MT5Handler,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Fetch candle data for a specific date range."""
    try:
        mt5_timeframe = handler.get_mt5_timeframe(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        if rates is None:
            logger.error(f"Failed to fetch data for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add technical indicators
        df = add_technical_indicators(df)

        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
        return None

def update_symbol_data(
    handler: MT5Handler,
    data_manager: DataManager,
    symbol: str,
    timeframe: str,
    lookback_days: int = 730
) -> bool:
    """Fetch and update data for a specific symbol and timeframe."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        logger.info(f"Fetching data for {symbol} {timeframe} from {start_date} to {end_date}")

        df = fetch_data_range(handler, symbol, timeframe, start_date, end_date)
        if df is None or df.empty:
            logger.info(f"No new data available for {symbol} {timeframe}.")
            return True

        return data_manager.update_data(symbol, timeframe, df)
    except Exception as e:
        logger.error(f"Error updating {symbol} {timeframe}: {e}")
        return False

def update_all_symbols(
    symbols: List[str],
    timeframes: List[str],
    credentials: MT5Credentials,
    max_workers: int = 5
) -> None:
    """Update data for all symbols and timeframes."""
    data_manager = DataManager()
    with MT5Handler(credentials) as handler:
        if not handler._initialized:
            logger.error("MT5 initialization failed.")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(update_symbol_data, handler, data_manager, symbol, timeframe): (symbol, timeframe)
                for symbol in symbols
                for timeframe in timeframes
            }

            for future in as_completed(futures):
                symbol, timeframe = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.warning(f"Failed to update {symbol} {timeframe}.")
                except Exception as e:
                    logger.error(f"Task failed for {symbol} {timeframe}: {e}")

def continuous_data_update():
    """Update candle data for all configured symbols and timeframes."""
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    symbols = config.get('assets', [])
    timeframes = config.get('data_config', {}).get('timeframe', [])

    if not symbols or not timeframes:
        logger.error("Symbols or timeframes not properly configured.")
        return

    credentials = MT5Credentials(
        login=int(config['mt5_credentials']['account_number']),
        password=config['mt5_credentials']['password'],
        server=config['mt5_credentials']['server']
    )

    update_all_symbols(
        symbols=symbols,
        timeframes=timeframes,
        credentials=credentials
    )

if __name__ == "__main__":
    # Load configuration and perform an update
    continuous_data_update()