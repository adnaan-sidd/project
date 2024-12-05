import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yaml
import logging
import asyncio
import os
import pandas as pd
import tensorflow as tf  # Import TensorFlow
from datetime import datetime
from news.news_scraper import NewsScraperEnhanced
from candles import continuous_data_update as candles_update
from yfinance import continuous_data_update as yfinance_update
from preprocess_data import main as preprocess_main, preprocess_new_data
from models.lstm_model import train_model, update_model_with_new_data
from backtest import Backtest
from mt5_trader import MT5Trader

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class AsyncTradingBot:
    """Asynchronous trading bot class that orchestrates all operations."""

    def __init__(self, config_path: str):
        """Initialize the trading bot with configuration."""
        self.config_path = config_path
        self.config = None
        self.logger = self._setup_logging()

        # Set TensorFlow logging level to DEBUG to capture more details
        tf.get_logger().setLevel('DEBUG')

        # Set environment variables for thread control
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
        os.environ['TF_NUM_INTEROP_THREADS'] = '4'

        # Limit TensorFlow resource usage
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure and return logger instance."""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        return logging.getLogger(__name__)

    def load_config(self) -> None:
        """Load and validate YAML configuration."""
        if not os.path.exists(self.config_path):
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise ConfigurationError("Configuration file not found.")

        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            if not self.config.get('assets'):
                raise ConfigurationError("Assets not specified in the configuration file.")
            self.logger.info("Configuration loaded successfully.")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}", exc_info=True)
            raise ConfigurationError("Error parsing YAML configuration.")

    async def fetch_news(self) -> None:
        """Fetch news articles."""
        try:
            scraper = NewsScraperEnhanced(config_path="config/config.yaml")
            scraper.fetch_news()
            self.logger.info("News articles fetched.")
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}", exc_info=True)
            raise

    async def update_candles(self) -> None:
        """Update candle data."""
        try:
            candles_update()
            self.logger.info("Candle data updated.")
        except Exception as e:
            self.logger.error(f"Error updating candle data: {e}", exc_info=True)
            raise

    async def update_yfinance(self) -> None:
        """Update Yahoo Finance data."""
        try:
            # Directly specify the correct symbols
            assets = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F"]
            self.logger.info(f"Updating Yahoo Finance data for assets: {assets}")
            yfinance_update(assets=assets)
            self.logger.info("Yahoo Finance data updated.")
        except Exception as e:
            self.logger.error(f"Error updating Yahoo Finance data: {e}", exc_info=True)
            raise

    async def preprocess_data(self) -> None:
        """Preprocess data."""
        try:
            preprocess_main()
            self.logger.info("Data preprocessing completed.")
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}", exc_info=True)
            raise

    async def train_models(self) -> None:
        """Train models for each asset."""
        try:
            for asset in self.config['assets']:
                self.logger.info(f"Training model for {asset}...")

                model_path = f'models/{asset}_best_model.keras'
                self.logger.info(f"Loading model from {model_path}...")

                # Load the model with a timeout
                model = load_model_test(model_path)
                if model:
                    self.logger.info(f"Model for {asset} loaded successfully.")
                else:
                    self.logger.error(f"Failed to load model for {asset}.")
                
                train_model(symbol=asset)
            self.logger.info("All models trained successfully.")
        except Exception as e:
            self.logger.error(f"Error training models: {e}", exc_info=True)
            raise

    async def run_backtest(self) -> None:
        """Run backtest."""
        try:
            assets = self.config.get('assets', [])
            initial_balance = self.config.get('initial_balance', 10000)
            backtest = Backtest(assets, initial_balance)
            backtest.run_backtest()
            self.logger.info("Backtest completed.")
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}", exc_info=True)
            raise

    async def execute_trades(self) -> None:
        """Execute trades."""
        try:
            trader = MT5Trader(self.config_path)
            trader.trade_on_predictions()
            trader.shutdown_mt5()
            self.logger.info("Trades executed successfully.")
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}", exc_info=True)
            raise

    async def process_new_data(self, symbol: str, new_data: pd.DataFrame):
        """Process new data for a given symbol and update the model."""
        try:
            X, y, _ = preprocess_new_data(symbol, new_data)
            update_model_with_new_data(symbol, X, y)
        except Exception as e:
            self.logger.error(f"Error processing new data for {symbol}: {e}", exc_info=True)
            raise

    async def run_sequence(self) -> None:
        """Run all tasks in sequence, then pause for 30 minutes."""
        while True:
            try:
                await self.fetch_news()
                await self.update_candles()
                await self.update_yfinance()
                await self.preprocess_data()
                await self.train_models()
                await self.run_backtest()
                await self.execute_trades()

                # Process new data as an example (fetch new data and process it)
                for asset in self.config['assets']:
                    new_data = self.fetch_new_data_for_asset(asset)  # Implement this function accordingly
                    await self.process_new_data(asset, new_data)

                self.logger.info("All tasks completed. Waiting for 30 minutes...")
                await asyncio.sleep(1800)  # 30-minute delay
            except Exception as e:
                self.logger.error(f"Error during execution: {e}", exc_info=True)
                self.logger.info("Retrying in 5 minutes...")
                await asyncio.sleep(300)  # Retry after 5 minutes

    def fetch_new_data_for_asset(self, asset: str) -> pd.DataFrame:
        """Fetch new data for a given asset (this is a placeholder function)."""
        # Implement this function to fetch new data for the given asset
        # This is just a placeholder and should be replaced with actual implementation
        return pd.DataFrame()

    def start(self) -> None:
        """Start the trading bot."""
        try:
            self.load_config()
            asyncio.run(self.run_sequence())
        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped.")
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")

def load_model_test(model_path):
    """Load model with timeout handling and resource monitoring."""
    import time
    start_time = time.time()
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    bot = AsyncTradingBot('config/config.yaml')
    bot.start()