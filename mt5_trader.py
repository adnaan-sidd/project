import MetaTrader5 as mt5
import logging
import yaml
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List
from models.lstm_model import make_predictions  # Ensure this is the correct import path
from concurrent.futures import ThreadPoolExecutor
import pytz  # Import pytz for timezone handling
import tensorflow as tf
import gc

# Disable TensorFlow oneDNN custom operations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class MT5Trader:
    """Class to handle MT5 trading operations."""
    
    BUY_THRESHOLD = 1.05
    SELL_THRESHOLD = 1.02

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = self._setup_logging()
        self.credentials = self.get_mt5_credentials()
        self.available_balance = None
        self.existing_trades = []
        self.initialize_mt5()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure and return logger instance."""
        os.makedirs('logs', exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(f'logs/mt5_trader_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        if not os.path.exists(self.config_path):
            raise ConfigurationError(f"Config file not found at {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self._validate_config(config)
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_fields = ['mt5_credentials', 'assets', 'email', 'trading_preferences', 'risk_management']
        mt5_required_fields = ['account_number', 'password', 'server']
        email_required_fields = ['sender', 'recipient', 'smtp_server', 'smtp_port', 'password']
        
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ConfigurationError(f"Missing required fields in config: {missing_fields}")
        
        missing_mt5_fields = [field for field in mt5_required_fields if field not in config['mt5_credentials']]
        if missing_mt5_fields:
            raise ConfigurationError(f"Missing required MT5 credentials: {missing_mt5_fields}")
        
        missing_email_fields = [field for field in email_required_fields if field not in config['email']]
        if missing_email_fields:
            raise ConfigurationError(f"Missing required email configuration: {missing_email_fields}")

    def get_mt5_credentials(self) -> Dict[str, Any]:
        """Get MT5 credentials from config."""
        try:
            return {
                'login': int(self.config['mt5_credentials']['account_number']),
                'password': self.config['mt5_credentials']['password'],
                'server': self.config['mt5_credentials']['server']
            }
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigurationError(f"Invalid MT5 credentials configuration: {e}")

    def initialize_mt5(self) -> None:
        """Initialize MT5 connection with a retry mechanism."""
        retries = 3
        for attempt in range(retries):
            if mt5.initialize(login=self.credentials['login'], password=self.credentials['password'], server=self.credentials['server']):
                self.logger.info("MT5 connection initialized successfully")
                self.available_balance = self.get_account_balance()
                self.logger.info(f"Available account balance: {self.available_balance}")
                self.existing_trades = self.fetch_existing_trades()
                return
            else:
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: {error_code}, attempt {attempt + 1} of {retries}")
                time.sleep(5)  # Wait before retrying
        raise RuntimeError(f"Failed to initialize MT5 after {retries} attempts")

    def shutdown_mt5(self) -> None:
        """Shutdown MT5 connection."""
        mt5.shutdown()
        self.logger.info("MT5 connection closed")

    def send_email(self, subject: str, body: str) -> None:
        """Send an email notification."""
        config = self.config['email']
        msg = MIMEMultipart()
        msg['From'] = config['sender']
        msg['To'] = config['recipient']
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['sender'], config['password'])
                server.sendmail(config['sender'], config['recipient'], msg.as_string())
            self.logger.info("Email notification sent successfully")
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    def get_account_balance(self) -> float:
        """Get the current account balance."""
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return None
        return account_info.balance

    def fetch_existing_trades(self) -> List[Dict[str, Any]]:
        """Fetch existing trades from the account."""
        return [trade for trade in mt5.positions_get() if trade]

    def calculate_position_size(self, risk: float, stop_loss: float, account_balance: float) -> float:
        """Calculate the position size based on risk management."""
        risk_amount = account_balance * risk
        position_size = risk_amount / stop_loss
        # Ensure the position size is within defined constraints
        return min(max(position_size, self.config['risk_management']['base_position_size']),
                   self.config['risk_management']['max_position_size'])
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if the market for the given symbol is open."""
        market_info = mt5.symbol_info(symbol)
        if market_info is None or not market_info.visible:
            self.logger.error(f"Symbol {symbol} not found or not visible.")
            return False

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            self.logger.error(f"Failed to get tick information for {symbol}.")
            return False

        server_time = tick_info.time
        current_time = datetime.fromtimestamp(server_time)
        
        timezone_str = self.config['trading_preferences'].get('timezone', 'UTC')
        start_hour = self.config['trading_preferences'].get('start_hour', 0)
        end_hour = self.config['trading_preferences'].get('end_hour', 23)

        try:
            timezone = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            self.logger.error(f"Unknown timezone: {timezone_str}. Defaulting to UTC.")
            timezone = pytz.utc
        
        current_time = current_time.astimezone(timezone)
        return start_hour <= current_time.hour <= end_hour
    
    def execute_trade(self, symbol: str, action: str) -> bool:
        """Execute a trade on MT5 with risk management."""
        if action not in ['buy', 'sell']:
            self.logger.error(f"Invalid trade action: {action}.")
            return False

        if not self.is_market_open(symbol):
            self.logger.error(f"Market closed for symbol {symbol}. Cannot execute trade.")
            return False

        # Check if symbol is visible in market
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found.")
            return False

        if not symbol_info.visible:
            self.logger.info(f"Symbol {symbol} is not visible, trying to add it.")
            if not mt5.symbol_select(symbol):
                self.logger.error(f"Failed to add symbol {symbol} to the market watch.")
                return False

        # Price determination for order execution
        price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
        if price is None:
            self.logger.error(f"No price available for executing trade on {symbol}.")
            return False

        # Calculate stop loss and take profit
        stop_loss_percentage = self.config['risk_management']['stop_loss']
        take_profit_percentage = self.config['risk_management']['take_profit']

        stop_loss = price * (1 - stop_loss_percentage) if action == 'buy' else price * (1 + stop_loss_percentage)
        take_profit = price * (1 + take_profit_percentage) if action == 'buy' else price * (1 - take_profit_percentage)

        # Get current balance and calculate position size
        account_balance = self.get_account_balance()
        if account_balance is None:
            return False
        
        position_size = self.calculate_position_size(
            self.config['risk_management']['max_risk_per_trade'],
            stop_loss_percentage,
            account_balance
        )
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 234000,
            "comment": "Trade based on predictions",
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = result.comment if result else "No result returned."
            self.logger.error(f"Failed to send trade order for {symbol}: {error_msg}")
            return False

        self.logger.info(f"Trade executed: {action} {symbol} at {price} with volume {position_size}")
        
        # Logging expected profit/loss details
        expected_profit = (take_profit - price) * position_size if action == 'buy' else (price - take_profit) * position_size
        expected_loss = (price - stop_loss) * position_size if action == 'buy' else (stop_loss - price) * position_size
        self.logger.info(f"Expected Profit: {expected_profit:.2f}, Expected Loss: {expected_loss:.2f}")

        # Update expected profit timeline
        duration_minutes = self.config['trading_preferences'].get('trade_duration', 60)
        profit_expected_at = datetime.now() + timedelta(minutes=duration_minutes)
        self.logger.info(f"Profit can be expected around: {profit_expected_at.strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    def manage_existing_trades(self, symbol: str, latest_prediction: float) -> None:
        """Modify existing trades based on the new prediction."""
        # Refresh existing trades
        self.existing_trades = self.fetch_existing_trades()

        for trade in self.existing_trades:
            if trade.symbol != symbol or (trade.type == mt5.ORDER_TYPE_BUY and latest_prediction <= self.BUY_THRESHOLD) or (trade.type == mt5.ORDER_TYPE_SELL and latest_prediction >= self.SELL_THRESHOLD):
                continue  # Skip trades that do not need to be modified
            
            # Adjust the stops based on the new prediction
            new_stop_loss = mt5.symbol_info_tick(symbol).ask * (1 - self.config['risk_management']['stop_loss']) if trade.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid * (1 + self.config['risk_management']['stop_loss'])
            new_take_profit = mt5.symbol_info_tick(symbol).ask * (1 + self.config['risk_management']['take_profit']) if trade.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid * (1 - self.config['risk_management']['take_profit'])

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": trade.ticket,
                "sl": new_stop_loss,
                "tp": new_take_profit,
                "deviation": 10,
                "magic": 234000,
                "comment": "Modified based on new predictions"
            }

            result = mt5.order_send(modify_request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else "No result returned."
                self.logger.error(f"Failed to modify trade {trade.ticket} for {symbol}: {error_msg}")
            else:
                self.logger.info(f"Modified trade {trade.ticket} with new SL: {new_stop_loss}, TP: {new_take_profit}")

    def trade_on_symbol(self, symbol: str) -> None:
        """Conduct trading operations for a single symbol based on predictions."""
        self.logger.info(f"Starting predictions for {symbol}")
        try:
            prediction_tuple = make_predictions(symbol)  
            if prediction_tuple is None or prediction_tuple[0] is None:
                self.logger.error(f"No predictions returned for {symbol}. Skipping.")
                return
            
            prediction_values = prediction_tuple[0]  
            self.logger.info(f"Prediction for {symbol}: {prediction_values}")
            latest_prediction = prediction_values[-1]

            # Determine action based on the prediction
            action = None
            if latest_prediction > self.BUY_THRESHOLD:
                action = 'buy'
            elif latest_prediction < self.SELL_THRESHOLD:
                action = 'sell'
            else:
                self.logger.info(f"Holding position for {symbol} as price is within the range.")
                return

            self.execute_trade(symbol, action)
            self.manage_existing_trades(symbol, latest_prediction)

        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")

    def trade_on_predictions(self) -> None:
        """Main function for trading based on predictions."""
        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit number of threads
            executor.map(self.trade_on_symbol, self.config['assets'])

if __name__ == '__main__':
    config_path = 'config/config.yaml'  # Correct path for the config file
    trader = MT5Trader(config_path)

    try:
        trader.trade_on_predictions()
    except Exception as e:
        trader.logger.error(f"An error occurred: {e}")
    finally:
        trader.shutdown_mt5()
        gc.collect()  # Explicit garbage collection