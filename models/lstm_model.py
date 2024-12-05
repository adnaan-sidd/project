import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import json
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

class HybridModel(nn.Module):
    """A hybrid model with 1D Convolutional and LSTM layers."""
    def __init__(self, input_shape, config):
        super(HybridModel, self).__init__()
        conv_filters = config['conv_params']['conv_filters']
        kernel_size = config['conv_params']['kernel_size']
        lstm_layers = config['model_params']['lstm']['layers']
        dropout_rate = config['model_params']['lstm']['dropout']
        num_classes = config['model_params']['num_classes']

        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(input_shape[0], conv_filters, kernel_size)
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # LSTM Layers
        lstm_input_size = conv_filters
        self.lstm_layers = nn.ModuleList()
        for lstm_units in lstm_layers:
            self.lstm_layers.append(nn.LSTM(lstm_input_size, lstm_units, batch_first=True, bidirectional=True))
            lstm_input_size = lstm_units * 2

        # Dynamically calculate fc_input_size
        with torch.no_grad():
            dummy_input = torch.randn(1, input_shape[1], input_shape[0])
            dummy_input = dummy_input.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)
            x = self.conv1d(dummy_input)
            x = self.maxpool(x)
            x = x.permute(0, 2, 1)  # Reshape back for LSTM
            
            for lstm in self.lstm_layers:
                x, _ = lstm(x)
            
            # Flatten the LSTM output
            fc_input_size = x.contiguous().view(x.size(0), -1).size(1)
            
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # Reshape back for LSTM
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Flatten the LSTM output to prepare it for the fully connected layer
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = self.fc(x)
        return x

def load_data(symbol: str):
    try:
        data_directory = "data"
        X = np.load(f'{data_directory}/{symbol}_X.npy')
        y = np.load(f'{data_directory}/{symbol}_y.npy')
        logger.info(f"Loaded data for {symbol}: X shape {X.shape}, y shape {y.shape}.")
        return X, y
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        raise

def check_for_new_data(symbol: str) -> bool:
    existing_data_path = f'data/{symbol}_X.npy'
    new_data_path = f'data/{symbol}_X_new.npy'
    
    try:
        if os.path.exists(existing_data_path) and os.path.exists(new_data_path):
            existing_data = np.load(existing_data_path)
            new_data = np.load(new_data_path)
            return existing_data.shape[0] < new_data.shape[0]
        else:
            logger.info("No existing or new data found.")
            return False
    except Exception as e:
        logger.error(f"Error checking for new data for {symbol}: {e}")
        return False

def train_model(symbol: str, config, model_path: str = None) -> str:
    model_path = model_path or f'models/{symbol}_best_model.pth'
    log_dir = f"{config['system']['log_directory']}/{symbol}"

    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}.")
        model = torch.load(model_path)

        if check_for_new_data(symbol):
            logger.info("New data detected. Retraining the model...")
            existing_X, existing_y = load_data(symbol)
            new_X, new_y = load_data(symbol)
            all_X = np.concatenate((existing_X, new_X))
            all_y = np.concatenate((existing_y, new_y))
        else:
            logger.info("No new data detected. Continuing with existing model.")
            all_X, all_y = load_data(symbol)
    else:
        logger.info("No existing model found. Training from scratch.")
        all_X, all_y = load_data(symbol)
        input_shape = (all_X.shape[2], all_X.shape[1])  # Adjust input shape
        logger.info(f"Input shape: {input_shape}")
        model = HybridModel(input_shape, config)

    X_train, X_val, y_train, y_val = train_test_split(all_X, all_y, test_size=config['preprocessing']['train_test_split'], shuffle=False)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=config['model_params']['lstm']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model_params']['lstm']['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer_type = config['model_params']['lstm']['optimizer']['type']
    learning_rate = config['model_params']['lstm']['optimizer'].get('learning_rate', 0.001)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized.")

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    patience = config['model_params']['lstm']['epochs'] // 10
    patience_counter = 0

    for epoch in range(config['model_params']['lstm']['epochs']):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info(f"Epoch {epoch+1}/{config['model_params']['lstm']['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    writer.close()
    save_training_history(symbol, train_loss, val_loss)
    return get_latest_prediction(symbol, model)

def save_training_history(symbol: str, train_loss, val_loss):
    history = {'train_loss': train_loss, 'val_loss': val_loss}
    with open(f'models/{symbol}_training_history.json', 'w') as f:
        json.dump(history, f)

def get_latest_prediction(symbol: str, model: nn.Module) -> str:
    try:
        data_directory = "data"
        X = np.load(f'{data_directory}/{symbol}_X.npy')
        latest_data = torch.tensor(X[-1:], dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            outputs = model(latest_data)
            predicted_class = torch.argmax(outputs, dim=1).item()

        signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
        return signal_map[predicted_class]

    except Exception as e:
        logger.error(f"Error getting latest prediction for {symbol}: {e}")
        return None

def make_predictions(symbol: str) -> str:
    model_path = f'models/{symbol}_best_model.pth'

    if not os.path.exists(model_path):
        logger.warning(f"No model found for {symbol}. Please train the model first.")
        return None 

    model = torch.load(model_path)
    data_directory = "data"
    X = np.load(f'{data_directory}/{symbol}_X.npy')

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            data = torch.tensor(X[i:i+1], dtype=torch.float32)
            outputs = model(data)
            predicted_class = torch.argmax(outputs, dim=1).item()
            predictions.append(predicted_class)

    signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
    predicted_labels = [signal_map[c] for c in predictions]

    return predicted_labels[-1]

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load configurations
    config = load_config('config/config.yaml')

    symbols = config['assets']

    for symbol in symbols:
        prediction = train_model(symbol, config)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")