import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Load preprocessed data for AAPL
data_path = os.path.join("preprocessed_data", "AAPL_preprocessed_data.csv")
data = pd.read_csv(data_path)

# Features and target column
features = ['Adj Close', 'Volume', 'MA7', 'MA30', 'EMA9', 'EMA50', 'Volatility', 'RSI',
            'BB_Width', 'MACD', 'MACD_Signal', 'Momentum', 'OBV', 'Daily Return']
target = 'Target'

# Filter data to a specific training range
train_start_date = "2015-02-13"  # Start of training data
train_end_date = "2019-12-31"    # End of training data, leaving 2020+ for evaluation
data["Date"] = pd.to_datetime(data["Date"])
data = data[(data["Date"] >= train_start_date) & (data["Date"] <= train_end_date)]

# Debug: Check data after filtering
print(f"Data shape after filtering for training: {data.shape}")
if data.empty:
    raise ValueError("No data available for training after filtering. Check date range or dataset.")

# Initialize scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale features and target separately for training
X_train = scaler_X.fit_transform(data[features])
y_train = scaler_y.fit_transform(data[[target]])

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Sequence length for LSTM
sequence_length = 50  # Example: using 50 previous time steps

# Prepare sequential data
def create_sequences(X, y, sequence_length):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length])
    return torch.stack(sequences_X), torch.stack(sequences_y)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)

# Split into training and validation sets (80/20 split)
train_size = int(0.8 * len(X_train_seq))
X_train, X_val = X_train_seq[:train_size], X_train_seq[train_size:]
y_train, y_val = y_train_seq[:train_size], y_train_seq[train_size:]

# Create data loaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# Load the model
from model3_architecture import StockPredictionModel3

input_size = len(features)  # Number of input features
d_model = 128               # Model dimension
num_heads = 4               # Number of attention heads
num_layers = 2              # Number of Transformer layers
ff_dim = 512                # Dimension of feed-forward layers
output_size = 1             # Single output (e.g., next day's price)

model = StockPredictionModel3(input_size, d_model, num_heads, num_layers, output_size, ff_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training the model
num_epochs = 1000
train_losses, val_losses = [], []
early_stopping_patience = 50
best_val_loss = float('inf')
epochs_no_improve = 0

print("Starting Training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    # Evaluate on validation data
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    val_loss /= len(val_loader.dataset)

    # Save training and validation losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Learning rate scheduling
    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model3.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best Validation Loss: {best_val_loss:.4f}")
            break

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate on the validation set
model.load_state_dict(torch.load('model3.pth'))
model.eval()
val_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        val_loss += loss.item() * batch_X.size(0)

val_loss /= len(val_loader.dataset)
print(f"Final Validation MSE: {val_loss:.4f}")
