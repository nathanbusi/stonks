import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Add positional encoding to input
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class StockPredictionModel3(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, output_size, ff_dim):
        super(StockPredictionModel3, self).__init__()
        self.d_model = d_model

        # Linear layer to project input to d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_proj(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Take the representation at the last time step
        output = self.fc(x[:, -1, :])  # Use the last time step
        return output


# Example configuration
if __name__ == "__main__":
    input_size = 14  # Number of features
    d_model = 128    # Model dimension
    num_heads = 4    # Number of attention heads
    num_layers = 2   # Number of transformer layers
    ff_dim = 512     # Dimension of feed-forward layers
    output_size = 1  # Predicting a single value (e.g., next day's price)

    model = StockPredictionModel3(input_size, d_model, num_heads, num_layers, output_size, ff_dim)
    print(model)
