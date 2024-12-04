import torch
import torch.nn as nn

class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StockPredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, seq_len, hidden_size)

        # Get the last hidden state for each batch
        last_hidden_state = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Fully connected layer
        output = self.fc(last_hidden_state)  # Shape: (batch_size, output_size)

        return output


# Example configuration
if __name__ == "__main__":
    input_size = 14  # Number of features in the dataset
    hidden_size = 64
    output_size = 1
    num_layers = 2

    model = StockPredictionModel(input_size, hidden_size, output_size, num_layers)
    print(model)
