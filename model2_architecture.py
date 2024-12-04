import torch
import torch.nn as nn


class StockPredictionModel2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(StockPredictionModel2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Dropout layer for the fully connected output
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)

        # Get the last hidden state for each batch
        last_hidden_state = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)

        # Apply batch normalization
        last_hidden_state = self.batch_norm(last_hidden_state)

        # Fully connected layer with dropout
        output = self.fc(self.dropout(last_hidden_state))  # (batch_size, output_size)

        return output


# Example configuration for Model2
if __name__ == "__main__":
    input_size = 14  # Number of features in the dataset
    hidden_size = 128  # Increased hidden size
    output_size = 1
    num_layers = 3  # Increased number of layers
    dropout = 0.3  # Dropout rate

    model = StockPredictionModel2(input_size, hidden_size, output_size, num_layers, dropout)
    print(model)

