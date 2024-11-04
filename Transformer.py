import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Load the dataset
data = pd.read_csv('BNBUSDT.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', ascending=True)
data.set_index('Date', inplace=True)

# Feature engineering
data['Price'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Normalize the features
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data.values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_length, :-1], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_length - 1, -1], dtype=torch.float32))

# Prepare dataset and dataloader
seq_length = 10
dataset = TimeSeriesDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=1, num_encoder_layers=3, batch_first=True)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Shift the input sequence to create the target sequence
        tgt = x[:, :-1]  # Target sequence is all but the last element
        src = x[:, 1:]   # Source sequence is all but the first element
        x = self.transformer(src, tgt)
        return self.fc(x[:, -1, :])  # Output from the last time step

# Train the model
model = TransformerModel(input_dim=5)  # 5 input features
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):  # Training epochs
    for inputs, target in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

# Prediction logic
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        prediction = model(input_tensor)
        return prediction.numpy()

# Example of predicting the next price
last_sequence = data[-seq_length:].values[:, :-1]
predicted_price = predict(model, last_sequence)
trend = "Bullish" if predicted_price > data['Close'].values[-1] else "Bearish"

print(f"Predicted Price: {predicted_price}, Trend: {trend}")