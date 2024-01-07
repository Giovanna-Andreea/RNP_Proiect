import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('gold_price_data.csv')
timeseries = df[["Value"]].values.astype('float32')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
timeseries_normalized = scaler.fit_transform(timeseries)

# train-test split for time series
train_size = int(len(timeseries_normalized) * 0.67)
test_size = len(timeseries_normalized) - train_size
train, test = timeseries_normalized[:train_size], timeseries_normalized[train_size:]


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)


class GoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


model = GoldModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = scaler.inverse_transform(y_pred[:, -1, :].numpy())
    train_plot[lookback:train_size] = y_pred
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_test)
    y_pred = scaler.inverse_transform(y_pred[:, -1, :].numpy())
    test_plot[train_size + lookback:len(timeseries)] = y_pred

# plot
plt.plot(scaler.inverse_transform(timeseries_normalized), label='Original Data')
plt.plot(train_plot, c='r', label='Train Predictions')
plt.plot(test_plot, c='g', label='Test Predictions')
plt.legend()
plt.show()
