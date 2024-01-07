import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

# Încărcarea și pregătirea datelor
df = pd.read_csv('Truck_sales.csv')
timeseries = df[["Number_Trucks_Sold"]].values.astype('float32')

# Normalizarea datelor
scaler = MinMaxScaler(feature_range=(-1, 1))
timeseries_normalized = scaler.fit_transform(timeseries)

# Divizarea în seturi de antrenament și test
train_size = int(len(timeseries_normalized) * 0.67)
test_size = len(timeseries_normalized) - train_size
train, test = timeseries_normalized[:train_size], timeseries_normalized[train_size:]

# Funcția pentru crearea setului de date
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:(i + lookback), 0])
        y.append(dataset[i + lookback, 0])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

lookback = 4
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)


# Definirea modelului
class TrucksModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


model = TrucksModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# Antrenarea modelului
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred_train, y_train))
        y_pred_test = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred_test, y_test))
    print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")

# Predicții și plotare
with torch.no_grad():
    train_plot = np.empty_like(timeseries)
    train_plot[:, :] = np.nan

    y_pred_train = model(X_train).squeeze().numpy()
    y_pred_train = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    train_plot[lookback:lookback + len(y_pred_train), 0] = y_pred_train

    test_plot = np.empty_like(timeseries)
    test_plot[:, :] = np.nan

    y_pred_test = model(X_test).squeeze().numpy()
    y_pred_test = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    test_plot[train_size + lookback:train_size + lookback + len(y_pred_test), 0] = y_pred_test

plt.plot(scaler.inverse_transform(timeseries_normalized), label='Original Data')
plt.plot(train_plot, c='r', label='Train Predictions')
plt.plot(test_plot, c='g', label='Test Predictions')
plt.legend()
plt.show()
