import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator

# Load data from a CSV file into a Pandas DataFrame
df_gold = pd.read_csv("gold_price_data.csv", index_col=0, parse_dates=True)
df_gold = df_gold.dropna()

# Check if the DataFrame is empty after dropping NaN values
if df_gold.empty:
    print("DataFrame is empty after removing NaN values.")
else:
    print("DataFrame shape:", df_gold.shape)
    print("DataFrame head:\n", df_gold.head())

    # Determine the index to split the dataset
    split_date = "2020-03-13"  # set the desired split date
    split_idx = int(len(df_gold) * 0.8)
    prediction_length = len(df_gold) - split_idx

    # Split the data for training and testing
    train_data = ListDataset(
        [{"start": df_gold.index[0], "target": df_gold.iloc[:split_idx]["Value"].values, "is_pad": False}],
        freq="D",  # Change the frequency to monthly for better visualization
    )
    test_data = ListDataset(
        [{"start": df_gold.index[0], "target": df_gold.iloc[split_idx:]["Value"].values, "is_pad": False}],
        freq="D",
    )

    # Train the model and make predictions
    model = DeepAREstimator(
        prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 5}
    ).train(train_data)

    forecasts = list(model.predict(test_data))

    # Plot predictions
    plt.plot(df_gold.index[:split_idx], df_gold.iloc[:split_idx]["Value"], color="black")  # plot training data
    for forecast in forecasts:
        forecast_index = pd.date_range(
            start=df_gold.index[split_idx], periods=forecast.samples.shape[1], freq="D"
        )
        plt.plot(forecast_index, forecast.samples[0], color="blue", alpha=0.5)  # plot forecasts

    plt.legend(["True values", "Predictions"], loc="upper left", fontsize="xx-large")
    plt.show()
