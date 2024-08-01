# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install yfinance
# pip install IPython
# pip install plotly
# pip install statsmodels
# pip install pytrends

import numpy as np
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import warnings
import time
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

warnings.filterwarnings("ignore")  # Suppress warnings

def fetch_google_trends_data(ticker, start_date, end_date, retries=5, delay=60):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([ticker], cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    for i in range(retries):
        try:
            trends_data = pytrends.interest_over_time()
            trends_data = trends_data.drop(columns='isPartial')
            return trends_data
        except Exception as e:
            if i < retries - 1:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

def load_and_process_data(ticker, start_date, end_date, local_file=None, split_ratio=0.7, split_by_date=False, columns_to_scale=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)
    data.fillna(method='ffill', inplace=True)

    # Add Google Trends data
    trends_cache_file = f"/content/drive/My Drive/Cos30018/{ticker}_trends.csv"
    if local_file and os.path.exists(trends_cache_file):
        trends_data = pd.read_csv(trends_cache_file, index_col='date', parse_dates=True)
    else:
        trends_data = fetch_google_trends_data(ticker, start_date, end_date)
        trends_data.to_csv(trends_cache_file)

    data = data.join(trends_data)

    # Drop the Ticker column
    data = data.drop(columns={ticker})

    if columns_to_scale is None or not columns_to_scale:
        columns_to_scale = ['Close']

    scaled_data = data.copy()
    scalers = {}
    for column in columns_to_scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_column = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scaled_data[f'Scaled_{column}'] = scaled_column
        scalers[column] = scaler

    close_prices = data['Close'].values.reshape(-1, 1)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = close_scaler.fit_transform(close_prices)
    scalers['Close'] = close_scaler

    if split_by_date:
        split_date = pd.Timestamp(split_ratio)
    else:
        split_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) * split_ratio

    if split_by_date:
        train_data = scaled_close_prices[data.index < split_date]
        test_data = scaled_close_prices[data.index >= split_date]
    else:
        train_data = scaled_close_prices[:int(len(scaled_close_prices) * split_ratio)]
        test_data = scaled_close_prices[int(len(scaled_close_prices) * split_ratio):]

    if local_file:
        file_path = f"/content/drive/My Drive/Cos30018/{local_file}"
        data.to_csv(file_path)

    return train_data, test_data, scalers, data, scaled_data

def display_custom_table(df, num_rows=5):
    if len(df) <= 2 * num_rows:
        display(df)
    else:
        head = df.head(num_rows)
        tail = df.tail(num_rows)
        ellipsis_row = pd.DataFrame([['...'] * len(df.columns)], columns=df.columns, index=['...'])
        df_display = pd.concat([head, ellipsis_row, tail])
        display(HTML(df_display.to_html(index=True)))

def display_trend_data(trends_data):
    display(trends_data)

def create_dl_model(input_shape, layers_config):
    model = Sequential()
    for i, layer in enumerate(layers_config):
        layer_type = layer.get("type")
        units = layer.get("units", 50)
        activation = layer.get("activation", "relu")
        return_sequences = layer.get("return_sequences", False)
        dropout_rate = layer.get("dropout_rate", 0.0)
        if layer_type == "LSTM":
            if i == 0:
                model.add(LSTM(units, activation=activation, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, activation=activation, return_sequences=return_sequences))
        elif layer_type == "GRU":
            if i == 0:
                model.add(GRU(units, activation=activation, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(GRU(units, activation=activation, return_sequences=return_sequences))
        elif layer_type == "RNN":
            if i == 0:
                model.add(SimpleRNN(units, activation=activation, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(SimpleRNN(units, activation=activation, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def experiment_with_models(train_data, test_data, scaler, layers_configs, epochs=10, batch_size=16):
    results = []
    time_steps = 60
    input_shape = (time_steps, 1)
    X_train, y_train = [], []
    X_test, y_test = [], []
    for i in range(time_steps, len(train_data)):
        X_train.append(train_data[i-time_steps:i, 0])
        y_train.append(train_data[i, 0])
    for i in range(time_steps, len(test_data)):
        X_test.append(test_data[i-time_steps:i, 0])
        y_test.append(test_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], time_steps, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], time_steps, 1))
    for config in layers_configs:
        print(f"Training model with config: {config}")
        model = create_dl_model(input_shape, config['layers'])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
        print(f"Model training completed for config: {config}")
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
        plt.figure(figsize=(14, 5))
        plt.plot(real_stock_price, color='red', label='Real Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
        unique_layers = []
        for layer in config['layers']:
            layer_desc = f"{layer['type']} (units={layer.get('units', 50)})"
            if layer_desc not in unique_layers:
                unique_layers.append(layer_desc)
        model_description = ' - '.join(unique_layers)
        plt.title(f'Stock Price Prediction: {model_description}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
        results.append({
            "config": config,
            "history": history,
            "predicted_stock_price": predicted_stock_price,
            "real_stock_price": real_stock_price
        })
    return results

def fit_arima_model(train_data, test_data, scaler, order=(5,1,0)):
    series = pd.Series(train_data.flatten())
    X = series.values
    history = [x for x in X]
    predictions = list()
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[t, 0]
        history.append(obs)
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

def ensemble_predictions(dl_predictions, arima_predictions, weights=[0.5, 0.5]):
    max_len = max(len(dl_predictions), len(arima_predictions))
    ensemble_preds = np.zeros((max_len, 1))
    for i in range(max_len):
        dl_pred = dl_predictions[i] if i < len(dl_predictions) else 0
        arima_pred = arima_predictions[i] if i < len(arima_predictions) else 0
        ensemble_preds[i] = (weights[0] * dl_pred) + (weights[1] * arima_pred)
    return ensemble_preds

if __name__ == "__main__":
    ticker = 'TSLA'
    start_date = '2016-01-01'
    end_date = '2024-03-20'
    local_file = 'tsla_data.csv'
    columns_to_scale = ["Close"]
    train_data, test_data, scalers, data, scaled_data = load_and_process_data(ticker, start_date, end_date, local_file, split_ratio=0.7, split_by_date=False, columns_to_scale=columns_to_scale)
    display_custom_table(data)

    # Load trend data separately and display it
    trends_cache_file = f"/content/drive/My Drive/Cos30018/{ticker}_trends.csv"
    trends_data = pd.read_csv(trends_cache_file, index_col='date', parse_dates=True)
    display_trend_data(trends_data)

    lstm_config = {
        "layers": [
            {"type": "LSTM", "units": 50, "return_sequences": True},
            {"type": "LSTM", "units": 50, "return_sequences": False},
            {"type": "Dense", "units": 1}
        ]
    }

    gru_config = {
        "layers": [
            {"type": "GRU", "units": 50, "return_sequences": True},
            {"type": "GRU", "units": 50, "return_sequences": False},
            {"type": "Dense", "units": 1}
        ]
    }
    rnn_config = {
        "layers": [
            {"type": "RNN", "units": 50, "return_sequences": True},
            {"type": "RNN", "units": 50, "return_sequences": False},
            {"type": "Dense", "units": 1}
        ]
    }

    layers_configs = [lstm_config, gru_config, rnn_config]

    results = experiment_with_models(train_data, test_data, scalers['Close'], layers_configs, epochs=50, batch_size=16)
    arima_predictions = fit_arima_model(train_data, test_data, scalers['Close'])

    ensemble_preds = ensemble_predictions(results[0]['predicted_stock_price'], arima_predictions)
    plt.figure(figsize=(14, 5))
    plt.plot(scalers['Close'].inverse_transform(test_data), color='red', label='Real Stock Price')
    plt.plot(ensemble_preds, color='green', label='Ensemble Predicted Stock Price (With Google trends)')
    plt.title(f'Ensemble Stock Price Prediction (With Google trends data)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    mse = mean_squared_error(scalers['Close'].inverse_transform(test_data), ensemble_preds)
    print(f"Mean Squared Error of the Ensemble Model: {mse}")