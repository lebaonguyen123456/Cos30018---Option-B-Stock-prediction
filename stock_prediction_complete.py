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
import plotly.graph_objects as go
import plotly.express as px

# Mount Google Drive
drive.mount('/content/drive')

warnings.filterwarnings("ignore")  # Suppress warnings

# Task 7 - Extension
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

# Task 2 - Data processing 1
def load_and_process_data(ticker, start_date, end_date, local_file=None, split_ratio=0.7, split_by_date=False, columns_to_scale=None):
    """
    Load and preprocess stock market data.

    Parameters:
    - ticker: Stock ticker symbol for downloading data from Yahoo Finance.
    - start_date: Start date for downloading data.
    - end_date: End date for downloading data.
    - local_file: Path to a local CSV file containing stock data.
    - split_ratio: Ratio for splitting data into training and testing sets.
    - split_by_date: If True, split by date instead of ratio.
    - columns_to_scale: List of columns to scale. Default is ['Close'].

    Returns:
    - train_data: Scaled training data.
    - test_data: Scaled testing data.
    - scalers: Dictionary of scalers used for normalization.
    - data: Original stock data.
    - scaled_data: Scaled stock data.
    """
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the index is a DateTimeIndex
    data.index = pd.to_datetime(data.index)

    # Fill NaN values with previous values
    data.fillna(method='ffill', inplace=True)

    # Add Google Trends data (If we don't want to add another trend, we can comment it to disable it)
    trends_cache_file = f"/content/drive/My Drive/Cos30018/{ticker}_trends.csv"
    if local_file and os.path.exists(trends_cache_file):
        trends_data = pd.read_csv(trends_cache_file, index_col='date', parse_dates=True)
    else:
        trends_data = fetch_google_trends_data(ticker, start_date, end_date)
        trends_data.to_csv(trends_cache_file)

    data = data.join(trends_data)

    # Drop the Ticker column
    data = data.drop(columns={ticker})

    # Sanity check: Ensure high is not less than low
    if (data['High'] < data['Low']).any():
        raise ValueError("Inconsistent data: High value is less than Low value for some periods.")

    # Default to scaling the 'Close' column if no columns are specified
    if columns_to_scale is None or not columns_to_scale:
        columns_to_scale = ['Close']

    # Create a DataFrame for scaled data
    scaled_data = data.copy()
    scalers = {}

    # Scale specified columns
    for column in columns_to_scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_column = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scaled_data[f'Scaled_{column}'] = scaled_column
        scalers[column] = scaler

    # Extract close prices and scale them
    close_prices = data['Close'].values.reshape(-1, 1)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = close_scaler.fit_transform(close_prices)
    scalers['Close'] = close_scaler

    # Determine split date based on split_ratio or split_by_date
    if split_by_date:
        split_date = pd.Timestamp(split_ratio)
    else:
        split_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) * split_ratio

    # Split data into train and test sets
    if split_by_date:
        train_data = scaled_close_prices[data.index < split_date]
        test_data = scaled_close_prices[data.index >= split_date]
    else:
        train_data = scaled_close_prices[:int(len(scaled_close_prices) * split_ratio)]
        test_data = scaled_close_prices[int(len(scaled_close_prices) * split_ratio):]

    # Save data to a local file, replacing any existing file
    if local_file:
        file_path = f"/content/drive/My Drive/Cos30018/{local_file}"  # Change to your desired path in Google Drive
        data.to_csv(file_path)

    return train_data, test_data, scalers, data, scaled_data

# Task 2 - Data processing 1
def display_custom_table(df, num_rows=5):
    if len(df) <= 2 * num_rows:
        display(df)
    else:
        head = df.head(num_rows)
        tail = df.tail(num_rows)
        ellipsis_row = pd.DataFrame([['...'] * len(df.columns)], columns=df.columns, index=['...'])
        df_display = pd.concat([head, ellipsis_row, tail])
        display(HTML(df_display.to_html(index=True)))

# Task 7 - Extension
def display_trend_data(trends_data):
    display(trends_data)

# Task 3 - Data processing 2
def candlestick_chart(data, title='Candlestick chart', n_days=1, candle_width=0.8):
    # Resample data if n_days > 1, aggregating over n trading days
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        data_resampled = data

    # Create candlestick chart using Plotly
    fig = go.Figure(data=[
        go.Candlestick(
            x=data_resampled.index,
            open=data_resampled['Open'],
            high=data_resampled['High'],
            low=data_resampled['Low'],
            close=data_resampled['Close'],
            name=title,
            increasing_line_width=2,
            decreasing_line_width=2,
            increasing=dict(line=dict(width=candle_width)),
            decreasing=dict(line=dict(width=candle_width))
        ),
        go.Bar(
            x=data_resampled.index,
            y=data_resampled['Volume'],
            name='Volume',
            marker_color='blue',
            yaxis='y2',
            opacity=0.3
        )
    ])

    # Set y-axis range with some padding
    price_range = data_resampled['High'].max() - data_resampled['Low'].min()
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis=dict(
            range=[
                data_resampled['Low'].min() - price_range * 0.05,
                data_resampled['High'].max() + price_range * 0.05
            ]
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis_rangeslider_visible=True
    )

    # Add range selector buttons
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    fig.show()

# Task 3 - Data processing 2
def boxplot_chart(data, title='Boxplot chart', window_size1=20, window_size2=40):
    fig = go.Figure()

    # Create boxplot for specified rolling windows
    for window_size, title_suffix in zip([window_size1, window_size2], [f"{window_size1} Day Rolling Window", f"{window_size2} Day Rolling Window"]):
        rolling_data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].rolling(window=window_size).mean().dropna()
        for col in rolling_data.columns:
            fig.add_trace(go.Box(
                y=rolling_data[col],
                name=f"{col} ({title_suffix})",
                boxmean=True,
                hovertemplate=(
                    f"<b>{col} ({title_suffix})</b><br>"
                    "Max: %{y}<br>"
                    "Q3: %{upperfence}<br>"
                    "Median: %{med}<br>"
                    "Q1: %{lowerfence}<br>"
                    "Min: %{min}<extra></extra>"
                )
            ))

    # Update layout for the boxplot
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Columns'
    )

    fig.show()

# Task 4 - Machine Learning 1
def create_dl_model(input_shape, layers_config):
    """
    Create a deep learning model based on the provided configuration.

    Parameters:
    - input_shape: Shape of the input data.
    - layers_config: List of dictionaries where each dictionary specifies the type and parameters of a layer.

    Returns:
    - model: Compiled Keras model.
    """
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

# Task 4 - Machine Learning 1
def experiment_with_models(train_data, test_data, scaler, layers_configs, epochs=50, batch_size=32):
    """
    Experiment with different DL networks and configurations.

    Parameters:
    - train_data: Scaled training data.
    - test_data: Scaled testing data.
    - scaler: Scaler used for normalization.
    - layers_configs: List of different configurations to test.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    """
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
            "model": model,  # Include the model in the results dictionary
            "config": config,
            "history": history,
            "predicted_stock_price": predicted_stock_price,
            "real_stock_price": real_stock_price
        })
    return results

# Task 5 - Machine Learning 2
def multistep_prediction(model, data, scaler, k=5):
    time_steps = 60
    input_data = data[-time_steps:]  # Select the last 'time_steps' data points
    predictions = []

    for _ in range(k):
        input_data_reshaped = input_data.reshape((1, time_steps, 1))
        next_prediction = model.predict(input_data_reshaped)
        predictions.append(next_prediction[0, 0])
        input_data = np.append(input_data, next_prediction, axis=0)
        input_data = input_data[1:]  # Slide the window forward by one step

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Task 5 - Machine Learning 2
def multivariate_prediction(model, data, scalers, feature_scalers=None, future_day=1):
    # Define the number of time steps to consider in the input data for the prediction
    time_steps = 60

    # Select the last 'time_steps' number of rows from the data
    # This forms the initial input window for the prediction
    input_data = data[-time_steps:]

    # Initialize a list to store the predictions for each future day
    predictions = []

    # Loop to make predictions for the specified number of future days
    for _ in range(future_day):
        # Reshape the input data to match the expected input shape for the model
        # Shape (1, time_steps, number_of_features)
        input_data_reshaped = input_data.reshape((1, time_steps, input_data.shape[1]))

        # Make a prediction using the model
        prediction = model.predict(input_data_reshaped)

        # If feature scalers are provided, use them to inverse transform the prediction
        if feature_scalers:
            prediction = feature_scalers["Close"].inverse_transform(prediction)
        else:
            # Otherwise, use the provided scalers to inverse transform the prediction
            prediction = scalers["Close"].inverse_transform(prediction)

        # Append the prediction (for the 'Close' value) to the list of predictions
        predictions.append(prediction[0])

        # Create the next row to append to the input data
        # Initialize an array of zeros with the same number of features as input_data
        next_row = np.zeros(input_data.shape[1])

        # Set the first element (assumed to be 'Close' value) to the predicted value
        next_row[0] = prediction[0, 0]

        # Shift the input data window by removing the first row and appending the new prediction
        input_data = np.append(input_data[1:], next_row.reshape(1, -1), axis=0)

    # Return the array of predictions
    return np.array(predictions)

# Task 5 - Machine Learning 2
def multistep_multivariate_prediction(model, data, scalers, feature_scalers=None, k=5):
    time_steps = 60
    input_data = data[-time_steps:]
    predictions = []

    for _ in range(k):
        input_data_reshaped = input_data.reshape((1, time_steps, input_data.shape[1]))
        next_prediction = model.predict(input_data_reshaped)

        if feature_scalers:
            next_prediction = feature_scalers["Close"].inverse_transform(next_prediction)
        else:
            next_prediction = scalers["Close"].inverse_transform(next_prediction)

        predictions.append(next_prediction[0])

        # Create the next row with the predicted Close value
        next_row = np.zeros(input_data.shape[1])
        next_row[0] = next_prediction[0, 0]

        # Shift the input data window
        input_data = np.append(input_data[1:], next_row.reshape(1, -1), axis=0)

    return np.array(predictions)

# Task 6 - Machine Learning 3
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

# Task 6 - Machine Learning 3
def ensemble_predictions(dl_predictions, arima_predictions, weights=[0.5, 0.5]):
    max_len = max(len(dl_predictions), len(arima_predictions))
    ensemble_preds = np.zeros((max_len, 1))
    for i in range(max_len):
        dl_pred = dl_predictions[i] if i < len(dl_predictions) else 0
        arima_pred = arima_predictions[i] if i < len(arima_predictions) else 0
        ensemble_preds[i] = (weights[0] * dl_pred) + (weights[1] * arima_pred)
    return ensemble_preds

# Example usage
if __name__ == "__main__":
    ticker = 'AMZN'
    start_date = '2016-01-01'
    end_date = '2024-03-20'
    local_file = 'amzn_data.csv'

    # Specify columns to scale
    columns_to_scale = ["Volume"]

    # Load and process data
    train_data, test_data, scalers, original_data, scaled_data = load_and_process_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        local_file=local_file,
        split_ratio=0.7,
        columns_to_scale=columns_to_scale
    )

    # Display the original and scaled data tables
    display_custom_table(scaled_data, num_rows=5)

    # Load trend data separately and display it
    trends_cache_file = f"/content/drive/My Drive/Cos30018/{ticker}_trends.csv"
    trends_data = pd.read_csv(trends_cache_file, index_col='date', parse_dates=True)
    display_trend_data(trends_data)

    # Task 3 - Data Processing 2 (code to run)

    # Plot candlestick chart
    candlestick_chart(original_data[(original_data.index >= start_date) & (original_data.index <= end_date)], title=f'{ticker} Candlestick Chart', n_days=7, candle_width=5)  # Use larger n_days to make each candlestick larger

    # Plot boxplot chart
    boxplot_chart(original_data[(original_data.index >= start_date) & (original_data.index <= end_date)], title=f'{ticker} Boxplot Chart', window_size1=20, window_size2=40)

    # Task 4 - Machine learning 1 (code to run)

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

    results = experiment_with_models(train_data, test_data, scalers["Close"], layers_configs, epochs=50, batch_size=32)

    # Task 5 - Machine learning 2 (code to run)
    key_value = 6 # You can change this to any number of future steps you want to predict

    # Multistep prediction example
    multistep_preds = multistep_prediction(results[0]['model'], test_data, scalers["Close"], k=key_value)
    print("Multistep Predictions for next 5 days:", multistep_preds)

    # Multivariate prediction example
    multivariate_preds = multivariate_prediction(results[0]['model'], test_data, scalers, feature_scalers=scalers, future_day=1)
    print("Multivariate Prediction for the next day:", multivariate_preds)

    # Multistep multivariate prediction example
    multistep_multivariate_preds = multistep_multivariate_prediction(results[0]['model'], test_data, scalers, feature_scalers=scalers, k=key_value)
    print("Multistep Multivariate Predictions for next 5 days:", multistep_multivariate_preds)




    # Task 6 - Machine Learning 3 (Code to run) (Try to comment the Task 7 - Extension to run it)
    print("Fitting ARIMA model...")
    best_dl_model = results[0]  # Assuming the first model is the best performing for simplicity
    dl_predictions = best_dl_model["predicted_stock_price"]
    real_stock_price = best_dl_model["real_stock_price"]

    arima_predictions = fit_arima_model(train_data, test_data, scalers['Close'], order=(5, 1, 0))
    arima_predictions = arima_predictions[:len(real_stock_price)]  # Trim ARIMA predictions to match the length

    ensemble_preds = ensemble_predictions(dl_predictions, arima_predictions, weights=[0.7, 0.3])

    plt.figure(figsize=(14, 5))
    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(dl_predictions, color='blue', label='LTSM Predictions')
    plt.plot(arima_predictions, color='green', label='ARIMA Predictions')
    plt.plot(ensemble_preds, color='purple', label='Ensemble Predictions')
    plt.title('Stock Price Prediction: LSTM vs ARIMA vs Ensemble')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


    # Task 7 - Extension (Code to run) (Try to comment the Task 6 - Machine Learning 3 to run it)
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