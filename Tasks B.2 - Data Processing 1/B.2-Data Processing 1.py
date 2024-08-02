# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install yfinance
# pip install IPython

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display, HTML
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)

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

def display_custom_table(df, num_rows=5):
    """
    Display the first few and last few rows of the DataFrame with ellipses in between.

    Parameters:
    - df: DataFrame to display.
    - num_rows: Number of rows to display from the start and end of the DataFrame.
    """
    if len(df) <= 2 * num_rows:
        # Display the entire DataFrame if it's small enough
        display(df)
    else:
        # Display the first few and last few rows with ellipses in between
        head = df.head(num_rows)
        tail = df.tail(num_rows)
        ellipsis_row = pd.DataFrame([['...'] * len(df.columns)], columns=df.columns, index=['...'])
        df_display = pd.concat([head, ellipsis_row, tail])
        display(HTML(df_display.to_html(index=True)))

# Example usage
if __name__ == "__main__":
    # Choose either to load data from Yahoo Finance or from a local CSV file
    ticker = 'AMZN'
    start_date = '2016-01-01'
    end_date = '2024-05-20'
    local_file = 'amzn_data.csv'

    # Specify columns to scale
    columns_to_scale = ["Close", "Volume"]

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