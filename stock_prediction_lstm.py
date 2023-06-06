import os
import asyncio
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_market_calendars as mcal

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from plotting import plot_ohlc_data
from fetch_news_sentiment_analysis import fetch_news
from feature_selection import iterative_feature_selection
from utils import (
    get_bars,
    generate_features,
    normalize_features,
    create_sequences,
    split_data,
)
from model import build_lstm_model

import logging

def load_api_keys():
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    return api_key, api_secret

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fetch historical stock data using Alpaca V2 bars (OHLC)"
    )
    parser.add_argument(
        "-s", "--symbol", type=str, required=True, help="Stock symbol"
    )
    parser.add_argument(
        "-n",
        "--ndays",
        type=int,
        default=5,
        help="Number of trading days from today (default: 5)",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        type=str,
        default="1Min",
        help="Timeframe of the aggregation (default: 1Min)",
    )
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load Alpaca API keys
        api_key, api_secret = load_api_keys()

        # Parse command-line arguments
        args = parse_arguments()

        # Convert the symbol to uppercase
        symbol = args.symbol.upper()

        # Create an instance of the NYSE calendar
        nyse = mcal.get_calendar("NYSE")

        # Calculate the start date based on the number of trading days
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        schedule = nyse.schedule(start_date="2000-01-01", end_date=end_date)
        trading_days = mcal.date_range(schedule, frequency="1D")
        start_date = trading_days[-args.ndays].strftime("%Y-%m-%d")

        # Fetch historical data using the Alpaca API
        timeframe = args.timeframe
        df = get_bars(
            symbol,
            timeframe,
            start_date,
            end_date,
            limit=10000,
            api_key=api_key,
            api_secret=api_secret,
        )
        df_orig = df.copy()

        # Sort the dataframe by the index (timestamp)
        df.sort_index(inplace=True)

        logger.info(df)

        # Generate technical indicators from the fetched historical data
        df_with_features = generate_features(df)

        # Define the list of features to normalize
        features_to_normalize = [
            "Open",
            "High",
            "Low",
            "Close",
            "v",
            "n",
            "vw",
            "MA20",
            "MA50",
            "MA200",
            "RSI",
            "MACD",
            "Signal",
            "UpperBB",
            "MiddleBB",
            "LowerBB",
            "OBV",
            "SlowK",
            "SlowD",
            "ATR",
            "MFI",
            "Tenkan",
            "Kijun",
            "SenkouA",
            "SenkouB",
            "Chikou",
            "PP",
            "ROC",
            "CMF",
            "CCI",
            "PSAR",
            "EMA20",
            "WMA20",
        ]

        normalized_df, scalers = normalize_features(
            df_with_features, features_to_normalize
        )
        close_scaler = scalers["Close"]
        df_normalized_filled = normalized_df.ffill().bfill()

        # Call fetch_news() from the fetch_news_sentiment_analysis script and merge the two DataFrames
        df_sentiment = asyncio.run(
            fetch_news(symbol, api_key, api_secret, args.ndays)
        )

        # Sort both DataFrames by their index
        df_normalized_filled.sort_index(inplace=True)
        df_sentiment.sort_index(inplace=True)

        # Merge the two DataFrames using pd.merge_asof within the specified tolerance (e.g., 1 minute)
        if df_sentiment.empty:
            logger.info("No sentiment data available. Skipping sentiment features.")
            combined_df = df_normalized_filled.copy()
            combined_df["Positive"] = 0
            combined_df["Negative"] = 0
            combined_df["Neutral"] = 0
        else:
            combined_df = pd.merge_asof(
                df_normalized_filled,
                df_sentiment[["Positive", "Negative", "Neutral"]],
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta(minutes=1),
                direction="nearest",
            )

        # Fill NaN values with 0
        # combined_df.fillna(value=0, inplace=True)
        # logger.info(combined_df)

        combined_df[["Positive", "Negative", "Neutral"]] = combined_df[["Positive", "Negative", "Neutral"]].fillna(method='ffill')
        combined_df[["Positive", "Negative", "Neutral"]] = combined_df[["Positive", "Negative", "Neutral"]].fillna(method='bfill')
        logger.info(combined_df)

        # Find the optimal features
        optimal_features = iterative_feature_selection(combined_df)

        # Add the sentiment columns to the optimal_features list
        if not df_sentiment.empty:
            optimal_features = optimal_features.union(
                ["Positive", "Negative", "Neutral"]
            )
            logger.info("Optimal Features (after adding sentiment columns):")
        logger.info(optimal_features)

        # Define target column, selected features, and window size
        target_col = "Close"
        window_size = 60

        X, y = create_sequences(
            combined_df, target_col, optimal_features, window_size
        )

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Build and train the LSTM model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test loss: {test_loss}")

        plt.figure()
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Make predictions
        predictions = model.predict(X_test)

        scaled_predictions = close_scaler.inverse_transform(predictions)
        scaled_y_test = close_scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate and print the error metrics
        mse = mean_squared_error(scaled_y_test, scaled_predictions)
        mae = mean_absolute_error(scaled_y_test, scaled_predictions)
        r2 = r2_score(scaled_y_test, scaled_predictions)

        logger.info(f"Mean Squared Error: {mse:.2f}")
        logger.info(f"Mean Absolute Error: {mae:.2f}")
        logger.info(f"R^2 Score: {r2:.2f}")

        # Calculate the index values for the test set
        test_index = combined_df.index[-len(scaled_y_test) :]

        # Create a DataFrame with the actual and predicted close prices
        comparison_df = pd.DataFrame(
            {
                "Actual": scaled_y_test.flatten(),
                "Predicted": scaled_predictions.flatten(),
            },
            index=test_index,
        )

        # Extract dates for positive and negative news
        positive_news_dates = df_sentiment[df_sentiment['Positive'] > 0.5].index.to_numpy()
        negative_news_dates = df_sentiment[df_sentiment['Negative'] > 0.5].index.to_numpy()

        # Get close prices for those dates
        positive_news_prices = df.asof(positive_news_dates)['Close'].to_numpy()
        negative_news_prices = df.asof(negative_news_dates)['Close'].to_numpy()

        # Extract actual close prices for the entire dataset
        full_dates = df_orig.index.to_numpy()
        full_close_prices = df_orig["Close"].to_numpy()

        # Plot the actual and predicted close prices
        plt.figure(figsize=(16, 8))
        plt.plot(
            full_dates,
            full_close_prices,
            label="Full Actual Close Prices",
            color="black",
            alpha=0.5,
        )

        # Plot only the portion of actual prices that are in the test set
        plt.plot(
            comparison_df.index.to_numpy(),
            comparison_df["Actual"].to_numpy(),
            label="Test Actual Close Prices",
            color="blue",
            alpha=0.5,
        )
        plt.plot(
            comparison_df.index.to_numpy(),
            comparison_df["Predicted"].to_numpy(),
            label="Predicted Close Prices",
            color="red",
            alpha=0.5,
        )

        # Add markers for news articles
        plt.scatter(positive_news_dates, positive_news_prices, color='green', marker='*', label='Positive news', s=70)
        plt.scatter(negative_news_dates, negative_news_prices, color='red', marker='o', label='Negative news', s=70)

        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title(f"{symbol} - Actual vs Predicted Close Prices")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        # Print the comparison DataFrame
        logger.info(comparison_df)

        # Call the plot_ohlc_data function with the DataFrame
        if args.timeframe == "1Day":
            plot_ohlc_data(df_orig, symbol)

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))

if __name__ == "__main__":
    main()
