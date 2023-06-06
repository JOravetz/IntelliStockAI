import os
import pytz
import talib
import requests
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=None, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def fetch_all_pages(
    symbol, timeframe, start, end, limit, api_key, api_secret
):
    # Modify fetch_page to only return the DataFrame
    def fetch_page(next_page_token=None):
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        params = {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": limit,
            "adjustment": "split",
        }

        if next_page_token:
            params["page_token"] = next_page_token

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            bars = data["bars"]
            next_page_token = data.get("next_page_token")
            df = pd.DataFrame(bars)
            df.set_index("t", inplace=True)
            df.index = pd.to_datetime(df.index).tz_convert("US/Eastern")
            df.columns = ["Open", "High", "Low", "Close", "v", "n", "vw"]

            return df, next_page_token
        else:
            raise Exception(f"Error fetching bars data: {response.text}")

    def worker():
        while True:
            next_page_token = task_queue.get()
            if next_page_token is None:
                break

            df, new_next_page_token = fetch_page(next_page_token)
            if not df.empty:
                dfs.append(df)

            if new_next_page_token:
                task_queue.put(new_next_page_token)

            task_queue.task_done()

    # Fetch the first page to determine the first next_page_token
    df, next_page_token = fetch_page()
    dfs = [df]

    if not next_page_token:
        return pd.concat(dfs, axis=0)

    task_queue = Queue()
    task_queue.put(next_page_token)

    num_workers = 5
    threads = []
    for _ in range(num_workers):
        t = Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    for _ in range(num_workers):
        task_queue.put(None)

    for t in threads:
        t.join()

    return pd.concat(dfs, axis=0)


# Update the get_bars function to use fetch_all_pages
def get_bars(symbol, timeframe, start, end, limit, api_key, api_secret):
    df = fetch_all_pages(
        symbol, timeframe, start, end, limit, api_key, api_secret
    )
    return df


def ichimoku_cloud(df, periods=(9, 26, 52)):
    high, low = df["High"], df["Low"]
    period1, period2, period3 = periods

    conversion_line = (
        high.rolling(period1).max() + low.rolling(period1).min()
    ) / 2
    base_line = (high.rolling(period2).max() + low.rolling(period2).min()) / 2

    lead_a = ((conversion_line + base_line) / 2).shift(period2)
    lead_b = (
        (high.rolling(period3).max() + low.rolling(period3).min()) / 2
    ).shift(period2)

    lagging_line = df["Close"].shift(-period2)

    return conversion_line, base_line, lead_a, lead_b, lagging_line


def pivot_points(df):
    high, low, close = df["High"], df["Low"], df["Close"]

    pivot_point = (high + low + close) / 3
    resistance1 = 2 * pivot_point - low
    support1 = 2 * pivot_point - high
    resistance2 = pivot_point + (high - low)
    support2 = pivot_point - (high - low)
    resistance3 = high + 2 * (pivot_point - low)
    support3 = low - 2 * (high - pivot_point)

    return pd.DataFrame(
        {
            "Pivot": pivot_point,
            "R1": resistance1,
            "S1": support1,
            "R2": resistance2,
            "S2": support2,
            "R3": resistance3,
            "S3": support3,
        },
        index=df.index,
    )


def generate_features(df):
    # Calculate Moving Averages
    for period in [20, 50, 200]:
        df[f"MA{period}"] = talib.SMA(df["Close"], timeperiod=period)

    # Calculate Relative Strength Index (RSI)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)

    # Calculate Moving Average Convergence Divergence (MACD)
    macd, signal, _ = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["Signal"] = signal

    # Calculate Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["UpperBB"] = upper
    df["MiddleBB"] = middle
    df["LowerBB"] = lower

    # Calculate On-Balance Volume (OBV)
    df["OBV"] = talib.OBV(df["Close"], df["v"])

    # Calculate Stochastic Oscillator
    slowk, slowd = talib.STOCH(
        df["High"],
        df["Low"],
        df["Close"],
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df["SlowK"] = slowk
    df["SlowD"] = slowd

    # Calculate Average True Range (ATR)
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

    # Calculate Money Flow Index (MFI)
    df["MFI"] = talib.MFI(
        df["High"], df["Low"], df["Close"], df["v"], timeperiod=14
    )

    # Calculate Ichimoku Cloud
    tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(
        df, periods=(9, 26, 52)
    )
    df["Tenkan"] = tenkan
    df["Kijun"] = kijun
    df["SenkouA"] = senkou_a
    df["SenkouB"] = senkou_b
    df["Chikou"] = chikou

    # Calculate Pivot Points
    pivot_df = pivot_points(df)
    df["PP"] = pivot_df["Pivot"]

    # Calculate Rate of Change (ROC)
    df["ROC"] = talib.ROC(df["Close"], timeperiod=10)

    # Calculate Chaikin Money Flow (CMF)
    df["CMF"] = talib.ADOSC(
        df["High"],
        df["Low"],
        df["Close"],
        df["v"],
        fastperiod=3,
        slowperiod=10,
    )

    # Calculate Commodity Channel Index (CCI)
    df["CCI"] = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=14)

    # Calculate Parabolic SAR (PSAR)
    df["PSAR"] = talib.SAR(
        df["High"], df["Low"], acceleration=0.02, maximum=0.2
    )

    # Calculate more moving average types like Exponential Moving Average (EMA), and Weighted Moving Average (WMA)
    df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
    df["WMA20"] = talib.WMA(df["Close"], timeperiod=20)

    return df


def normalize_features(df, features_to_normalize):
    df_normalized = df.copy()
    scalers = {}

    for feature in features_to_normalize:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df_normalized[[feature]])
        df_normalized[feature] = scaled_values
        scalers[feature] = scaler

    return df_normalized, scalers


def create_sequences(df, target_col, selected_features, window_size):
    data = df[selected_features].values
    target = df[target_col].shift(-window_size).dropna().values

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(target[i])

    return np.array(X), np.array(y)
