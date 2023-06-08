"""
This script fetches news data for a given stock symbol using the Alpaca API, performs sentiment analysis on the news content using FinBERT, and stores the results in a pandas DataFrame.

Requirements:
- Python 3.6 or higher
- aiohttp
- pandas
- pandas_market_calendars
- beautifulsoup4
- transformers
- torch

Usage:
python sentiment_analysis.py --symbol <stock_symbol> --ndays <number_of_days>

Example:
python sentiment_analysis.py --symbol AAPL --ndays 2
"""

import os
import re
import asyncio
import argparse
from datetime import datetime, timedelta
import pytz
import torch
import aiohttp
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Compile the regular expression pattern for removing unwanted whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")

# Retrieve Alpaca API keys from environment variables
API_KEY = os.environ["APCA_API_KEY_ID"]
SECRET_KEY = os.environ["APCA_API_SECRET_KEY"]

# Check if CUDA is available and set device to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU is available. GPU: {torch.cuda.get_device_name()}")
else:
    print("GPU is not available. Using CPU.")

# Initialize the tokenizer and model for FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)  # Send the model to the device (GPU if available)

pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)


async def fetch(session, url):
    """
    Fetch data from the given URL using aiohttp.

    Args:
        session: aiohttp.ClientSession object.
        url: URL to fetch data from.

    Returns:
        JSON response from the URL.
    """
    async with session.get(url) as response:
        if response.status == 429:
            raise Exception("Rate limit exceeded")
        elif response.status >= 400:
            raise Exception(f"HTTP error {response.status}")
        return await response.json()


async def fetch_news(session, stock_symbol, num_days, page_token=None):
    """
    Fetch news data for a given stock symbol and number of trading days.

    Args:
        session: aiohttp.ClientSession object.
        stock_symbol: Stock symbol.
        num_days: Number of trading days.
        page_token: Page token for pagination (optional).

    Returns:
        JSON response containing news data.
    """
    start_date = get_trading_date_before_days(num_days * 2)
    end_date = get_trading_date_before_days(1)

    url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&symbols={stock_symbol}&limit=50&include_content=true"
    if page_token:
        url += f"&page_token={page_token}"

    response = await fetch(session, url)
    return response


def clean_text(html_text):
    """
    Clean the input text by removing HTML tags and extra whitespaces.

    Args:
        html_text: HTML text to be cleaned.

    Returns:
        Cleaned text.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()
    return text


def get_trading_date_before_days(days):
    """
    Get the trading date a given number of days before the current date.

    Args:
        days: Number of days.

    Returns:
        Trading date.
    """
    nyse = mcal.get_calendar("NYSE")
    end_date = datetime.now()
    schedule = nyse.schedule(
        start_date=end_date - timedelta(days=days * 2), end_date=end_date
    )
    return schedule.iloc[-days].name.date()


def convert_to_est(time_str):
    """
    Convert time in string format to Eastern Standard Time.

    Args:
        time_str: Time string.

    Returns:
        Time string in Eastern Standard Time format.
    """
    datetime_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    datetime_obj_utc = datetime_obj.replace(tzinfo=pytz.UTC)
    datetime_obj_est = datetime_obj_utc.astimezone(
        pytz.timezone("US/Eastern")
    )
    return datetime_obj_est.isoformat()


def get_finbert_sentiment(text):
    """
    Get sentiment score from FinBERT for the given text.

    Args:
        text: Text to analyze.

    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'.
    """
    inputs = tokenizer.encode_plus(
        text, return_tensors="pt", max_length=512, truncation=True
    )
    inputs = {
        name: tensor.to(device) for name, tensor in inputs.items()
    }  # Send inputs to the device
    outputs = model(**inputs)
    sentiment_scores = (
        outputs.logits.detach().cpu().numpy()[0]
    )  # Transfer logits back to CPU
    sentiment_index = np.argmax(
        sentiment_scores
    )  # 0: negative, 1: neutral, 2: positive

    # Map the numeric value to a sentiment label
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_mapping.get(sentiment_index)

    return sentiment


async def main():
    """
    Main function to parse command line arguments and fetch news data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="The stock symbol")
    parser.add_argument(
        "--ndays",
        required=False,
        default=2,
        help="Number of trading days",
        type=int,
    )
    args = parser.parse_args()

    stock_symbol = args.symbol.upper()
    num_days = args.ndays
    data = []
    page_token = None

    headers = {"Apca-Api-Key-Id": API_KEY, "Apca-Api-Secret-Key": SECRET_KEY}

    # Create a single aiohttp ClientSession
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            response = await fetch_news(
                session, stock_symbol, num_days, page_token
            )
            for item in response["news"]:
                headline = clean_text(item["headline"])
                content = clean_text(item["content"])
                created_at = convert_to_est(item["created_at"])
                if content:
                    sentiment = get_finbert_sentiment(content)
                    data.append([created_at, headline, content, sentiment])

            page_token = response.get("next_page_token")
            if not page_token:
                break

    df = pd.DataFrame(
        data, columns=["created_at", "headline", "content", "sentiment"]
    )
    print(df)


if __name__ == "__main__":
    # Run the main function when the script is executed
    asyncio.run(main())
