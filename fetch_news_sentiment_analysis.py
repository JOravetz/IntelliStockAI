import argparse
import os
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
import pytz
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import logging

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

pd.set_option("display.width", None)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def remove_html_tags(text):
    """Remove HTML tags and other special characters from a given text."""
    text = re.sub("<[^>]*>", "", text)
    text = re.sub("\n+", "\n", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
    )
    return text.strip()

def prepare_for_sentiment_analysis(header, content):
    """Prepare the text for sentiment analysis by combining header and content, and tokenizing."""
    combined_text = f"{header} {content}"
    tokens = tokenizer.tokenize(combined_text)
    if len(tokens) > 512:
        tokens = tokens[:512]
    return " ".join(tokens)

def get_trading_date_before_days(days):
    """Get the trading date for the specified number of days before the current date."""
    nyse = mcal.get_calendar("NYSE")
    end_date = datetime.now()
    schedule = nyse.schedule(
        start_date=end_date - timedelta(days=days * 2), end_date=end_date
    )
    return schedule.iloc[-days].name.date()

def compute_sentiment_score(text):
    """Compute sentiment scores (positive, negative, neutral) for the given text."""
    inputs = tokenizer.encode_plus(
        text, return_tensors="pt", max_length=512, truncation="only_first"
    )
    outputs = model(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=1)
    sentiment_scores = scores.tolist()[0]
    pos_score, neg_score, neu_score = sentiment_scores
    return pos_score, neg_score, neu_score

def convert_to_est(date_time):
    """Convert the given datetime to Eastern Standard Time."""
    utc_dt = date_time.replace(tzinfo=pytz.utc)
    est = pytz.timezone("US/Eastern")
    est_dt = utc_dt.astimezone(est)
    return est_dt

async def fetch_news(symbol, api_key, api_secret, ndays):
    base_url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"Apca-Api-Key-Id": api_key, "Apca-Api-Secret-Key": api_secret}

    end_date = datetime.now()
    start_date = get_trading_date_before_days(ndays)

    logger = logging.getLogger(__name__)
    logger.info(f"Start Date: {start_date}, End Date: {end_date.date()}")

    params = {
        "symbols": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
        "sort": "DESC",
        "include_content": int(True),
        "exclude_contentless": int(True),
    }

    df_sentiment = pd.DataFrame(
        columns=["Symbol", "Positive", "Negative", "Neutral", "Headline",]
    )
    df_sentiment.index.name = "Timestamp"

    async def fetch_page(session, url, params):
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_all_pages(params):
        nonlocal df_sentiment
        async with aiohttp.ClientSession() as session:
            while True:
                data = await fetch_page(session, base_url, params)

                news = data.get("news", [])
                logger.info("Fetched {} news articles".format(len(news)))

                for article in news:
                    created_at = datetime.fromisoformat(
                        article["created_at"].rstrip("Z")
                    )
                    created_at_est = convert_to_est(created_at)
                    headline = article["headline"]
                    content = remove_html_tags(article["content"])
                    preprocessed_text = prepare_for_sentiment_analysis(
                        headline, content
                    )
                    sentiment_scores = compute_sentiment_score(preprocessed_text)
                    (
                        positive_score,
                        negative_score,
                        neutral_score,
                    ) = sentiment_scores
                    df_sentiment = pd.concat(
                        [
                            df_sentiment,
                            pd.DataFrame(
                                {
                                    "Symbol": [symbol],
                                    "Positive": [positive_score],
                                    "Negative": [negative_score],
                                    "Neutral": [neutral_score],
                                    "Headline": [headline[:100].ljust(100)],
                                },
                                index=pd.DatetimeIndex(
                                    [created_at_est.replace(second=0, microsecond=0)]
                                ),
                            ),
                        ],
                        ignore_index=False,
                    )

                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    logger.info("No more news articles found.")
                    break

                params["page_token"] = next_page_token

    try:
        await fetch_all_pages(params)
        df_sentiment.index = df_sentiment.index.tz_convert("US/Eastern")
        pd.set_option("display.max_columns", None)
        logger.info(df_sentiment.to_string())
        return df_sentiment
    except Exception as err:
        logger.exception("An error occurred while fetching news: %s", str(err))
        return pd.DataFrame()

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fetch news using the Alpaca market news API"
    )
    parser.add_argument("-s", "--symbol", required=True, help="Stock symbol")
    parser.add_argument(
        "-n",
        "--ndays",
        type=int,
        default=5,
        help="Number of trading days before today (default: 5)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    api_key = os.environ["APCA_API_KEY_ID"]
    api_secret = os.environ["APCA_API_SECRET_KEY"]
    ndays = args.ndays

    asyncio.run(fetch_news(symbol, api_key, api_secret, ndays))
