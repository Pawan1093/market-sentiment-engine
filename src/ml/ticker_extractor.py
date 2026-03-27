import re
import pandas as pd
from loguru import logger


KNOWN_TICKERS = {
    
    "RELIANCE":   "Reliance Industries",
    "TCS":        "Tata Consultancy Services",
    "INFY":       "Infosys",
    "HDFCBANK":   "HDFC Bank",
    "ICICIBANK":  "ICICI Bank",
    "WIPRO":      "Wipro",
    "SBIN":       "State Bank of India",
    "BAJFINANCE": "Bajaj Finance",
    "TATAMOTORS": "Tata Motors",
    "ADANIENT":   "Adani Enterprises",
    "HINDUNILVR": "Hindustan Unilever",
    "MARUTI":     "Maruti Suzuki",
    "SUNPHARMA":  "Sun Pharmaceutical",
    "LTIM":       "LTIMindtree",
    "TECHM":      "Tech Mahindra",
    "AXISBANK":   "Axis Bank",
    "KOTAKBANK":  "Kotak Mahindra Bank",
    "ONGC":       "Oil and Natural Gas Corp",
    "NTPC":       "NTPC Limited",
    "POWERGRID":  "Power Grid Corp",

    
    "NIFTY":      "Nifty 50",
    "SENSEX":     "BSE Sensex",
    "BANKNIFTY":  "Bank Nifty",
}


COMPANY_TO_TICKER = {v.lower(): k for k, v in KNOWN_TICKERS.items()}

def extract_tickers_from_text(text):
    
    if not text:
        return []

    text_str = str(text)
    found    = set()

    
    dollar_tickers = re.findall(r'\$([A-Z]{1,12})', text_str)
    for t in dollar_tickers:
        if t in KNOWN_TICKERS:
            found.add(t)

    
    words = re.findall(r'\b([A-Z]{2,6})\b', text_str)
    for w in words:
        if w in KNOWN_TICKERS:
            found.add(w)

    
    text_lower = text_str.lower()
    for company, ticker in COMPANY_TO_TICKER.items():
        if company in text_lower:
            found.add(ticker)

    return list(found)

def add_tickers_to_dataframe(df, text_column="title"):
  
    df = df.copy()
    df["tickers"] = df[text_column].apply(extract_tickers_from_text)
    df["ticker_count"] = df["tickers"].apply(len)
    df["has_ticker"]   = df["ticker_count"] > 0

    
    df["tickers_str"] = df["tickers"].apply(lambda x: ", ".join(x) if x else "NONE")

    return df

def get_ticker_sentiment_summary(df):
    
    
    df_exploded = df.explode("tickers").dropna(subset=["tickers"])
    df_exploded = df_exploded[df_exploded["tickers"] != ""]

    if df_exploded.empty:
        logger.warning("No tickers found in data")
        return pd.DataFrame()

    summary = df_exploded.groupby("tickers").agg(
        company        = ("tickers", lambda x: KNOWN_TICKERS.get(x.iloc[0], x.iloc[0])),
        total_mentions = ("tickers", "count"),
        bullish_count  = ("sentiment_label", lambda x: (x == "bullish").sum()),
        bearish_count  = ("sentiment_label", lambda x: (x == "bearish").sum()),
        neutral_count  = ("sentiment_label", lambda x: (x == "neutral").sum()),
        avg_confidence = ("confidence", "mean"),
        avg_bullish    = ("bullish_score", "mean"),
        avg_bearish    = ("bearish_score", "mean"),
    ).reset_index()

    
    summary["sentiment_score"] = (
        (summary["bullish_count"] - summary["bearish_count"]) /
        summary["total_mentions"]
    ).round(3)

   
    summary = summary.sort_values("total_mentions", ascending=False)

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("TICKER EXTRACTION ENGINE")
    print("=" * 60)

    
    reddit_df = pd.read_csv("data/reddit_sentiment.csv")
    news_df   = pd.read_csv("data/news_sentiment.csv")

    
    print("\nExtracting tickers from Reddit posts...")
    reddit_df = add_tickers_to_dataframe(reddit_df, "title")

    print("Extracting tickers from News articles...")
    news_df = add_tickers_to_dataframe(news_df, "title")

    
    combined_df = pd.concat([reddit_df, news_df], ignore_index=True)

    
    combined_df.to_csv("data/combined_sentiment.csv", index=False)
    logger.info(f"Saved {len(combined_df)} combined records")

    
    print(f"\nPosts with stock tickers: {combined_df['has_ticker'].sum()}")
    print("\nSample with tickers:")
    ticker_posts = combined_df[combined_df["has_ticker"]]
    print(ticker_posts[["title", "tickers_str", "sentiment_label"]].head(10).to_string())

    
    print("\n" + "=" * 60)
    print("TICKER SENTIMENT SUMMARY")
    print("=" * 60)
    summary = get_ticker_sentiment_summary(combined_df)

    if not summary.empty:
        print(summary[["tickers", "company", "total_mentions",
                       "bullish_count", "bearish_count",
                       "sentiment_score"]].to_string())

    
    summary.to_csv("data/ticker_summary.csv", index=False)
    print("\nSaved ticker summary to data/ticker_summary.csv")