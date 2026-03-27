import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

def get_engine():
    
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    name     = os.getenv("DB_NAME", "market_sentiment")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres123")

    url = f"postgresql://{user}:{password}@{host}:{port}/{name}"

    engine = create_engine(url)
    logger.info("Database connection created")
    return engine


def create_tables():
    
    engine = get_engine()

    with engine.connect() as conn:

        # Table 1: Reddit & News posts with sentiment
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id              SERIAL PRIMARY KEY,
                source          VARCHAR(50),
                title           TEXT,
                sentiment_label VARCHAR(20),
                bullish_score   FLOAT,
                bearish_score   FLOAT,
                neutral_score   FLOAT,
                confidence      FLOAT,
                tickers_str     VARCHAR(200),
                created_at      TIMESTAMP DEFAULT NOW()
            );
        """))

        # Table 2: Live stock prices
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id          SERIAL PRIMARY KEY,
                ticker      VARCHAR(30),
                company     VARCHAR(100),
                price_inr   FLOAT,
                change_inr  FLOAT,
                change_pct  VARCHAR(20),
                volume      BIGINT,
                fetched_at  TIMESTAMP DEFAULT NOW()
            );
        """))

        # Table 3: Ticker sentiment summary
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ticker_sentiment (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(30),
                company         VARCHAR(100),
                total_mentions  INT,
                bullish_count   INT,
                bearish_count   INT,
                neutral_count   INT,
                sentiment_score FLOAT,
                recorded_at     TIMESTAMP DEFAULT NOW()
            );
        """))

        # Table 4: Anomaly alerts
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_alerts (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(30),
                company         VARCHAR(100),
                sentiment_score FLOAT,
                mentions        INT,
                price_anomaly   FLOAT,
                price_change    FLOAT,
                alert_type      VARCHAR(50),
                action          VARCHAR(50),
                created_at      TIMESTAMP DEFAULT NOW()
            );
        """))

        # Table 5: Price anomalies history
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS price_anomalies (
                id           SERIAL PRIMARY KEY,
                ticker       VARCHAR(30),
                date         DATE,
                close_inr    FLOAT,
                return_pct   FLOAT,
                volume_ratio FLOAT,
                anomaly_pct  FLOAT,
                signal       VARCHAR(50),
                created_at   TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.commit()
        logger.info("All tables created successfully!")


def save_sentiment_data(df, source="combined"):
    
    engine = get_engine()

    # Select only columns that exist in our table
    cols = ["title", "sentiment_label", "bullish_score",
            "bearish_score", "neutral_score", "confidence"]

    if "tickers_str" in df.columns:
        cols.append("tickers_str")

    if "source" not in df.columns:
        df = df.copy()
        df["source"] = source

    cols.append("source")
    save_df = df[cols].copy()

    save_df.to_sql("sentiment_data", engine,
                   if_exists="append", index=False)
    logger.info(f"Saved {len(save_df)} sentiment records to database")


def save_stock_prices(df):
    
    engine = get_engine()
    cols   = ["ticker", "company", "price_inr",
              "change_inr", "change_pct", "volume"]
    df[cols].to_sql("stock_prices", engine,
                    if_exists="append", index=False)
    logger.info(f"Saved {len(df)} stock prices to database")


def save_ticker_sentiment(df):
    
    engine = get_engine()
    cols   = ["tickers", "company", "total_mentions",
              "bullish_count", "bearish_count",
              "neutral_count", "sentiment_score"]
    save_df = df[cols].copy()
    save_df = save_df.rename(columns={"tickers": "ticker"})
    save_df.to_sql("ticker_sentiment", engine,
                   if_exists="append", index=False)
    logger.info(f"Saved {len(save_df)} ticker sentiment records")


def save_market_alerts(df):
    
    engine = get_engine()
    df.to_sql("market_alerts", engine,
              if_exists="append", index=False)
    logger.info(f"Saved {len(df)} market alerts to database")


def save_anomalies(df, ticker):
    
    engine    = get_engine()
    anomalies = df[df["is_anomaly"] == True].copy()
    anomalies["ticker"] = ticker

    cols = ["ticker", "date", "close_inr",
            "return_pct", "volume_ratio", "anomaly_pct"]
    anomalies[cols].to_sql("price_anomalies", engine,
                            if_exists="append", index=False)
    logger.info(f"Saved {len(anomalies)} anomalies for {ticker}")


def load_latest_alerts():
    
    engine = get_engine()
    query  = """
        SELECT * FROM market_alerts
        ORDER BY created_at DESC
        LIMIT 50
    """
    return pd.read_sql(query, engine)


def load_sentiment_trend():
    
    engine = get_engine()
    query  = """
        SELECT
            DATE(created_at)    as date,
            sentiment_label,
            COUNT(*)            as count
        FROM sentiment_data
        GROUP BY DATE(created_at), sentiment_label
        ORDER BY date DESC
    """
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE SETUP")
    print("=" * 60)

    # Create all tables
    print("\nCreating tables...")
    create_tables()

    # Load and save all our collected data
    print("\nSaving sentiment data...")
    sentiment_df = pd.read_csv("data/combined_sentiment.csv")
    save_sentiment_data(sentiment_df)

    print("Saving ticker sentiment...")
    ticker_df = pd.read_csv("data/ticker_summary.csv")
    save_ticker_sentiment(ticker_df)

    print("Saving market alerts...")
    alerts_df = pd.read_csv("data/market_alerts.csv")
    save_market_alerts(alerts_df)

    print("Saving stock prices...")
    prices_df = pd.read_csv("data/indian_stock_quotes.csv")
    save_stock_prices(prices_df)

    print("Saving TCS anomalies...")
    anomaly_df = pd.read_csv("data/tcs_anomalies.csv")
    anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])
    save_anomalies(anomaly_df, "TCS.BSE")

    # Verify everything saved
    engine = get_engine()
    tables = ["sentiment_data", "stock_prices",
              "ticker_sentiment", "market_alerts", "price_anomalies"]

    print("\nDatabase summary:")
    for table in tables:
        count = pd.read_sql(
            f"SELECT COUNT(*) as n FROM {table}", engine
        ).iloc[0]["n"]
        print(f"  {table}: {count} rows")

    print("\nDatabase setup complete!")