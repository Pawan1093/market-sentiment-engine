import schedule
import time
from loguru import logger
from dotenv import load_dotenv
import sys
import os

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.reddit_collector  import fetch_all_subreddits, save_to_csv as save_reddit
from src.ingestion.news_collector    import fetch_all_news,       save_to_csv as save_news
from src.ingestion.stock_collector   import fetch_all_indian_quotes, fetch_daily_history
from src.ml.sentiment_analyzer       import FinBERTAnalyzer
from src.ml.ticker_extractor         import add_tickers_to_dataframe, get_ticker_sentiment_summary
from src.ml.anomaly_detector         import AnomalyDetector, cross_reference_sentiment
from src.database.db_manager         import (
    create_tables, save_sentiment_data, save_stock_prices,
    save_ticker_sentiment, save_market_alerts, save_anomalies
)
import pandas as pd

analyzer = None  

def load_analyzer():
    global analyzer
    if analyzer is None:
        logger.info("Loading FinBERT model (one time only)...")
        analyzer = FinBERTAnalyzer()
    return analyzer

def run_pipeline():
   
    logger.info("=" * 50)
    logger.info("PIPELINE STARTED")
    logger.info("=" * 50)

    try:
        
        logger.info("Step 1: Collecting Reddit posts...")
        reddit_posts = fetch_all_subreddits()
        reddit_df    = save_reddit(reddit_posts)

        logger.info("Step 1: Collecting news articles...")
        news_articles = fetch_all_news()
        news_df       = save_news(news_articles)

        logger.info("Step 1: Fetching Indian stock prices...")
        quotes    = fetch_all_indian_quotes()
        prices_df = pd.DataFrame(quotes)
        if not prices_df.empty:
            prices_df.to_csv("data/indian_stock_quotes.csv", index=False)

        
        logger.info("Step 2: Running FinBERT sentiment analysis...")
        nlp = load_analyzer()

        reddit_df  = nlp.analyze_dataframe(reddit_df,  text_column="title")
        news_df    = nlp.analyze_dataframe(news_df,    text_column="title")

        
        logger.info("Step 3: Extracting stock tickers...")
        reddit_df  = add_tickers_to_dataframe(reddit_df, "title")
        news_df    = add_tickers_to_dataframe(news_df,   "title")

        combined_df = pd.concat([reddit_df, news_df], ignore_index=True)
        combined_df.to_csv("data/combined_sentiment.csv", index=False)

        ticker_summary = get_ticker_sentiment_summary(combined_df)
        ticker_summary.to_csv("data/ticker_summary.csv", index=False)

        
        logger.info("Step 4: Running anomaly detection on TCS...")
        history = fetch_daily_history("TCS.BSE")

        alerts_df = pd.DataFrame()
        if history is not None:
            history.to_csv("data/tcs_history.csv", index=False)
            detector   = AnomalyDetector(contamination=0.05)
            trained_df = detector.train(history, ticker="TCS.BSE")
            result_df  = detector.detect(trained_df)
            result_df.to_csv("data/tcs_anomalies.csv", index=False)

            if not ticker_summary.empty:
                alerts_df = cross_reference_sentiment(result_df, ticker_summary)
                alerts_df.to_csv("data/market_alerts.csv", index=False)

        
        logger.info("Step 5: Saving all data to PostgreSQL...")
        save_sentiment_data(combined_df)

        if not prices_df.empty:
            save_stock_prices(prices_df)

        if not ticker_summary.empty:
            save_ticker_sentiment(ticker_summary)

        if not alerts_df.empty:
            save_market_alerts(alerts_df)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Reddit posts: {len(reddit_df)}")
        logger.info(f"News articles: {len(news_df)}")
        logger.info(f"Tickers found: {len(ticker_summary)}")
        logger.info("Next run in 2 hours...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Will retry at next scheduled run")

if __name__ == "__main__":
    print("=" * 50)
    print("MARKET SENTIMENT ENGINE - AUTO PIPELINE")
    print("Runs every 2 hours automatically")
    print("Press Ctrl+C to stop")
    print("=" * 50)

   
    create_tables()

    
    logger.info("Running first pipeline now...")
    run_pipeline()

    
    schedule.every(2).hours.do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)