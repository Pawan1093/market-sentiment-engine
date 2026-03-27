import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger
import os

class AnomalyDetector:


    def __init__(self, contamination=0.05):
        # contamination = expected % of anomalies (5% = 1 in 20 days is unusual)
        self.model       = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler      = StandardScaler()
        self.is_trained  = False
        self.ticker      = None

    def prepare_features(self, df):
       
        df = df.copy()

        
        df["return_pct"]     = df["close_inr"].pct_change() * 100

        
        df["volume_ma20"]    = df["volume"].rolling(20).mean()
        df["volume_ratio"]   = df["volume"] / df["volume_ma20"]

        
        df["price_range_pct"]= ((df["high_inr"] - df["low_inr"]) / df["close_inr"]) * 100

        
        df["momentum_5d"]    = df["close_inr"].pct_change(5) * 100

        
        df["ma20"]           = df["close_inr"].rolling(20).mean()
        df["ma20_deviation"] = ((df["close_inr"] - df["ma20"]) / df["ma20"]) * 100

        
        df = df.dropna().reset_index(drop=True)

        return df

    def train(self, df, ticker="UNKNOWN"):
        
        self.ticker  = ticker
        df           = self.prepare_features(df)

        feature_cols = [
            "return_pct",
            "volume_ratio",
            "price_range_pct",
            "momentum_5d",
            "ma20_deviation"
        ]

        X = df[feature_cols].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest
        self.model.fit(X_scaled)
        self.is_trained  = True
        self.feature_cols= feature_cols
        self.trained_df  = df

        logger.info(f"Anomaly detector trained on {len(df)} days of {ticker} data")
        return df

    def detect(self, df):
        
        if not self.is_trained:
            raise Exception("Model not trained yet! Call train() first.")

        df       = self.prepare_features(df)
        X        = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Predict: -1 = anomaly, 1 = normal
        predictions    = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        df["is_anomaly"]     = predictions == -1
        df["anomaly_score"]  = anomaly_scores
        
        df["anomaly_pct"]    = (
            (anomaly_scores - anomaly_scores.min()) /
            (anomaly_scores.max() - anomaly_scores.min())
        ) * 100
        df["anomaly_pct"]    = 100 - df["anomaly_pct"]  

        anomaly_count = df["is_anomaly"].sum()
        logger.info(f"Found {anomaly_count} anomalies in {len(df)} days for {self.ticker}")

        return df

    def get_anomaly_summary(self, df):
        """Get a clean summary of detected anomalies"""
        anomalies = df[df["is_anomaly"]].copy()

        if anomalies.empty:
            return pd.DataFrame()

        summary = anomalies[[
            "date", "close_inr", "return_pct",
            "volume_ratio", "anomaly_pct"
        ]].copy()

        summary["signal"] = summary["return_pct"].apply(
            lambda x: "PRICE SPIKE UP 🚀" if x > 2
            else ("PRICE DROP 📉" if x < -2
            else "VOLUME SPIKE 📊")
        )

        return summary.sort_values("anomaly_pct", ascending=False)


def cross_reference_sentiment(anomaly_df, sentiment_summary):
   
    alerts = []

    for _, row in sentiment_summary.iterrows():
        ticker          = row["tickers"]
        sentiment_score = row["sentiment_score"]
        mentions        = row["total_mentions"]

        # Check if this ticker has price anomalies
        ticker_anomalies = anomaly_df[anomaly_df["is_anomaly"] == True]

        if not ticker_anomalies.empty:
            latest_anomaly = ticker_anomalies.iloc[-1]

            alert_type = "CONFIRMATION"
            if abs(sentiment_score) > 0.3 and mentions >= 2:
                alert_type = "EARLY WARNING"

            alerts.append({
                "ticker":          ticker,
                "company":         row["company"],
                "sentiment_score": round(sentiment_score, 3),
                "mentions":        mentions,
                "price_anomaly":   round(latest_anomaly["anomaly_pct"], 1),
                "price_change":    round(latest_anomaly["return_pct"], 2),
                "alert_type":      alert_type,
                "action":          "BUY SIGNAL 🟢" if sentiment_score > 0.3
                                   else ("SELL SIGNAL 🔴" if sentiment_score < -0.3
                                   else "WATCH 🟡")
            })

    return pd.DataFrame(alerts)


if __name__ == "__main__":
    print("=" * 60)
    print("ANOMALY DETECTION ENGINE")
    print("=" * 60)

    # Load TCS historical data
    history_path = "data/tcs_history.csv"
    if not os.path.exists(history_path):
        print("No history data found. Run stock_collector.py first!")
        exit()

    df = pd.read_csv(history_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\nLoaded {len(df)} days of TCS price history")

    # Train and detect anomalies
    detector    = AnomalyDetector(contamination=0.05)
    trained_df  = detector.train(df, ticker="TCS.BSE")
    result_df   = detector.detect(trained_df)

    # Show anomalies
    summary = detector.get_anomaly_summary(result_df)

    print(f"\nAnomalies detected: {len(summary)}")
    print("\nTop anomalous days for TCS:")
    print(summary.head(10).to_string())

    # Save results
    result_df.to_csv("data/tcs_anomalies.csv", index=False)
    print("\nSaved to data/tcs_anomalies.csv")

    # Cross reference with sentiment
    print("\n" + "=" * 60)
    print("CROSS-REFERENCING WITH SENTIMENT DATA")
    print("=" * 60)

    sentiment_path = "data/ticker_summary.csv"
    if os.path.exists(sentiment_path):
        sentiment_df = pd.read_csv(sentiment_path)
        alerts       = cross_reference_sentiment(result_df, sentiment_df)

        if not alerts.empty:
            print("\nMARKET ALERTS GENERATED:")
            print(alerts.to_string())
            alerts.to_csv("data/market_alerts.csv", index=False)
            print("\nSaved alerts to data/market_alerts.csv")
        else:
            print("\nNo cross-referenced alerts yet")
            print("(Need more Indian ticker mentions in news/reddit data)")
    else:
        print("No sentiment summary found. Run ticker_extractor.py first!")