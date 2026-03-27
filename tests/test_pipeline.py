import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.ticker_extractor  import extract_tickers_from_text
from src.ml.anomaly_detector  import AnomalyDetector

class TestTickerExtractor:
    

    def test_extracts_dollar_ticker(self):
        text   = "I bought $RELIANCE today"
        result = extract_tickers_from_text(text)
        assert "RELIANCE" in result

    def test_extracts_company_name(self):
        text   = "Infosys reported strong quarterly earnings"
        result = extract_tickers_from_text(text)
        assert "INFY" in result

    def test_extracts_multiple_tickers(self):
        text   = "TCS and Wipro both beat estimates"
        result = extract_tickers_from_text(text)
        assert "TCS"   in result
        assert "WIPRO" in result

    def test_empty_text_returns_empty(self):
        result = extract_tickers_from_text("")
        assert result == []

    def test_no_ticker_in_random_text(self):
        text   = "The weather is nice today"
        result = extract_tickers_from_text(text)
        assert result == []


class TestAnomalyDetector:
    

    def get_sample_df(self):
        
        import numpy as np
        dates  = pd.date_range("2025-01-01", periods=80)
        prices = [2400 + i + np.random.normal(0, 20) for i in range(80)]
        
        prices[40] = 1800  

        return pd.DataFrame({
            "date":      dates,
            "open_inr":  prices,
            "high_inr":  [p + 30  for p in prices],
            "low_inr":   [p - 30  for p in prices],
            "close_inr": prices,
            "volume":    [200000  + np.random.randint(-50000, 50000)
                          for _ in range(80)],
        })

    def test_detector_trains_without_error(self):
        df       = self.get_sample_df()
        detector = AnomalyDetector(contamination=0.05)
        result   = detector.train(df, ticker="TEST")
        assert detector.is_trained == True

    def test_detector_finds_anomalies(self):
        df       = self.get_sample_df()
        detector = AnomalyDetector(contamination=0.05)
        trained  = detector.train(df, ticker="TEST")
        result   = detector.detect(trained)
        assert "is_anomaly"    in result.columns
        assert "anomaly_score" in result.columns
        assert result["is_anomaly"].sum() > 0

    def test_anomaly_score_is_between_0_and_100(self):
        df       = self.get_sample_df()
        detector = AnomalyDetector(contamination=0.05)
        trained  = detector.train(df, ticker="TEST")
        result   = detector.detect(trained)
        assert result["anomaly_pct"].min() >= 0
        assert result["anomaly_pct"].max() <= 100