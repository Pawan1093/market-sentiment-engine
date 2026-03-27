import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import time

class FinBERTAnalyzer:
   

    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer   = None
        self.model       = None
        self.labels      = ["bearish", "bullish", "neutral"]  
        self._load_model()

    def _load_model(self):
        
        logger.info("Loading FinBERT model ...")
        logger.info("downloading FinBERT model")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()  

        logger.info("FinBERT model loaded successfully!")

    def analyze_text(self, text):
        
        if not text or len(str(text).strip()) < 5:
            return {
                "label":      "neutral",
                "bullish":    0.33,
                "bearish":    0.33,
                "neutral":    0.34,
                "confidence": 0.34
            }

        
        text = str(text)[:512]

        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

      
        with torch.no_grad():
            outputs = self.model(**inputs)

        
        probs = torch.softmax(outputs.logits, dim=1)[0]

        bearish_score = float(probs[0])
        bullish_score = float(probs[1])
        neutral_score = float(probs[2])

       
        max_idx = int(torch.argmax(probs))
        label   = self.labels[max_idx]

        return {
            "label":      label,
            "bullish":    round(bullish_score, 4),
            "bearish":    round(bearish_score, 4),
            "neutral":    round(neutral_score, 4),
            "confidence": round(float(probs[max_idx]), 4)
        }

    def analyze_batch(self, texts, batch_size=16):
        
        results  = []
        total    = len(texts)

        logger.info(f"Analyzing {total} texts for sentiment...")

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                result = self.analyze_text(text)
                results.append(result)

            # Show progress
            done = min(i + batch_size, total)
            logger.info(f"Progress: {done}/{total} texts analyzed")

        return results

    def analyze_dataframe(self, df, text_column):
        
        texts   = df[text_column].fillna("").tolist()
        results = self.analyze_batch(texts)

        result_df = pd.DataFrame(results)

        df = df.copy()
        df["sentiment_label"]  = result_df["label"].values
        df["bullish_score"]    = result_df["bullish"].values
        df["bearish_score"]    = result_df["bearish"].values
        df["neutral_score"]    = result_df["neutral"].values
        df["confidence"]       = result_df["confidence"].values

        return df


def run_sentiment_on_reddit():
    
    logger.info("Loading Reddit data...")
    df = pd.read_csv("data/reddit_raw.csv")
    logger.info(f"Loaded {len(df)} Reddit posts")

    analyzer    = FinBERTAnalyzer()
    analyzed_df = analyzer.analyze_dataframe(df, text_column="title")

   
    analyzed_df.to_csv("data/reddit_sentiment.csv", index=False)
    logger.info("Saved to data/reddit_sentiment.csv")

    return analyzed_df


def run_sentiment_on_news():
    
    logger.info("Loading news data...")
    df = pd.read_csv("data/news_raw.csv")
    logger.info(f"Loaded {len(df)} news articles")

    analyzer    = FinBERTAnalyzer()
    analyzed_df = analyzer.analyze_dataframe(df, text_column="title")

    
    analyzed_df.to_csv("data/news_sentiment.csv", index=False)
    logger.info("Saved to data/news_sentiment.csv")

    return analyzed_df


if __name__ == "__main__":
    print("=" * 60)
    print("FINBERT SENTIMENT ANALYSIS ENGINE")
    print("=" * 60)

    # Analyze Reddit posts
    print("\nStep 1: Analyzing Reddit posts...")
    reddit_df = run_sentiment_on_reddit()

    print("\nReddit Sentiment Results:")
    print(reddit_df[["title", "sentiment_label", "confidence"]].head(10).to_string())

    print("\nSentiment Distribution (Reddit):")
    print(reddit_df["sentiment_label"].value_counts())

    # Analyze News articles
    print("\nStep 2: Analyzing News articles...")
    news_df = run_sentiment_on_news()

    print("\nNews Sentiment Results:")
    print(news_df[["title", "sentiment_label", "confidence"]].head(10).to_string())

    print("\nSentiment Distribution (News):")
    print(news_df["sentiment_label"].value_counts())

    