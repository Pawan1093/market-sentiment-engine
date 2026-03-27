import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from loguru import logger


SUBREDDITS = [
    "wallstreetbets",
    "investing",
    "stocks",
    "IndiaInvestments",
    "StockMarketIndia"
]

def fetch_subreddit_rss(subreddit, limit=25):
   
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 market-sentiment-research"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for post in data["data"]["children"]:
            p = post["data"]
            posts.append({
                "id":        p["id"],
                "title":     p["title"],
                "text":      p.get("selftext", ""),
                "score":     p["score"],
                "comments":  p["num_comments"],
                "created":   datetime.fromtimestamp(p["created_utc"]),
                "subreddit": subreddit,
                "url":       p["url"]
            })
        
        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        return posts
    
    except Exception as e:
        logger.error(f"Error fetching r/{subreddit}: {e}")
        return []

def fetch_all_subreddits():
    
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = fetch_subreddit_rss(subreddit)
        all_posts.extend(posts)
    return all_posts

def save_to_csv(posts, filename="reddit_raw.csv"):
   
    df = pd.DataFrame(posts)
    filepath = f"data/{filename}"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} posts to {filepath}")
    return df

if __name__ == "__main__":
    print("Fetching live Reddit posts via RSS ")
    
    all_posts = fetch_all_subreddits()
    df = save_to_csv(all_posts)
    
    print(f"\nTotal posts collected: {len(df)}")
    print("\nSample posts:")
    print(df[["title", "score", "comments", "subreddit"]].head(10).to_string())