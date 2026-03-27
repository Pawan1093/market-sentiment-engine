import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from loguru import logger

# 100% free financial news RSS feeds - no API key needed
NEWS_FEEDS = {
    "reuters_business":  "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance":     "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha":     "https://seekingalpha.com/feed.xml",
    "marketwatch":       "https://feeds.marketwatch.com/marketwatch/topstories",
    "moneycontrol":      "https://www.moneycontrol.com/rss/business.xml"
}

def fetch_news_feed(feed_name, url):
  
    headers = {
        "User-Agent": "Mozilla/5.0 market-sentiment-research"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)
        articles = []

        
        items = root.findall(".//item")  # RSS format
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")  # Atom format

        for item in items:
            # Get title
            title_el = item.find("title")
            title = title_el.text if title_el is not None else ""

            # Get description/summary
            desc_el = item.find("description")
            if desc_el is None:
                desc_el = item.find("{http://www.w3.org/2005/Atom}summary")
            description = desc_el.text if desc_el is not None else ""

            # Get publish date
            date_el = item.find("pubDate")
            if date_el is None:
                date_el = item.find("{http://www.w3.org/2005/Atom}updated")
            pub_date = date_el.text if date_el is not None else str(datetime.now())

            # Get link
            link_el = item.find("link")
            link = link_el.text if link_el is not None else ""

            if title:  
                articles.append({
                    "source":      feed_name,
                    "title":       title.strip(),
                    "description": str(description).strip()[:500],  # limit length
                    "published":   pub_date,
                    "url":         link,
                    "fetched_at":  datetime.now()
                })

        logger.info(f"Fetched {len(articles)} articles from {feed_name}")
        return articles

    except Exception as e:
        logger.warning(f"Could not fetch {feed_name}: {e}")
        return []

def fetch_all_news():
  
    all_articles = []
    for feed_name, url in NEWS_FEEDS.items():
        articles = fetch_news_feed(feed_name, url)
        all_articles.extend(articles)
    return all_articles

def save_to_csv(articles, filename="news_raw.csv"):

    df = pd.DataFrame(articles)
    filepath = f"data/{filename}"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} articles to {filepath}")
    return df

if __name__ == "__main__":
    print("Fetching live financial news from RSS feeds...")
    print("=" * 60)

    all_articles = fetch_all_news()
    df = save_to_csv(all_articles)

    print(f"\nTotal articles collected: {len(df)}")
    print("\nSample headlines:")
    print(df[["source", "title"]].head(10).to_string())