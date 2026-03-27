import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import os
import time

load_dotenv()

BASE_URL    = "https://www.alphavantage.co/query"

# Indian stocks on NSE - format is SYMBOL.BSE or SYMBOL.NSE
INDIAN_TICKERS = {
    "RELIANCE.BSE": "Reliance Industries",
    "TCS.BSE":      "Tata Consultancy Services",
    "INFY.BSE":     "Infosys",
    "HDFCBANK.BSE": "HDFC Bank",
    "ICICIBANK.BSE":"ICICI Bank",
    "WIPRO.BSE":    "Wipro",
    "SBIN.BSE":     "State Bank of India",
    "BAJFINANCE.BSE":"Bajaj Finance",
    "TATAMOTORS.BSE":"Tata Motors",
    "ADANIENT.BSE": "Adani Enterprises",
    "HINDUNILVR.BSE":"Hindustan Unilever",
    "MARUTI.BSE":   "Maruti Suzuki",
}

def get_usd_to_inr():
    
    api_key = os.getenv("ALPHA_VANTAGE_KEY")

    params = {
        "function":    "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "INR",
        "apikey":      api_key
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data     = response.json()
        rate     = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        logger.info(f"Live USD to INR rate: ₹{rate:.2f}")
        return rate

    except Exception as e:
        logger.warning(f"Could not fetch exchange rate, using fallback: {e}")
        return 83.5  # Fallback rate


def fetch_stock_quote(ticker, inr_rate):
   
    api_key = os.getenv("ALPHA_VANTAGE_KEY")

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol":   ticker,
        "apikey":   api_key
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data     = response.json()
        quote    = data.get("Global Quote", {})

        if not quote or not quote.get("05. price"):
            logger.warning(f"No data returned for {ticker}")
            return None

        price     = float(quote.get("05. price", 0))
        change    = float(quote.get("09. change", 0))
        prev_close= float(quote.get("08. previous close", 0))

        return {
            "ticker":       ticker,
            "company":      INDIAN_TICKERS.get(ticker, ticker),
            "price_inr":    round(price, 2),
            "change_inr":   round(change, 2),
            "change_pct":   quote.get("10. change percent", "0%").replace("%",""),
            "volume":       int(quote.get("06. volume", 0)),
            "prev_close":   round(prev_close, 2),
            "high_inr":     round(float(quote.get("03. high", 0)), 2),
            "low_inr":      round(float(quote.get("04. low", 0)), 2),
            "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None


def fetch_daily_history(ticker):
    
    api_key = os.getenv("ALPHA_VANTAGE_KEY")

    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     ticker,
        "outputsize": "compact",
        "apikey":     api_key
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        data     = response.json()
        ts       = data.get("Time Series (Daily)", {})

        if not ts:
            logger.warning(f"No history for {ticker}")
            return None

        rows = []
        for date, values in ts.items():
            rows.append({
                "ticker":     ticker,
                "company":    INDIAN_TICKERS.get(ticker, ticker),
                "date":       date,
                "open_inr":   float(values["1. open"]),
                "high_inr":   float(values["2. high"]),
                "low_inr":    float(values["3. low"]),
                "close_inr":  float(values["4. close"]),
                "volume":     int(values["5. volume"]),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        
        df["daily_return_pct"] = df["close_inr"].pct_change() * 100

        logger.info(f"Fetched {len(df)} days of history for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching history for {ticker}: {e}")
        return None


def fetch_all_indian_quotes():
    
    inr_rate = get_usd_to_inr()
    results  = []

    for i, (ticker, company) in enumerate(INDIAN_TICKERS.items()):
        logger.info(f"Fetching {company} ({i+1}/{len(INDIAN_TICKERS)})...")
        quote = fetch_stock_quote(ticker, inr_rate)
        if quote:
            results.append(quote)
        time.sleep(12)  
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("INDIAN STOCK MARKET - LIVE PRICE COLLECTOR")
    print("All prices in Indian Rupees (₹)")
    print("=" * 60)

    
    inr_rate = get_usd_to_inr()
    print(f"\nLive exchange rate: 1 USD = ₹{inr_rate:.2f}")

    
    priority_tickers = {
        "RELIANCE.BSE": "Reliance Industries",
        "TCS.BSE":      "Tata Consultancy Services",
        "INFY.BSE":     "Infosys",
        "HDFCBANK.BSE": "HDFC Bank",
    }

    print(f"\nFetching live prices for top Indian stocks...")
    results = []

    for ticker, company in priority_tickers.items():
        logger.info(f"Fetching {company}...")
        quote = fetch_stock_quote(ticker, inr_rate)
        if quote:
            results.append(quote)
            print(f"  ✓ {company}: ₹{quote['price_inr']:,.2f} ({quote['change_pct']}%)")
        time.sleep(13)

    if results:
        df = pd.DataFrame(results)
        df.to_csv("data/indian_stock_quotes.csv", index=False)
        print("\nLive Indian Stock Prices:")
        print(df[["company", "price_inr", "change_inr",
                   "change_pct", "volume"]].to_string())
        print("\nSaved to data/indian_stock_quotes.csv")
    else:
        print("\nNo data fetched - check ALPHA_VANTAGE_KEY in .env file")

    # Step 3: Fetch history for TCS (for anomaly detection later)
    print("\nFetching 100 days history for TCS...")
    history = fetch_daily_history("TCS.BSE")
    if history is not None:
        history.to_csv("data/tcs_history.csv", index=False)
        print(f"\nTCS last 5 days:")
        print(history[["date", "close_inr", "volume",
                        "daily_return_pct"]].tail(5).to_string())