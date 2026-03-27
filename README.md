# Real-Time Indian Market Sentiment & Anomaly Detection Engine

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT-purple)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-green)

## What This Project Does
An end-to-end distributed ML system that streams live financial data from Reddit and financial news feeds, runs NLP sentiment analysis using FinBERT, detects price anomalies in BSE-listed Indian stocks, and displays everything on a live dashboard — fully automated, runs every 2 hours.

**The problem it solves:** Institutional investors have expensive algorithms that analyze news before retail investors see it. This system gives retail Indian investors access to the same sentiment signals — for free.

## Live Demo
> Dashboard link coming soon (Streamlit Cloud)

## Architecture
```
Reddit RSS ──────┐
News RSS ────────┼──► Apache Kafka ──► FinBERT NLP ──► Sentiment Scores ──┐
Alpha Vantage ───┘    (message broker)  (transformer)                       ├──► PostgreSQL ──► Streamlit Dashboard
                                                                            │
BSE Price History ──────────────────► Isolation Forest ──► Anomaly Alerts ─┘
```

## Tech Stack
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Ingestion | Python, Requests, RSS | Live Reddit + News + Stock data |
| Message Broker | Apache Kafka | Fault-tolerant streaming |
| NLP Model | FinBERT (HuggingFace) | Financial sentiment classification |
| Anomaly Detection | Isolation Forest (scikit-learn) | Unusual price/volume detection |
| Database | PostgreSQL + SQLAlchemy | Storing all results |
| Dashboard | Streamlit + Plotly | Live visualization |
| Automation | schedule library | Runs every 2 hours automatically |
| Testing | pytest (8/8 passing) | Unit tests for ML pipeline |

## Results
- Processes **200+ posts** per run from r/WallStreetBets, r/StockMarketIndia, r/investing
- Correctly flagged **TCS -6.94% crash** on Feb 4 2026 as highest anomaly (4x normal volume)
- Sentiment scoring across **12 BSE-listed stocks** including TCS, Infosys, HDFC Bank
- All prices displayed in **Indian Rupees (₹)**

## Project Structure
```
market-sentiment-engine/
├── src/
│   ├── ingestion/
│   │   ├── reddit_collector.py   # Reddit RSS fetcher
│   │   ├── news_collector.py     # Financial news RSS fetcher  
│   │   └── stock_collector.py    # Alpha Vantage BSE stock prices
│   ├── ml/
│   │   ├── sentiment_analyzer.py # FinBERT sentiment engine
│   │   ├── ticker_extractor.py   # Stock ticker detection
│   │   └── anomaly_detector.py   # Isolation Forest anomaly detection
│   ├── database/
│   │   └── db_manager.py         # PostgreSQL operations
│   └── dashboard/
│       └── app.py                # Streamlit live dashboard
├── tests/
│   └── test_pipeline.py          # 8 unit tests (all passing)
├── main.py                       # Auto pipeline scheduler
├── requirements.txt
└── README.md
```

## How To Run

### 1. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/market-sentiment-engine.git
cd market-sentiment-engine
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Setup database
```bash
# Create PostgreSQL database named market_sentiment
python src/database/db_manager.py
```

### 4. Run automated pipeline
```bash
python main.py
```

### 5. Launch dashboard
```bash
streamlit run src/dashboard/app.py
```

### 6. Run tests
```bash
pytest tests/test_pipeline.py -v
```

## Free APIs Used
- **Reddit** — Public JSON feed (no API key needed)
- **Alpha Vantage** — Free tier (25 calls/day) for BSE stock prices
- **Yahoo Finance RSS** — Free financial news feed
- **HuggingFace** — Free FinBERT model download

## Author
Pawan Tatyaso Pawar
```

---

### Step 2 — Create `.env.example` file

Create `.env.example` in root folder (this is safe to push to GitHub — no real keys):
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=market_sentiment_bot
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=market_sentiment
DB_USER=postgres
DB_PASSWORD=your_postgres_password