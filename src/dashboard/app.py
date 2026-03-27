import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title = "Indian Market Sentiment Engine",
    page_icon  = "📈",
    layout     = "wide"
)

# ── Database connection ─────────────────────────────────────
@st.cache_resource
def get_engine():
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    name     = os.getenv("DB_NAME",     "market_sentiment")
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "postgres123")
    url      = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url)

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_data():
    engine = get_engine()
    sentiment_df  = pd.read_sql("SELECT * FROM sentiment_data  ORDER BY created_at DESC", engine)
    ticker_df     = pd.read_sql("SELECT * FROM ticker_sentiment ORDER BY total_mentions DESC", engine)
    alerts_df     = pd.read_sql("SELECT * FROM market_alerts    ORDER BY created_at DESC", engine)
    prices_df     = pd.read_sql("SELECT * FROM stock_prices     ORDER BY fetched_at DESC", engine)
    anomalies_df  = pd.read_sql("SELECT * FROM price_anomalies  ORDER BY anomaly_pct DESC", engine)
    return sentiment_df, ticker_df, alerts_df, prices_df, anomalies_df

# ── Header ──────────────────────────────────────────────────
st.title("Indian Market Sentiment & Anomaly Detection Engine")
st.markdown("*Real-time NLP analysis of Reddit + News · Powered by FinBERT · Anomalies via Isolation Forest*")
st.divider()

# ── Load data ───────────────────────────────────────────────
try:
    sentiment_df, ticker_df, alerts_df, prices_df, anomalies_df = load_data()
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# ── Row 1: KPI Cards ────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_posts    = len(sentiment_df)
bullish_count  = (sentiment_df["sentiment_label"] == "bullish").sum()
bearish_count  = (sentiment_df["sentiment_label"] == "bearish").sum()
neutral_count  = (sentiment_df["sentiment_label"] == "neutral").sum()
total_anomalies= len(anomalies_df)

col1.metric("📰 Total Posts Analyzed",  f"{total_posts:,}")
col2.metric("🟢 Bullish Signals",        f"{bullish_count:,}",
            f"{bullish_count/total_posts*100:.1f}%")
col3.metric("🔴 Bearish Signals",        f"{bearish_count:,}",
            f"{bearish_count/total_posts*100:.1f}%",
            delta_color="inverse")
col4.metric("🟡 Neutral Signals",        f"{neutral_count:,}")
col5.metric("🚨 Anomalies Detected",     f"{total_anomalies:,}")

st.divider()

# ── Row 2: Sentiment Pie + Live Prices ──────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Overall Market Sentiment")
    sentiment_counts = sentiment_df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    color_map = {
        "bullish": "#00C853",
        "bearish": "#D50000",
        "neutral": "#FF6F00"
    }

    fig_pie = px.pie(
        sentiment_counts,
        values = "Count",
        names  = "Sentiment",
        color  = "Sentiment",
        color_discrete_map = color_map,
        hole   = 0.4
    )
    fig_pie.update_layout(
        showlegend    = True,
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("💰 Live Indian Stock Prices (₹)")
    if not prices_df.empty:
        for _, row in prices_df.iterrows():
            change = float(row["change_pct"])
            color  = "🟢" if change > 0 else "🔴"
            st.metric(
                label = f"{color} {row['company']}",
                value = f"₹{float(row['price_inr']):,.2f}",
                delta = f"{change:.2f}%"
            )
    else:
        st.info("Run stock_collector.py to fetch live prices")

st.divider()

# ── Row 3: Ticker Sentiment Bar Chart ───────────────────────
st.subheader("🎯 Ticker Sentiment Scores")
st.caption("Sentiment score: +1.0 = fully bullish · -1.0 = fully bearish")

if not ticker_df.empty:
    ticker_df["color"] = ticker_df["sentiment_score"].apply(
        lambda x: "#00C853" if x > 0 else ("#D50000" if x < 0 else "#FF6F00")
    )

    fig_bar = px.bar(
        ticker_df,
        x     = "ticker",
        y     = "sentiment_score",
        color = "sentiment_score",
        color_continuous_scale = ["#D50000", "#FF6F00", "#00C853"],
        range_color            = [-1, 1],
        text  = "total_mentions",
        hover_data = ["company", "bullish_count", "bearish_count"],
        labels = {
            "sentiment_score":  "Sentiment Score",
            "ticker":           "Stock Ticker",
            "total_mentions":   "Mentions"
        }
    )
    fig_bar.update_traces(texttemplate="%{text} mentions", textposition="outside")
    fig_bar.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_bar.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        coloraxis_showscale = False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Row 4: Market Alerts Table ──────────────────────────────
st.subheader("🚨 Market Alerts")
st.caption("Cross-referenced anomalies: sentiment shift + price movement")

if not alerts_df.empty:
    display_cols = ["ticker", "company", "sentiment_score",
                    "mentions", "price_anomaly", "price_change",
                    "alert_type", "action"]

    available = [c for c in display_cols if c in alerts_df.columns]

    def color_action(val):
        if "BUY"   in str(val): return "background-color: #1B5E20; color: white"
        if "SELL"  in str(val): return "background-color: #B71C1C; color: white"
        if "WATCH" in str(val): return "background-color: #E65100; color: white"
        return ""

    styled = alerts_df[available].style.applymap(
        color_action, subset=["action"]
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No alerts generated yet")

st.divider()

# ── Row 5: Price Anomalies ───────────────────────────────────
st.subheader("📉 TCS Price Anomalies Detected")
st.caption("Days where price/volume movement was statistically unusual (Isolation Forest)")

if not anomalies_df.empty:
    fig_anomaly = px.scatter(
        anomalies_df,
        x        = "date",
        y        = "close_inr",
        size     = "anomaly_pct",
        color    = "return_pct",
        color_continuous_scale = ["#D50000", "#FF6F00", "#00C853"],
        hover_data = ["return_pct", "volume_ratio", "anomaly_pct"],
        labels = {
            "close_inr":   "TCS Price (₹)",
            "date":        "Date",
            "return_pct":  "Daily Return %",
            "anomaly_pct": "Anomaly Score"
        }
    )
    fig_anomaly.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

st.divider()

# ── Row 6: Recent Headlines ──────────────────────────────────
st.subheader("📰 Recent Headlines & Sentiment")

col_filter1, col_filter2 = st.columns([1, 3])
with col_filter1:
    selected_sentiment = st.selectbox(
        "Filter by sentiment",
        ["All", "bullish", "bearish", "neutral"]
    )

filtered_df = sentiment_df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["sentiment_label"] == selected_sentiment
    ]

def sentiment_color(val):
    if val == "bullish": return "color: #00C853; font-weight: bold"
    if val == "bearish": return "color: #D50000; font-weight: bold"
    return "color: #FF6F00"

display_df = filtered_df[["source", "title", "sentiment_label",
                            "confidence", "tickers_str"]].head(50)

styled_headlines = display_df.style.applymap(
    sentiment_color, subset=["sentiment_label"]
)
st.dataframe(styled_headlines, use_container_width=True, height=400)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:12px'>
    Built with FinBERT · Isolation Forest · PostgreSQL · Streamlit
    · Data from Reddit RSS & Yahoo Finance
    </div>
    """,
    unsafe_allow_html=True
)