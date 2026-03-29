import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title = "Indian Market Sentiment Engine",
    page_icon  = "📈",
    layout     = "wide"
)

@st.cache_data
def load_data():
    base = "data/"
    sentiment_df  = pd.read_csv(base + "combined_sentiment.csv")  if os.path.exists(base + "combined_sentiment.csv")  else pd.DataFrame()
    ticker_df     = pd.read_csv(base + "ticker_summary.csv")      if os.path.exists(base + "ticker_summary.csv")      else pd.DataFrame()
    alerts_df     = pd.read_csv(base + "market_alerts.csv")       if os.path.exists(base + "market_alerts.csv")       else pd.DataFrame()
    prices_df     = pd.read_csv(base + "indian_stock_quotes.csv") if os.path.exists(base + "indian_stock_quotes.csv") else pd.DataFrame()
    anomalies_df  = pd.read_csv(base + "tcs_anomalies.csv")       if os.path.exists(base + "tcs_anomalies.csv")       else pd.DataFrame()
    return sentiment_df, ticker_df, alerts_df, prices_df, anomalies_df

sentiment_df, ticker_df, alerts_df, prices_df, anomalies_df = load_data()

st.title("🇮🇳 Indian Market Sentiment & Anomaly Detection Engine")
st.markdown("*Real-time NLP analysis · FinBERT Transformer · Isolation Forest Anomaly Detection*")
st.divider()

# ── KPI Cards ───────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_posts     = len(sentiment_df)
bullish_count   = (sentiment_df["sentiment_label"] == "bullish").sum() if not sentiment_df.empty else 0
bearish_count   = (sentiment_df["sentiment_label"] == "bearish").sum() if not sentiment_df.empty else 0
neutral_count   = (sentiment_df["sentiment_label"] == "neutral").sum() if not sentiment_df.empty else 0
total_anomalies = len(anomalies_df)

col1.metric("📰 Posts Analyzed",    f"{total_posts:,}")
col2.metric("🟢 Bullish Signals",   f"{bullish_count:,}", f"{bullish_count/max(total_posts,1)*100:.1f}%")
col3.metric("🔴 Bearish Signals",   f"{bearish_count:,}", f"{bearish_count/max(total_posts,1)*100:.1f}%", delta_color="inverse")
col4.metric("🟡 Neutral Signals",   f"{neutral_count:,}")
col5.metric("🚨 Anomalies Found",   f"{total_anomalies:,}")

st.divider()

# ── Sentiment Pie + Stock Prices ────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Overall Market Sentiment")
    if not sentiment_df.empty:
        counts = sentiment_df["sentiment_label"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, values="Count", names="Sentiment",
            color="Sentiment",
            color_discrete_map={"bullish":"#00C853","bearish":"#D50000","neutral":"#FF6F00"},
            hole=0.4
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("💰 Live Indian Stock Prices (₹)")
    if not prices_df.empty:
        for _, row in prices_df.iterrows():
            try:
                change = float(row["change_pct"])
                icon   = "🟢" if change > 0 else "🔴"
                st.metric(
                    label = f"{icon} {row['company']}",
                    value = f"₹{float(row['price_inr']):,.2f}",
                    delta = f"{change:.2f}%"
                )
            except:
                pass
    else:
        st.info("Stock price data not available")

st.divider()

# ── Ticker Sentiment Bar Chart ───────────────────────────────
st.subheader("🎯 Stock Ticker Sentiment Scores")
st.caption("Score: +1.0 = fully bullish · -1.0 = fully bearish")

if not ticker_df.empty:
    fig_bar = px.bar(
        ticker_df,
        x     = "tickers",
        y     = "sentiment_score",
        color = "sentiment_score",
        color_continuous_scale = ["#D50000","#FF6F00","#00C853"],
        range_color = [-1, 1],
        text  = "total_mentions",
        hover_data = ["company","bullish_count","bearish_count"],
        labels = {"sentiment_score":"Sentiment Score","tickers":"Stock Ticker"}
    )
    fig_bar.update_traces(texttemplate="%{text} mentions", textposition="outside")
    fig_bar.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Market Alerts ────────────────────────────────────────────
st.subheader("🚨 Market Alerts")
st.caption("Cross-referenced: sentiment shift + price anomaly")

if not alerts_df.empty:
    def color_action(val):
        if "BUY"   in str(val): return "background-color:#1B5E20;color:white"
        if "SELL"  in str(val): return "background-color:#B71C1C;color:white"
        if "WATCH" in str(val): return "background-color:#E65100;color:white"
        return ""
    cols = [c for c in ["ticker","company","sentiment_score","mentions","price_anomaly","price_change","alert_type","action"] if c in alerts_df.columns]
    st.dataframe(alerts_df[cols].style.applymap(color_action, subset=["action"] if "action" in cols else []), use_container_width=True)

st.divider()

# ── TCS Anomalies ────────────────────────────────────────────
st.subheader("📉 TCS Price Anomalies")
st.caption("Statistically unusual days detected by Isolation Forest")

if not anomalies_df.empty:
    fig_a = px.scatter(
        anomalies_df,
        x="date", y="close_inr",
        size="anomaly_pct", color="return_pct",
        color_continuous_scale=["#D50000","#FF6F00","#00C853"],
        hover_data=["return_pct","volume_ratio","anomaly_pct"],
        labels={"close_inr":"TCS Price (₹)","date":"Date","return_pct":"Daily Return %"}
    )
    fig_a.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_a, use_container_width=True)

st.divider()

# ── Headlines Table ──────────────────────────────────────────
st.subheader("📰 Headlines & Sentiment")

if not sentiment_df.empty:
    selected = st.selectbox("Filter by sentiment", ["All","bullish","bearish","neutral"])
    filtered = sentiment_df if selected == "All" else sentiment_df[sentiment_df["sentiment_label"] == selected]

    def color_sentiment(val):
        if val == "bullish": return "color:#00C853;font-weight:bold"
        if val == "bearish": return "color:#D50000;font-weight:bold"
        return "color:#FF6F00"

    cols = [c for c in ["source","title","sentiment_label","confidence","tickers_str"] if c in filtered.columns]
    st.dataframe(
        filtered[cols].head(50).style.applymap(color_sentiment, subset=["sentiment_label"]),
        use_container_width=True, height=400
    )

st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:12px'>Built with FinBERT · Isolation Forest · Streamlit · Data from Reddit & Yahoo Finance</div>", unsafe_allow_html=True)