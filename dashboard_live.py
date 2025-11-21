import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.database import SentimentDatabase
from src.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.sentiment_index import SentimentIndexCalculator
from src.news_fetcher import FinancialNewsFetcher, validate_financial_text

# Page configuration
st.set_page_config(
    page_title="Investor Market Sentiment Index - Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 24px !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .financial-warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .financial-success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_database():
    return SentimentDatabase()

@st.cache_resource
def init_calculator():
    return SentimentIndexCalculator()

@st.cache_resource
def init_fetcher():
    return FinancialNewsFetcher()

@st.cache_resource
def init_analyzer():
    return FinBERTSentimentAnalyzer()

db = init_database()
calculator = init_calculator()
fetcher = init_fetcher()

# Title
st.title("📈 Investor Market Sentiment Index")
st.markdown("*Real-time financial news sentiment analysis powered by FinBERT*")
st.markdown("**🔴 LIVE MODE** - Fetching real financial news from multiple sources")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio(
        "Navigate to:",
        ["Dashboard", "Live News Feed", "Historical Analysis", "Source Breakdown", "Analyze Text"]
    )

    st.markdown("---")

    # Live data controls
    st.subheader("🔴 Live Data")

    if st.button("Fetch Live News", type="primary"):
        with st.spinner("Fetching live financial news..."):
            # Use RSS feeds (no API keys needed for demo)
            df = fetcher.fetch_rss_fallback()

            if not df.empty:
                st.success(f"Fetched {len(df)} financial articles!")

                # Analyze sentiment
                analyzer = init_analyzer()
                df = analyzer.analyze_dataframe(df, text_column='text', batch_size=32)

                # Calculate indices
                overall = calculator.calculate_overall_index(df)
                daily_index = calculator.calculate_daily_index(df)
                source_index = calculator.calculate_source_index(df)

                # Save to database
                db.save_sentiment_records(df)
                db.save_daily_index(daily_index)
                db.save_source_index(source_index)

                st.success("✅ Sentiment analysis complete!")
                st.rerun()
            else:
                st.error("No articles fetched. Check your internet connection.")

    st.markdown("---")

    st.subheader("API Keys (Optional)")
    st.markdown("For more data sources, add API keys:")

    with st.expander("API Key Configuration"):
        newsapi_key = st.text_input("NewsAPI.org", type="password", help="Get free key: https://newsapi.org/register")
        guardian_key = st.text_input("The Guardian", type="password", help="Get free key: https://open-platform.theguardian.com")
        finnhub_key = st.text_input("Finnhub", type="password", help="Get free key: https://finnhub.io/register")

        if st.button("Fetch with API Keys"):
            api_keys = {
                'newsapi': newsapi_key if newsapi_key else None,
                'guardian': guardian_key if guardian_key else None,
                'finnhub': finnhub_key if finnhub_key else None,
            }

            with st.spinner("Fetching from multiple sources..."):
                df = fetcher.fetch_all(api_keys=api_keys, days=7)

                if not df.empty:
                    st.success(f"Fetched {len(df)} articles!")

                    analyzer = init_analyzer()
                    df = analyzer.analyze_dataframe(df, text_column='text', batch_size=32)

                    overall = calculator.calculate_overall_index(df)
                    daily_index = calculator.calculate_daily_index(df)
                    source_index = calculator.calculate_source_index(df)

                    db.save_sentiment_records(df)
                    db.save_daily_index(daily_index)
                    db.save_source_index(source_index)

                    st.success("✅ Analysis complete!")
                    st.rerun()

    st.markdown("---")

    st.subheader("About")
    st.info("""
    **Financial-Only Analysis:**
    - ✅ Only analyzes financial news
    - ✅ Validates content before analysis
    - ✅ Real-time data from RSS feeds
    - ✅ Multiple source support

    **Index Scale:**
    - 120+: Very Bullish 🚀
    - 110-120: Bullish 📈
    - 90-110: Neutral ➡️
    - 80-90: Bearish 📉
    - <80: Very Bearish 💥
    """)

# Main content
if page == "Dashboard":
    st.header("Current Market Sentiment")

    # Get latest data (session is automatically reset inside the method)
    latest = db.get_latest_index()

    if latest:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Sentiment Index",
                value=f"{latest['sentiment_index']:.2f}",
                delta=f"{latest.get('momentum', 0):.2f}"
            )

        with col2:
            interpretation = calculator.interpret_index(latest['sentiment_index'])
            mood_emoji = {
                "Very Bullish": "🚀",
                "Bullish": "📈",
                "Neutral": "➡️",
                "Bearish": "📉",
                "Very Bearish": "💥"
            }.get(interpretation, "")

            st.metric(
                label="Market Mood",
                value=f"{mood_emoji} {interpretation}"
            )

        with col3:
            st.metric(
                label="Compound Score",
                value=f"{latest['avg_compound']:.3f}"
            )

        with col4:
            st.metric(
                label="Articles Analyzed",
                value=f"{latest['article_count']}"
            )

        # Gauge chart
        st.subheader("Sentiment Index Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest['sentiment_index'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment Index"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 80], 'color': "darkred"},
                    {'range': [80, 90], 'color': "orange"},
                    {'range': [90, 110], 'color': "yellow"},
                    {'range': [110, 120], 'color': "lightgreen"},
                    {'range': [120, 200], 'color': "darkgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution
        st.subheader("Sentiment Distribution")

        if latest.get('label_distribution'):
            import json
            # Check if it's already a dict or needs parsing
            dist = latest['label_distribution']
            if isinstance(dist, str):
                dist = json.loads(dist)

            fig_pie = px.pie(
                names=list(dist.keys()),
                values=list(dist.values()),
                title="Article Sentiment Breakdown",
                color_discrete_map={'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Recent trend (last 30 days)
        st.subheader("30-Day Trend")
        history = db.get_historical_index(days=30)

        if history:
            df_hist = pd.DataFrame(history)
            fig_trend = px.line(df_hist, x='date', y='sentiment_index',
                              title="Sentiment Index Over Time")
            fig_trend.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Neutral")
            st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.warning("⚠️ No data available. Click 'Fetch Live News' in the sidebar to get started!")

elif page == "Live News Feed":
    st.header("🔴 Live Financial News Feed")

    st.markdown("Recent articles analyzed for sentiment:")

    # Get recent records (session is automatically reset inside the method)
    records = db.get_latest_records(limit=50)

    if records:
        df_records = pd.DataFrame(records)

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by sentiment",
                options=['positive', 'neutral', 'negative'],
                default=['positive', 'neutral', 'negative']
            )
        with col2:
            source_filter = st.multiselect(
                "Filter by source",
                options=df_records['source'].unique().tolist(),
                default=df_records['source'].unique().tolist()
            )

        # Apply filters
        df_filtered = df_records[
            (df_records['sentiment_label'].isin(sentiment_filter)) &
            (df_records['source'].isin(source_filter))
        ]

        # Display articles
        for idx, row in df_filtered.iterrows():
            with st.expander(f"{row['source']} - {row['sentiment_label'].upper()} ({row['sentiment_score']:.2%})"):
                st.markdown(f"**Headline:** {row['headline'][:200]}...")
                st.markdown(f"**Date:** {row['date']}")
                st.markdown(f"**Sentiment:** {row['sentiment_label']} (Confidence: {row['sentiment_score']:.2%})")

                # Sentiment breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{row['positive_score']:.2%}")
                with col2:
                    st.metric("Neutral", f"{row['neutral_score']:.2%}")
                with col3:
                    st.metric("Negative", f"{row['negative_score']:.2%}")
    else:
        st.info("No articles yet. Fetch live news from the sidebar!")

elif page == "Historical Analysis":
    st.header("Historical Sentiment Analysis")

    days = st.slider("Select time period (days)", 7, 90, 30)

    # Get historical data (session is automatically reset inside the method)
    history = db.get_historical_index(days=days)

    if history and len(history) > 0:
        df_history = pd.DataFrame(history)
        df_history['date'] = pd.to_datetime(df_history['date'])

        # Main trend chart
        st.subheader(f"Sentiment Trend (Last {days} Days)")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['sentiment_index'],
            mode='lines+markers',
            name='Sentiment Index',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))

        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['avg_compound'],
            mode='lines',
            name='Compound Score',
            line=dict(color='orange', width=1, dash='dash'),
            yaxis='y2'
        ))

        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Neutral")

        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Sentiment Index",
            yaxis2=dict(
                title="Compound Score",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volatility chart
        st.subheader("Sentiment Volatility")
        fig_vol = px.bar(df_history, x='date', y='volatility',
                         title="Daily Sentiment Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Index", f"{df_history['sentiment_index'].mean():.2f}")
        with col2:
            st.metric("Highest Index", f"{df_history['sentiment_index'].max():.2f}")
        with col3:
            st.metric("Lowest Index", f"{df_history['sentiment_index'].min():.2f}")

    else:
        st.warning("⚠️ No historical data available for selected period. Fetch live news to populate data!")

elif page == "Source Breakdown":
    st.header("Sentiment by News Source")

    days = st.slider("Select time period (days)", 7, 30, 7)

    # Get source data (session is automatically reset inside the method)
    by_source = db.get_sentiment_by_source(days=days)

    if by_source:
        df_source = pd.DataFrame(by_source)

        # Group by source and get latest
        latest_by_source = df_source.sort_values('date').groupby('source').last().reset_index()

        # Bar chart
        fig = px.bar(
            latest_by_source,
            x='source',
            y='sentiment_index',
            color='sentiment_index',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Latest Sentiment Index by Source"
        )
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.subheader("Source Details")
        display_df = latest_by_source[['source', 'sentiment_index', 'avg_compound', 'article_count']]
        display_df['interpretation'] = display_df['sentiment_index'].apply(calculator.interpret_index)
        st.dataframe(display_df, use_container_width=True)

    else:
        st.warning("No source data available for selected period.")

elif page == "Analyze Text":
    st.header("Analyze Custom Financial Text")

    st.markdown("""
    **⚠️ Financial Content Only:**
    This tool only analyzes financial news and content. Non-financial text will be rejected.
    """)

    text_input = st.text_area(
        "Enter financial news or text",
        height=150,
        placeholder="Example: Apple stock surges 5% on strong earnings report beating analyst expectations..."
    )

    if st.button("Analyze Sentiment", type="primary"):
        if text_input and len(text_input.strip()) > 0:
            # Validate financial content FIRST
            is_valid, reason = validate_financial_text(text_input)

            if not is_valid:
                st.markdown(f"""
                <div class="financial-warning">
                    <h4>⚠️ Invalid Financial Content</h4>
                    <p><strong>Reason:</strong> {reason}</p>
                    <p>Please enter financial news related to:</p>
                    <ul>
                        <li>Stock markets, trading, investments</li>
                        <li>Company earnings, revenue, profits</li>
                        <li>Economic indicators (GDP, inflation, interest rates)</li>
                        <li>Financial markets (bonds, forex, commodities)</li>
                        <li>Central banks, monetary policy</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing financial sentiment..."):
                    analyzer = init_analyzer()
                    result = analyzer.analyze_sentiment(text_input)

                    # Calculate compound score (positive - negative)
                    compound = result['positive'] - result['negative']

                    # Display success
                    st.markdown(f"""
                    <div class="financial-success">
                        <h4>✅ Valid Financial Content - Analysis Complete!</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Sentiment", result['label'].upper())
                        st.metric("Confidence", f"{result['score']:.2%}")

                    with col2:
                        st.metric("Compound Score", f"{compound:.3f}")

                        # Calculate index
                        index = 100 + (compound * 50)
                        interpretation = calculator.interpret_index(index)
                        st.metric("Sentiment Index", f"{index:.2f} ({interpretation})")

                    # Sentiment breakdown chart
                    st.subheader("Sentiment Breakdown")

                    sentiment_data = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Score': [result['positive'], result['neutral'], result['negative']]
                    })

                    fig = px.bar(
                        sentiment_data,
                        x='Sentiment',
                        y='Score',
                        color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'},
                        title="Sentiment Component Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please enter text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by FinBERT | Real-time Financial Sentiment Analysis</p>
    <p>Data sources: Reuters, CNBC, MarketWatch, Bloomberg (via RSS)</p>
</div>
""", unsafe_allow_html=True)
