import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Pioneer Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    /* Cards/Containers */
    .css-1r6q64v {
        background-color: #161b22;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30363d;
    }

    /* Tab Label Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #ff4b4b !important;
        border-bottom-color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("🔍 Analysis Parameters")

with st.sidebar:
    st.subheader("Select Stocks")
    stock_input = st.text_input(
        "Enter symbols (comma-separated)",
        value="SPY, AAPL, GOOGL, MSFT, TSLA",
        help="e.g., SPY, AAPL, MSFT"
    )
    tickers = [t.strip().upper() for t in stock_input.split(",")]

    st.subheader("Timeframe & Interval")
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=365)
    
    date_range = st.date_input(
        "Select period",
        value=(start_date_default, end_date),
        max_value=end_date
    )

    interval = st.selectbox(
        "Select interval",
        options=["1d", "1wk", "1mo"],
        index=0,
        help="Data granularity (1 day, 1 week, 1 month)"
    )

    fetch_button = st.button("🔄 Refresh Data")

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def get_stock_data(tickers, start, end, interval):
    try:
        data = yf.download(tickers, start=start, end=end, interval=interval)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers[0]]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- Main Logic ---
st.title("🚀 Stock Pioneer Dashboard")

if len(date_range) == 2:
    start_dt, end_dt = date_range
    
    # Reactive fetching: Fetch whenever tickers, dates, or interval change
    # st.cache_data ensures we don't spam the API unnecessarily
    with st.spinner("Fetching market data..."):
        df = get_stock_data(tuple(tickers), start_dt, end_dt, interval)
        if df is None or df.empty:
            st.error("No data found for the selected parameters.")
            st.stop()

    # --- Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Price Charts", "⚡ Risk-Return", "🔗 Correlation"])

    with tab1:
        st.header("Relative Performance")
        chart_type = st.radio("Chart Type Scale", ["Relative Performance (%)", "Absolute Prices"], horizontal=True)
        
        if chart_type == "Relative Performance (%)":
            # Normalize to 0% at start
            rel_perf = (df / df.iloc[0] - 1) * 100
            fig = px.line(rel_perf, labels={"value": "Return (%)", "Date": "Date"})
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        else:
            fig = px.line(df, labels={"value": "Price ($)", "Date": "Date"})

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                font=dict(color="white", size=12)
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Risk-Return Analysis")
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate metrics
        stats = pd.DataFrame({
            'Annual Return (%)': returns.mean() * 252 * 100,
            'Annual Volatility (%)': returns.std() * np.sqrt(252) * 100
        })
        stats['Ticker'] = stats.index

        fig_scatter = px.scatter(
            stats, 
            x='Annual Volatility (%)', 
            y='Annual Return (%)',
            text='Ticker',
            color='Ticker',
            size_max=15,
            labels={'Annual Volatility (%)': 'Risk (Annualized Volatility %)', 'Annual Return (%)': 'Reward (Annualized Return %)'},
            title="Reward vs Risk (Annualized)"
        )
        
        fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="white"))
        )
        
        # Add quadrant lines if possible (using mean or 0)
        fig_scatter.add_vline(x=stats['Annual Volatility (%)'].mean(), line_dash="dot", opacity=0.3)
        fig_scatter.add_hline(y=stats['Annual Return (%)'].mean(), line_dash="dot", opacity=0.3)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.dataframe(stats[['Annual Return (%)', 'Annual Volatility (%)']].style.format("{:.2f}%"), use_container_width=True)

    with tab3:
        st.header("Correlation Matrix")
        returns = df.pct_change().dropna()
        corr = returns.corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            labels=dict(color="Correlation")
        )
        
        fig_corr.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(
                title=dict(font=dict(color="white")),
                tickfont=dict(color="white")
            )
        )
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("Please select a date range and click 'Fetch Data'.")
