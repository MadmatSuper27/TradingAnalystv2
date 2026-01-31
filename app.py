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

    /* Active State for ALL Radio Labels in the app to be consistent (Solid Red) */
    div.stRadio > div > label:has(input:checked) {
        color: #ffffff !important;
        background-color: #ff4b4b !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3) !important;
    }

    /* Isolated Navigation Tabs Container Styling */
    div[data-testid="stWidgetProp-nav_radio"] div.stRadio > div,
    .st-key-nav_radio div.stRadio > div {
        display: flex;
        flex-direction: row;
        gap: 10px;
        background-color: #161b22;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #30363d;
        width: fit-content;
    }
    
    .st-key-nav_radio div.stRadio > div > label {
        background-color: transparent !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        color: #8b949e !important;
        font-weight: 600 !important;
        margin-bottom: 0px !important;
    }

    .st-key-nav_radio div.stRadio > div > label:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Hide the radio circle for navigation tabs */
    .st-key-nav_radio div.stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none !important;
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

    # --- Analysis Navigation ---
    nav_options = ["📊 Price Charts", "⚡ Risk-Return", "🔗 Correlation", "🔍 Company Insights"]
    
    # Initialize navigation in session state if not present
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = nav_options[0]

    active_tab = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.active_tab),
        horizontal=True,
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state.active_tab = active_tab

    if active_tab == "📊 Price Charts":
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
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True, key="price_performance_chart")

    elif active_tab == "⚡ Risk-Return":
        st.header("Risk-Return Analysis")
        # Calculate metrics
        returns = df.pct_change().dropna()
        
        # Calculate individual metrics
        n_years = len(df) / 252
        cagr = (df.iloc[-1] / df.iloc[0])**(1/n_years) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = cagr / vol
        
        # Sortino Ratio
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol
        
        # Drawdowns
        cum_rets = (1 + returns).cumprod()
        running_max = cum_rets.cummax()
        drawdowns = (cum_rets / running_max) - 1
        max_dd = drawdowns.min()
        dd_5th = drawdowns.quantile(0.05)
        cagr_over_max_dd = cagr / abs(max_dd)
        
        # Combine into stats DataFrame
        stats = pd.DataFrame({
            'Ticker': df.columns,
            'Annualized Return (%)': cagr * 100,
            'Annualized Volatility (%)': vol * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_dd * 100,
            'Drawdown 5th Perc (%)': dd_5th * 100,
            'CAGR/MaxDD': cagr_over_max_dd
        }).set_index('Ticker')

        fig_scatter = px.scatter(
            stats.reset_index(), 
            x='Annualized Volatility (%)', 
            y='Annualized Return (%)',
            text='Ticker',
            color='Ticker',
            size_max=15,
            labels={'Annualized Volatility (%)': 'Risk (Annualized Volatility %)', 'Annualized Return (%)': 'Reward (CAGR %)'},
            title="Reward vs Risk (Annualized)"
        )
        
        fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=600,
            showlegend=True,
            legend=dict(font=dict(color="white"))
        )
        
        # Add quadrant lines if possible (using mean or 0)
        fig_scatter.add_vline(x=stats['Annualized Volatility (%)'].mean(), line_dash="dot", opacity=0.3)
        fig_scatter.add_hline(y=stats['Annualized Return (%)'].mean(), line_dash="dot", opacity=0.3)
        
        st.plotly_chart(fig_scatter, use_container_width=True, key="risk_return_scatter")
        
        st.subheader("📊 Performance Summary Table")
        st.dataframe(
            stats.style.format({
                'Annualized Return (%)': "{:.2f}%",
                'Annualized Volatility (%)': "{:.2f}%",
                'Sharpe Ratio': "{:.2f}",
                'Sortino Ratio': "{:.2f}",
                'Max Drawdown (%)': "{:.2f}%",
                'Drawdown 5th Perc (%)': "{:.2f}%",
                'CAGR/MaxDD': "{:.2f}"
            }), 
            use_container_width=True
        )

    elif active_tab == "🔗 Correlation":
        st.header("Correlation Matrix")
        returns = df.pct_change().dropna()
        corr = returns.corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=0.2, zmax=1,
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
        st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap")

    elif active_tab == "🔍 Company Insights":
        st.header("Company Insights")
        insight_ticker = st.selectbox("Select ticker for insights", tickers)
        
        @st.cache_data(ttl=3600)
        def get_company_insights(symbol):
            try:
                t = yf.Ticker(symbol)
                info = t.info
                
                # Financials for EPS
                annual_financials = t.financials
                quarterly_financials = t.quarterly_financials
                
                # History for chart
                hist = t.history(period="1y")
                
                return info, annual_financials, quarterly_financials, hist
            except Exception as e:
                return None, None, None, None

        with st.spinner(f"Loading insights for {insight_ticker}..."):
            info, annual_f, quarterly_f, hist = get_company_insights(insight_ticker)
            
            if info:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Company Overview")
                    sector = info.get('sector')
                    industry = info.get('industry')
                    if sector and industry:
                        st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
                    st.write(info.get('longBusinessSummary', "No description available."))
                
                with col2:
                    st.subheader("Key Metrics")
                    roe = info.get('returnOnEquity')
                    if roe:
                        st.metric("ROE", f"{roe*100:.2f}%")
                    inst_holders = info.get('heldPercentInstitutions')
                    if inst_holders:
                        st.metric("Institutional Holders", f"{inst_holders*100:.2f}%")
                    else:
                        st.metric("Institutional Holders", info.get('heldPercentInstitutions', "N/A"))
                    
                    # YoY Quarterly EPS Growth
                    if quarterly_f is not None and not quarterly_f.empty:
                        eps_row = [row for row in quarterly_f.index if 'EPS' in row and 'Basic' in row]
                        if not eps_row:
                            eps_row = [row for row in quarterly_f.index if 'EPS' in row]
                        
                        if eps_row:
                            eps_vals = quarterly_f.loc[eps_row[0]].dropna()
                            if len(eps_vals) >= 5:
                                latest_eps = eps_vals.iloc[0]
                                last_year_eps = eps_vals.iloc[4]
                                if last_year_eps != 0:
                                    yoy_eps_growth = (latest_eps / last_year_eps - 1) * 100
                                    st.metric("YoY Quarterly EPS Growth", f"{yoy_eps_growth:.2f}%")
                    
                    # Key Reminders Table
                    st.markdown("---")
                    st.markdown("**💡 Strategy Reminders**")
                    reminder_data = pd.DataFrame({
                        "Metric": ["EPS Growth", "ROE"],
                        "Target": ["> 25%", "> 17%"]
                    })
                    st.table(reminder_data)

                # Simple Price/Volume Chart
                st.subheader("Price & Volume (1Y)")
                if not hist.empty:
                    fig_ins = go.Figure()
                    fig_ins.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#ff4b4b')))
                    fig_ins.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', yaxis='y2', opacity=0.3, marker_color='gray'))
                    
                    fig_ins.update_layout(
                        template="plotly_dark",
                        yaxis=dict(title="Price ($)"),
                        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False),
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_ins, use_container_width=True, key="company_insights_price_volume")

                # EPS Growth Tables
                st.subheader("📈 EPS Growth Analysis")
                
                def calc_eps_growth(financials, label="Period"):
                    if financials is not None and not financials.empty:
                        # Find EPS row - it might be 'Basic EPS' or similar
                        eps_row = [row for row in financials.index if 'EPS' in row and 'Basic' in row]
                        if not eps_row:
                            eps_row = [row for row in financials.index if 'EPS' in row]
                        
                        if eps_row:
                            eps = financials.loc[eps_row[0]].dropna()
                            eps = eps[::-1] # Oldest to newest
                            growth = eps.pct_change() * 100
                            df_growth = pd.DataFrame({
                                label: eps.index.astype(str),
                                'EPS': eps.values,
                                'Growth (%)': growth.values
                            })
                            return df_growth
                    return None

                col_annual, col_quarterly = st.columns(2)
                
                with col_annual:
                    st.write("**Past 5 Years (Annual)**")
                    annual_eps = calc_eps_growth(annual_f, "Year")
                    if annual_eps is not None:
                        st.dataframe(annual_eps.style.format({'EPS': "{:.2f}", 'Growth (%)': "{:.2f}%"}), use_container_width=True)
                    else:
                        st.info("Annual EPS data not available.")

                with col_quarterly:
                    st.write("**Last 5 Quarters (Quarterly)**")
                    quarter_eps = calc_eps_growth(quarterly_f, "Quarter")
                    if quarter_eps is not None:
                        st.dataframe(quarter_eps.style.format({'EPS': "{:.2f}", 'Growth (%)': "{:.2f}%"}), use_container_width=True)
                    else:
                        st.info("Quarterly EPS data not available.")
            else:
                st.error("No data available for this ticker.")

else:
    st.info("Please select a date range and click 'Fetch Data'.")
