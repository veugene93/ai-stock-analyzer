
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("AI STOCK ANALYZER")
st.caption(
    "Independent quantitative stock analysis "
    "| Historical market data | Risk assessment"
)


# ==========================================
# 1. STOCK SEARCH
# ==========================================

@st.cache_data(ttl=3600)
def search_stock(query):
    try:
        result = yf.Search(query, max_results=10)
        quotes = result.quotes

        stocks = []

        for item in quotes:
            if item.get("quoteType") == "EQUITY":
                stocks.append({
                    "symbol": item.get("symbol"),
                    "name": (
                        item.get("shortname")
                        or item.get("longname")
                        or item.get("symbol")
                    )
                })

        return stocks

    except Exception:
        return []


# ==========================================
# 2. HISTORICAL DATA
# ==========================================

@st.cache_data(ttl=1800)
def load_history(ticker):
    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
            period="max",
            interval="1d",
            auto_adjust=True
        )

        if data.empty:
            return None

        data = data.dropna(subset=["Close"])

        return data

    except Exception:
        return None


# ==========================================
# 3. TECHNICAL INDICATORS
# ==========================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA100"] = close.rolling(100).mean()
    df["SMA200"] = close.rolling(200).mean()

    # Exponential moving averages

    df["EMA12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    # MACD

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # RSI

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df["RSI"] = df["RSI"].fillna(50)

    # Average True Range

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # Bollinger Bands

    rolling_std = close.rolling(20).std()

    df["BB_UPPER"] = (
        df["SMA20"] + 2 * rolling_std
    )

    df["BB_LOWER"] = (
        df["SMA20"] - 2 * rolling_std
    )

    # On Balance Volume

    direction = np.sign(close.diff()).fillna(0)

    df["OBV"] = (
        direction * volume.fillna(0)
    ).cumsum()

    # Returns

    df["RETURNS"] = close.pct_change()

    # Annualized historical volatility

    df["VOLATILITY"] = (
        df["RETURNS"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    # Average trading volume

    df["AVG_VOLUME"] = volume.rolling(20).mean()

    return df


# ==========================================
# 4. RISK ANALYSIS
# ==========================================

def calculate_risks(df):

    returns = df["RETURNS"].dropna()

    if len(returns) < 2:
        return None

    cumulative_returns = (
        1 + returns
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max
    ) - 1

    var_95 = np.percentile(returns, 5)

    tail_returns = returns[returns <= var_95]

    cvar_95 = (
        tail_returns.mean()
        if len(tail_returns) > 0
        else var_95
    )

    annual_volatility = (
        returns.std() * np.sqrt(252) * 100
    )

    return {
        "annual_volatility": annual_volatility,
        "max_drawdown": abs(drawdown.min()) * 100,
        "var_95": abs(var_95) * 100,
        "cvar_95": abs(cvar_95) * 100
    }


# ==========================================
# 5. TECHNICAL SCORING
# ==========================================

def calculate_score(df):

    last = df.iloc[-1]

    score = 0

    # Trend analysis

    if last["Close"] > last["SMA20"]:
        score += 1

    if last["Close"] > last["SMA50"]:
        score += 1

    if last["Close"] > last["SMA200"]:
        score += 2

    # RSI analysis

    if 50 < last["RSI"] < 70:
        score += 1

    elif last["RSI"] < 30:
        score += 1

    elif last["RSI"] > 75:
        score -= 2

    # MACD analysis

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 1
    else:
        score -= 1

    # Volume analysis

    if (
        pd.notna(last["AVG_VOLUME"])
        and last["AVG_VOLUME"] > 0
        and last["Volume"] > last["AVG_VOLUME"]
    ):
        if last["Close"] > df["Close"].iloc[-2]:
            score += 1
        else:
            score -= 1

    if score >= 5:
        signal = "BULLISH"

    elif score >= 2:
        signal = "MODERATELY BULLISH"

    elif score <= -2:
        signal = "BEARISH"

    else:
        signal = "HOLD / WAIT"

    return score, signal


# ==========================================
# 6. SUPPORT AND RESISTANCE
# ==========================================

def calculate_levels(df):

    recent = df.tail(252)

    support = recent["Low"].quantile(0.10)

    resistance = recent["High"].quantile(0.90)

    return support, resistance


# ==========================================
# 7. POSITION SIZING
# ==========================================

def calculate_position_size(
    capital,
    risk_percent,
    entry_price,
    stop_price
):
    pass

# ==========================================
# BACKTESTING ENGINE
# ==========================================

def run_backtest(
    df,
    initial_capital=10000,
    transaction_cost_bps=10
):

    data = df.copy()

    if len(data) < 210:
        raise ValueError(
            "At least 210 trading observations are required."
        )

    # Generate trading signals using historical closing data.
    data["SIGNAL"] = (
        (data["SMA50"] > data["SMA200"]) &
        (data["RSI"] > 50)
    ).astype(int)

    # Execute signals on the following trading session.
    data["POSITION"] = data["SIGNAL"].shift(1).fillna(0)

    # Daily asset returns.
    data["MARKET_RETURN"] = data["Close"].pct_change()

    # Position changes represent transactions.
    data["TURNOVER"] = data["POSITION"].diff().abs()

    data["TURNOVER"] = data["TURNOVER"].fillna(
        data["POSITION"].abs()
    )

    # Transaction costs in basis points.
    cost_rate = transaction_cost_bps / 10000

    data["TRANSACTION_COST"] = (
        data["TURNOVER"] * cost_rate
    )

    # Strategy returns after transaction costs.
    data["STRATEGY_RETURN"] = (
        data["POSITION"] * data["MARKET_RETURN"]
        - data["TRANSACTION_COST"]
    )

    data["STRATEGY_RETURN"] = (
        data["STRATEGY_RETURN"].fillna(0)
    )

    data["BUY_HOLD_RETURN"] = (
        data["MARKET_RETURN"].fillna(0)
    )

    # Portfolio equity curves.
    data["STRATEGY_EQUITY"] = (
        initial_capital *
        (1 + data["STRATEGY_RETURN"]).cumprod()
    )

    data["BUY_HOLD_EQUITY"] = (
        initial_capital *
        (1 + data["BUY_HOLD_RETURN"]).cumprod()
    )

    # Calculate drawdowns.
    data["STRATEGY_DRAWDOWN"] = (
        data["STRATEGY_EQUITY"] /
        data["STRATEGY_EQUITY"].cummax()
    ) - 1

    data["BUY_HOLD_DRAWDOWN"] = (
        data["BUY_HOLD_EQUITY"] /
        data["BUY_HOLD_EQUITY"].cummax()
    ) - 1

    # Performance metrics.
    observations = len(data)

    years = observations / 252

    total_return = (
        data["STRATEGY_EQUITY"].iloc[-1] /
        initial_capital
    ) - 1

    buy_hold_return = (
        data["BUY_HOLD_EQUITY"].iloc[-1] /
        initial_capital
    ) - 1

    if years > 0:
        cagr = (
            data["STRATEGY_EQUITY"].iloc[-1] /
            initial_capital
        ) ** (1 / years) - 1
    else:
        cagr = 0

    daily_returns = data["STRATEGY_RETURN"]

    annual_volatility = (
        daily_returns.std() * np.sqrt(252)
    )

    if daily_returns.std() > 0:
        sharpe_ratio = (
            daily_returns.mean() /
            daily_returns.std()
        ) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    max_drawdown = abs(
        data["STRATEGY_DRAWDOWN"].min()
    )

    exposure = data["POSITION"].mean()

    trades = int(
        (data["TURNOVER"] > 0).sum()
    )

    metrics = {
        "Initial Capital": initial_capital,
        "Final Capital": data["STRATEGY_EQUITY"].iloc[-1],
        "Total Return": total_return * 100,
        "CAGR": cagr * 100,
        "Buy and Hold Return": buy_hold_return * 100,
        "Annual Volatility": annual_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown * 100,
        "Market Exposure": exposure * 100,
        "Transactions": trades
    }

    return data, metrics    

    risk_amount = capital * risk_percent / 100

    risk_per_share = abs(
        entry_price - stop_price
    )

    if risk_per_share <= 0:
        return 0

    shares = int(
        risk_amount / risk_per_share
    )

    return max(shares, 0)


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

st.sidebar.header("Analysis Settings")

capital = st.sidebar.number_input(
    "Investment Capital ($)",
    min_value=100.0,
    value=10000.0,
    step=500.0
)

risk_percent = st.sidebar.slider(
    "Maximum Planned Risk Per Trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

query = st.text_input(
    "Search for a company or stock ticker",
    placeholder="Example: Apple or AAPL"
)


if query:

    shares = None
    current_price = None

    with st.spinner("Searching for stocks..."):

        results = search_stock(query)

    if not results:

        st.error("No matching stocks found.")

        st.stop()

    selected = st.selectbox(
        "Select a company",
        results,
        format_func=lambda x: (
            f'{x["name"]} ({x["symbol"]})'
        )
    )

    ticker = selected["symbol"]

    with st.spinner(
        "Downloading historical data and calculating indicators..."
    ):

        history = load_history(ticker)

    if history is None or len(history) < 200:

        st.error(
            "Insufficient historical data "
            "for technical analysis."
        )

        st.stop()

    df = calculate_indicators(history)

    risks = calculate_risks(df)

    if risks is None:
        st.error("Unable to calculate historical risk.")
        st.stop()

    score, signal = calculate_score(df)

    support, resistance = calculate_levels(df)

    last = df.iloc[-1]

    current_price = float(last["Close"])

    # ======================================
    # 9. OVERVIEW
    # ======================================

    st.divider()

    st.header(
        f'{selected["name"]} ({ticker})'
    )

    st.caption(
        f"Historical data: "
        f"{df.index[0].date()} - {df.index[-1].date()}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        f"${current_price:,.2f}"
    )

    col2.metric(
        "RSI",
        f'{last["RSI"]:.2f}'
    )

    col3.metric(
        "MACD",
        f'{last["MACD"]:.3f}'
    )

    col4.metric(
        "Technical Score",
        f"{score}"
    )

    # ======================================
    # 10. ANALYSIS SIGNAL
    # ======================================

    st.subheader("Technical Analysis Result")

    if signal == "BULLISH":

        st.success(signal)

    elif signal == "BEARISH":

        st.error(signal)

    else:

        st.warning(signal)

    st.caption(
        "Technical score is a rule-based indicator, "
        "not a calibrated probability of future returns."
    )

    # ======================================
    # 11. PRICE CHART
    # ======================================

    st.subheader("Historical Price Chart")

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Stock Price",
            line=dict(
                color="#2196F3",
                width=2
            )
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA50"],
            name="SMA 50",
            line=dict(color="#FFA726")
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA200"],
            name="SMA 200",
            line=dict(color="#AB47BC")
        )
    )

    chart.update_layout(
        height=550,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Adjusted Price",
        hovermode="x unified"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    # ======================================
    # 12. RSI CHART
    # ======================================

    st.subheader("Relative Strength Index")

    rsi_chart = go.Figure()

    rsi_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color="#00C896")
        )
    )

    rsi_chart.add_hline(
        y=70,
        line_dash="dash",
        line_color="red"
    )

    rsi_chart.add_hline(
        y=30,
        line_dash="dash",
        line_color="green"
    )

    rsi_chart.update_layout(
        height=300,
        template="plotly_dark",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        rsi_chart,
        use_container_width=True
    )

    # ======================================
    # 13. RISK DASHBOARD
    # ======================================

    st.subheader("Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Annual Volatility",
        f'{risks["annual_volatility"]:.2f}%'
    )

    col2.metric(
        "Maximum Drawdown",
        f'{risks["max_drawdown"]:.2f}%'
    )

    col3.metric(
        "Historical VaR 95%",
        f'{risks["var_95"]:.2f}%'
    )

    col4.metric(
        "Historical CVaR 95%",
        f'{risks["cvar_95"]:.2f}%'
    )

    st.caption(
        "Historical risk estimates do not guarantee "
        "maximum future losses."
    )

    # ======================================
    # 14. SUPPORT AND RESISTANCE
    # ======================================

    st.subheader("Technical Price Levels")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Support",
        f"${support:.2f}"
    )

    col2.metric(
        "Current Price",
        f"${current_price:.2f}"
    )

    col3.metric(
        "Resistance",
        f"${resistance:.2f}"
    )


    # ==========================================
    # POSITION RISK CALCULATOR
    # ==========================================

    st.subheader("Position Risk Calculator")

    if (
        current_price is not None
        and pd.notna(current_price)
        and last["ATR"] is not None
        and pd.notna(last["ATR"])
        and last["ATR"] > 0
    ):

        stop_price = (
            current_price - 2 * float(last["ATR"])
        )

        shares = calculate_position_size(
            capital,
            risk_percent,
            current_price,
            stop_price
        )

        if shares is None:
            shares = 0

        risk_amount = (
            capital * risk_percent / 100
        )

        position_value = (
            shares * current_price
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Suggested Share Quantity",
            f"{shares:,}"
        )

        col2.metric(
            "Planned Risk Budget",
            f"${risk_amount:,.2f}"
        )

        col3.metric(
            "Position Value",
            f"${position_value:,.2f}"
        )

        st.write(
            f"Illustrative Stop Level: ${stop_price:.2f}"
        )

        st.caption(
            "This is an illustrative position-sizing calculation. "
            "Stop orders cannot guarantee execution at the specified price."
        )

    else:

        st.warning(
            "Insufficient price or volatility data "
            "to calculate position sizing."
        )

    # ======================================
    # 16. HISTORICAL DATA TABLE
    # ======================================

    st.subheader("Historical Technical Data")

    columns = [
        "Close",
        "SMA20",
        "SMA50",
        "SMA100",
        "SMA200",
        "RSI",
        "MACD",
        "ATR",
        "VOLATILITY"
    ]

    st.dataframe(
        df[columns].tail(100),
        use_container_width=True
    )

    # ======================================
    # 17. CSV EXPORT
    # ======================================

    csv = df.to_csv().encode("utf-8")

    st.download_button(
        label="Download Full Historical Analysis",
        data=csv,
        file_name=f"{ticker}_analysis.csv",
        mime="text/csv"
    )

    
# ==========================================
# BACKTESTING DASHBOARD
# ==========================================

st.divider()

st.header("Historical Strategy Backtesting")

st.write(
    "Test the technical trading strategy against "
    "historical market data."
)

transaction_cost = st.slider(
    "Transaction Cost (Basis Points)",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)

if st.button("Run Historical Backtest"):

    try:

        backtest_df, backtest_metrics = run_backtest(
            df,
            initial_capital=capital,
            transaction_cost_bps=transaction_cost
        )

        st.subheader("Backtesting Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Final Portfolio Value",
            f'${backtest_metrics["Final Capital"]:,.2f}'
        )

        col2.metric(
            "Total Strategy Return",
            f'{backtest_metrics["Total Return"]:.2f}%'
        )

        col3.metric(
            "Annualized Return",
            f'{backtest_metrics["CAGR"]:.2f}%'
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sharpe Ratio",
            f'{backtest_metrics["Sharpe Ratio"]:.2f}'
        )

        col2.metric(
            "Maximum Drawdown",
            f'{backtest_metrics["Maximum Drawdown"]:.2f}%'
        )

        col3.metric(
            "Annual Volatility",
            f'{backtest_metrics["Annual Volatility"]:.2f}%'
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Buy and Hold Return",
            f'{backtest_metrics["Buy and Hold Return"]:.2f}%'
        )

        col2.metric(
            "Market Exposure",
            f'{backtest_metrics["Market Exposure"]:.2f}%'
        )

        col3.metric(
            "Transactions",
            f'{backtest_metrics["Transactions"]}'
        )

        # Equity Curve

        st.subheader("Portfolio Equity Curve")

        equity_chart = go.Figure()

        equity_chart.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["STRATEGY_EQUITY"],
                name="Trading Strategy",
                line=dict(
                    color="#00C896",
                    width=2
                )
            )
        )

        equity_chart.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["BUY_HOLD_EQUITY"],
                name="Buy and Hold",
                line=dict(
                    color="#2196F3",
                    width=2
                )
            )
        )

        equity_chart.update_layout(
            template="plotly_dark",
            height=450,
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            hovermode="x unified"
        )

        st.plotly_chart(
            equity_chart,
            use_container_width=True
        )

        # Drawdown Chart

        st.subheader("Historical Drawdown")

        drawdown_chart = go.Figure()

        drawdown_chart.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["STRATEGY_DRAWDOWN"] * 100,
                name="Strategy Drawdown",
                fill="tozeroy",
                line=dict(color="#FF5252")
            )
        )

        drawdown_chart.update_layout(
            template="plotly_dark",
            height=300,
            yaxis_title="Drawdown (%)"
        )

        st.plotly_chart(
            drawdown_chart,
            use_container_width=True
        )

        # Export Results

        st.download_button(
            "Download Backtesting Results",
            data=backtest_df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_backtest.csv",
            mime="text/csv"
        )
# NEW BACKTESTING DASHBOARD
        st.success("Backtesting completed successfully.")

        st.warning(
            "Historical backtesting does not predict future "
            "performance. Results exclude taxes, market impact, "
            "financing costs and other possible trading expenses."
        )

    except Exception as error:

        st.error(f"Backtesting error: {error}")

else:

    st.info(
        "Enter a company name or stock ticker "
        "to start the analysis."
    )

    st.markdown(
        """
        ### Available Analysis Modules

        - Historical stock prices
        - Technical indicators
        - Trend identification
        - Momentum analysis
        - Volatility measurement
        - Historical risk assessment
        - Support and resistance levels
        - Position sizing
        - Historical data export
        """
    )


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Stock Analyzer | Independent Quantitative Analysis"
)

st.caption(
    "This application provides mathematical market analysis "
    "and does not guarantee investment returns or eliminate risk."
)
