
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Multi-Asset Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("AI MULTI-ASSET MARKET ANALYZER")

st.caption(
    "Stocks | ETFs | Crypto | Mutual Funds | Futures "
    "| Technical Analysis | Risk Management | Backtesting"
)

# ==========================================
# 1. ASSET SEARCH
# ==========================================

ASSET_TYPES = {
    "EQUITY": "Stock",
    "ETF": "ETF",
    "CRYPTOCURRENCY": "Cryptocurrency",
    "MUTUALFUND": "Mutual Fund",
    "FUTURE": "Futures Contract"
}

@st.cache_data(ttl=3600)
def search_assets(query):

    try:
        result = yf.Search(
            query,
            max_results=25,
            news_count=0
        )

        assets = []

        for item in result.quotes:

            quote_type = item.get("quoteType", "")

            if quote_type not in ASSET_TYPES:
                continue

            symbol = item.get("symbol")

            if not symbol:
                continue

            assets.append({
                "symbol": symbol,
                "name": (
                    item.get("shortname")
                    or item.get("longname")
                    or symbol
                ),
                "type": ASSET_TYPES[quote_type],
                "exchange": (
                    item.get("exchange")
                    or item.get("exchDisp")
                    or "Unknown"
                ),
                "currency": item.get("currency", "Unknown")
            })

        unique = {
            asset["symbol"]: asset
            for asset in assets
        }

        return list(unique.values())

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

    # Simple Moving Averages

    for period in [20, 50, 100, 200]:

        df[f"SMA{period}"] = (
            close.rolling(period).mean()
        )

    # Exponential Moving Averages

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

    df["RSI"] = (
        100 - (100 / (1 + rs))
    )

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

    direction = np.sign(
        close.diff()
    ).fillna(0)

    df["OBV"] = (
        direction * volume.fillna(0)
    ).cumsum()

    # Returns

    df["RETURNS"] = close.pct_change()

    # Annualized volatility

    df["VOLATILITY"] = (
        df["RETURNS"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    # Average volume

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

    running_max = (
        cumulative_returns.cummax()
    )

    drawdown = (
        cumulative_returns / running_max
    ) - 1

    var_95 = np.percentile(
        returns,
        5
    )

    tail_returns = returns[
        returns <= var_95
    ]

    cvar_95 = (
        tail_returns.mean()
        if len(tail_returns) > 0
        else var_95
    )

    annual_volatility = (
        returns.std()
        * np.sqrt(252)
        * 100
    )

    return {

        "annual_volatility": annual_volatility,

        "max_drawdown": (
            abs(drawdown.min()) * 100
        ),

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

    # Final signal

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
    stop_price,
    fractional=False
):

    if capital <= 0 or entry_price <= 0:
        return 0

    risk_amount = (
        capital * risk_percent / 100
    )

    risk_per_unit = abs(
        entry_price - stop_price
    )

    if risk_per_unit <= 0:
        return 0

    risk_quantity = (
        risk_amount / risk_per_unit
    )

    affordable_quantity = (
        capital / entry_price
    )

    quantity = min(
        risk_quantity,
        affordable_quantity
    )

    if fractional:

        return round(quantity, 8)

    return max(int(quantity), 0)


# ==========================================
# 8. BACKTESTING ENGINE
# ==========================================

def run_backtest(
    df,
    initial_capital=10000,
    transaction_cost_bps=10,
    annualization=252
):

    data = df.copy()

    if len(data) < 210:

        raise ValueError(
            "At least 210 trading observations are required."
        )

    # Generate trading signals

    data["SIGNAL"] = (

        (data["SMA50"] > data["SMA200"])
        & (data["RSI"] > 50)

    ).astype(int)

    # Execute signals on the following session

    data["POSITION"] = (
        data["SIGNAL"].shift(1).fillna(0)
    )

    # Market returns

    data["MARKET_RETURN"] = (
        data["Close"].pct_change()
    )

    # Calculate transactions

    data["TURNOVER"] = (
        data["POSITION"].diff().abs()
    )

    data["TURNOVER"] = (
        data["TURNOVER"].fillna(
            data["POSITION"].abs()
        )
    )

    # Transaction costs

    cost_rate = (
        transaction_cost_bps / 10000
    )

    data["TRANSACTION_COST"] = (
        data["TURNOVER"] * cost_rate
    )

    # Strategy returns

    data["STRATEGY_RETURN"] = (

        data["POSITION"] * data["MARKET_RETURN"]
        - data["TRANSACTION_COST"]

    ).fillna(0)

    # Buy and hold returns

    data["BUY_HOLD_RETURN"] = (
        data["MARKET_RETURN"].fillna(0)
    )

    # Equity curves

    data["STRATEGY_EQUITY"] = (

        initial_capital
        * (1 + data["STRATEGY_RETURN"]).cumprod()

    )

    data["BUY_HOLD_EQUITY"] = (

        initial_capital
        * (1 + data["BUY_HOLD_RETURN"]).cumprod()

    )

    # Drawdowns

    data["STRATEGY_DRAWDOWN"] = (

        data["STRATEGY_EQUITY"]
        / data["STRATEGY_EQUITY"].cummax()

    ) - 1

    data["BUY_HOLD_DRAWDOWN"] = (

        data["BUY_HOLD_EQUITY"]
        / data["BUY_HOLD_EQUITY"].cummax()

    ) - 1

    # Performance calculations

    observations = len(data)

    years = observations / annualization

    final_capital = (
        data["STRATEGY_EQUITY"].iloc[-1]
    )

    total_return = (
        final_capital / initial_capital
    ) - 1

    buy_hold_return = (

        data["BUY_HOLD_EQUITY"].iloc[-1]
        / initial_capital

    ) - 1

    if years > 0 and final_capital > 0:

        cagr = (

            (final_capital / initial_capital)
            ** (1 / years)

        ) - 1

    else:

        cagr = 0

    daily_returns = data["STRATEGY_RETURN"]

    annual_volatility = (

        daily_returns.std()
        * np.sqrt(annualization)

    )

    if daily_returns.std() > 0:

        sharpe_ratio = (

            daily_returns.mean()
            / daily_returns.std()

        ) * np.sqrt(annualization)

    else:

        sharpe_ratio = 0

    max_drawdown = abs(
        data["STRATEGY_DRAWDOWN"].min()
    )

    exposure = (
        data["POSITION"].mean()
    )

    trades = int(
        (data["TURNOVER"] > 0).sum()
    )

    # Metrics dictionary

    metrics = {

        "Initial Capital": initial_capital,

        "Final Capital": final_capital,

        "Total Return": total_return * 100,

        "CAGR": cagr * 100,

        "Buy and Hold Return": (
            buy_hold_return * 100
        ),

        "Annual Volatility": (
            annual_volatility * 100
        ),

        "Sharpe Ratio": sharpe_ratio,

        "Maximum Drawdown": (
            max_drawdown * 100
        ),

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
# 9. MAIN APPLICATION SETTINGS
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

st.sidebar.divider()

st.sidebar.subheader("Supported Asset Classes")

st.sidebar.write("📈 Stocks")
st.sidebar.write("📊 ETFs")
st.sidebar.write("₿ Cryptocurrencies")
st.sidebar.write("💼 Mutual Funds")
st.sidebar.write("📉 Futures Contracts")

# ==========================================
# 10. ASSET SEARCH
# ==========================================

query = st.text_input(
    "Search for any financial instrument",
    placeholder="AAPL, SPY, BTC-USD, ETH-USD, GLD, QQQ"
)

if query:

    with st.spinner("Searching financial markets..."):

        results = search_assets(query)

    if not results:

        st.error("No matching assets found.")

        st.stop()

    selected = st.selectbox(
        "Select your financial instrument",
        results,
        format_func=lambda x: (
            f'{x["name"]} | {x["symbol"]} | '
            f'{x["type"]} | {x["exchange"]}'
        )
    )

    ticker = selected["symbol"]

    asset_type = selected["type"]

    st.info(
        f'Asset: {asset_type} | '
        f'Exchange: {selected["exchange"]} | '
        f'Currency: {selected["currency"]}'
    )

    # ======================================
    # 11. DOWNLOAD MARKET DATA
    # ======================================

    with st.spinner(
        "Downloading historical market data..."
    ):

        history = load_history(ticker)

    if history is None or len(history) < 200:

        st.error(
            "Insufficient historical data "
            "for technical analysis."
        )

        st.stop()

    # Calculate indicators

    df = calculate_indicators(history)

    risks = calculate_risks(df)

    if risks is None:

        st.error("Unable to calculate historical risk.")

        st.stop()

    score, signal = calculate_score(df)

    support, resistance = calculate_levels(df)

    last = df.iloc[-1]

    current_price = float(last["Close"])

    fractional = asset_type == "Cryptocurrency"

    annualization = (
        365 if fractional else 252
    )

    # ======================================
    # 12. ASSET OVERVIEW
    # ======================================

    st.divider()

    st.header(
        f'{selected["name"]} ({ticker})'
    )

    st.caption(
        f'Asset Type: {asset_type} | '
        f'Historical Data: {df.index[0].date()} '
        f'- {df.index[-1].date()}'
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        f"${current_price:,.4f}"
    )

    col2.metric(
        "RSI",
        f'{last["RSI"]:.2f}'
    )

    col3.metric(
        "MACD",
        f'{last["MACD"]:.4f}'
    )

    col4.metric(
        "Technical Score",
        str(score)
    )

    # ======================================
    # 13. TECHNICAL SIGNAL
    # ======================================

    st.subheader("Technical Analysis Result")

    if signal == "BULLISH":

        st.success(signal)

    elif signal == "BEARISH":

        st.error(signal)

    else:

        st.warning(signal)

    st.caption(
        "This technical score is a rule-based indicator, "
        "not a calibrated probability of future returns."
    )

    # ======================================
    # 14. HISTORICAL PRICE CHART
    # ======================================

    st.subheader("Historical Price Chart")

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Market Price",
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

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_UPPER"],
            name="Bollinger Upper",
            line=dict(dash="dot")
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_LOWER"],
            name="Bollinger Lower",
            line=dict(dash="dot")
        )
    )

    chart.update_layout(
        height=550,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Market Price",
        hovermode="x unified"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    # ======================================
    # 15. RSI CHART
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
    # 16. MACD CHART
    # ======================================

    st.subheader("MACD Momentum Analysis")

    macd_chart = go.Figure()

    macd_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD"
        )
    )

    macd_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_SIGNAL"],
            name="Signal Line"
        )
    )

    macd_chart.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_HIST"],
            name="MACD Histogram"
        )
    )

    macd_chart.update_layout(
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        macd_chart,
        use_container_width=True
    )

    # ======================================
    # 17. RISK DASHBOARD
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
    # 18. SUPPORT AND RESISTANCE
    # ======================================

    st.subheader("Technical Price Levels")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Support",
        f"${support:.4f}"
    )

    col2.metric(
        "Current Price",
        f"${current_price:.4f}"
    )

    col3.metric(
        "Resistance",
        f"${resistance:.4f}"
    )

    # ======================================
    # 19. POSITION RISK CALCULATOR
    # ======================================

    st.subheader("Position Risk Calculator")

    atr = float(last["ATR"])

    if (
        pd.notna(atr)
        and atr > 0
        and current_price > 0
    ):

        stop_price = (
            current_price - 2 * atr
        )

        if stop_price <= 0:

            stop_price = current_price * 0.5

        quantity = calculate_position_size(
            capital,
            risk_percent,
            current_price,
            stop_price,
            fractional=fractional
        )

        risk_amount = (
            capital * risk_percent / 100
        )

        position_value = (
            quantity * current_price
        )

        actual_risk = (
            quantity * abs(
                current_price - stop_price
            )
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Suggested Quantity",
            f"{quantity:,.8f}"
            if fractional
            else f"{quantity:,}"
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
            f"Illustrative Stop Level: ${stop_price:,.4f}"
        )

        st.write(
            f"Estimated Planned Risk: ${actual_risk:,.2f}"
        )

        st.caption(
            "Futures contract multipliers, margin, fees and "
            "slippage are not included. Stop orders cannot "
            "guarantee execution at the specified price."
        )

    else:

        st.warning(
            "Insufficient data for position sizing."
        )

    # ======================================
    # 20. HISTORICAL TECHNICAL DATA
    # ======================================

    st.subheader("Historical Technical Data")

    columns = [
        "Close",
        "SMA20",
        "SMA50",
        "SMA100",
        "SMA200",
        "EMA12",
        "EMA26",
        "RSI",
        "MACD",
        "ATR",
        "VOLATILITY",
        "OBV"
    ]

    st.dataframe(
        df[columns].tail(100),
        use_container_width=True
    )

    # ======================================
    # 21. CSV EXPORT
    # ======================================

    st.download_button(
        label="Download Full Historical Analysis",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_analysis.csv",
        mime="text/csv"
    )

# ==========================================
# 22. HISTORICAL STRATEGY BACKTESTING
# ==========================================

if query:

    st.divider()

    st.header("Historical Strategy Backtesting")

    st.write(
        "Test the technical trading strategy against "
        "historical market data and compare it with "
        "a buy-and-hold strategy."
    )

    transaction_cost = st.slider(
        "Transaction Cost (Basis Points)",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )

    if st.button(
        "Run Historical Backtest",
        type="primary"
    ):

        try:

            backtest_df, metrics = run_backtest(
                df,
                initial_capital=capital,
                transaction_cost_bps=transaction_cost,
                annualization=annualization
            )

            # ==================================
            # PERFORMANCE METRICS
            # ==================================

            st.subheader("Backtesting Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Final Portfolio Value",
                f'${metrics["Final Capital"]:,.2f}'
            )

            col2.metric(
                "Total Strategy Return",
                f'{metrics["Total Return"]:.2f}%'
            )

            col3.metric(
                "Annualized Return",
                f'{metrics["CAGR"]:.2f}%'
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Sharpe Ratio",
                f'{metrics["Sharpe Ratio"]:.2f}'
            )

            col2.metric(
                "Maximum Drawdown",
                f'{metrics["Maximum Drawdown"]:.2f}%'
            )

            col3.metric(
                "Annual Volatility",
                f'{metrics["Annual Volatility"]:.2f}%'
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Buy and Hold Return",
                f'{metrics["Buy and Hold Return"]:.2f}%'
            )

            col2.metric(
                "Market Exposure",
                f'{metrics["Market Exposure"]:.2f}%'
            )

            col3.metric(
                "Transactions",
                f'{metrics["Transactions"]}'
            )

            # ==================================
            # EQUITY CURVE
            # ==================================

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

            # ==================================
            # DRAWDOWN CHART
            # ==================================

            st.subheader("Historical Drawdown")

            drawdown_chart = go.Figure()

            drawdown_chart.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=(
                        backtest_df["STRATEGY_DRAWDOWN"]
                        * 100
                    ),
                    name="Strategy Drawdown",
                    fill="tozeroy",
                    line=dict(color="#FF5252")
                )
            )

            drawdown_chart.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=(
                        backtest_df["BUY_HOLD_DRAWDOWN"]
                        * 100
                    ),
                    name="Buy and Hold Drawdown",
                    line=dict(color="#2196F3")
                )
            )

            drawdown_chart.update_layout(
                template="plotly_dark",
                height=350,
                yaxis_title="Drawdown (%)",
                xaxis_title="Date"
            )

            st.plotly_chart(
                drawdown_chart,
                use_container_width=True
            )

            # ==================================
            # BACKTEST DATA EXPORT
            # ==================================

            st.subheader("Export Backtesting Results")

            st.download_button(
                label="Download Backtesting Results",
                data=backtest_df.to_csv().encode("utf-8"),
                file_name=f"{ticker}_backtest.csv",
                mime="text/csv"
            )

            st.success(
                "Historical backtesting completed successfully."
            )

            st.warning(
                "Historical backtesting does not predict future "
                "performance. Results exclude taxes, market impact, "
                "financing costs and other possible trading expenses."
            )

        except Exception as error:

            st.error(
                f"Backtesting error: {error}"
            )


# ==========================================
# 23. WELCOME SCREEN
# ==========================================

else:

    st.info(
        "Enter a company name, stock ticker, ETF "
        "or cryptocurrency to start your analysis."
    )

    st.subheader("Available Analysis Modules")

    st.markdown("""
    ### Supported Financial Instruments

    - Stocks and equities
    - Exchange-traded funds (ETFs)
    - Cryptocurrencies
    - Mutual funds
    - Futures contracts

    ### Quantitative Analysis

    - Historical market prices
    - SMA and EMA indicators
    - RSI momentum analysis
    - MACD analysis
    - Bollinger Bands
    - Average True Range
    - On Balance Volume
    - Historical volatility

    ### Risk Management

    - Maximum drawdown
    - Historical VaR
    - Historical CVaR
    - Position sizing
    - Risk per trade
    - Support and resistance

    ### Strategy Backtesting

    - Historical strategy simulation
    - Buy and hold comparison
    - Portfolio equity curves
    - Sharpe ratio
    - CAGR
    - Transaction cost modeling
    - Drawdown analysis

    ### Data Export

    - Historical analysis CSV
    - Backtesting results CSV
    """)


# ==========================================
# 24. APPLICATION FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Multi-Asset Analyzer | Independent Quantitative Analysis"
)

st.caption(
    "Historical market data provided through Yahoo Finance "
    "using the yfinance Python library."
)

st.caption(
    "Asset availability, historical coverage and data quality "
    "depend on the data provider."
)

st.caption(
    "This application provides mathematical market analysis "
    "and educational risk estimates. It does not execute trades "
    "or guarantee investment returns."
)
