import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# --- Configuración Streamlit ---
st.set_page_config(page_title="Modelo Inversión Cripto Spot (Binance)", layout="wide")
st.title("📊 Modelo de Inversión Cripto Spot (Binance)")
st.markdown("Análisis técnico con indicadores y backtesting para ETH, LINK, OP y ARB a 30 días.")

# --- Sidebar ---
symbols = {
    'Ethereum (ETH)': 'ETHUSDT',
    'Chainlink (LINK)': 'LINKUSDT',
    'Optimism (OP)': 'OPUSDT',
    'Arbitrum (ARB)': 'ARBUSDT'
}

symbol_name = st.sidebar.selectbox("Criptomoneda", list(symbols.keys()))
symbol = symbols[symbol_name]
capital = st.sidebar.number_input("Capital inicial ($)", 100, 100000, 1000, 100)
rsi_buy = st.sidebar.slider("RSI para comprar", 10, 50, 30)
rsi_sell = st.sidebar.slider("RSI para vender", 50, 90, 70)

# --- Obtener datos de Binance ---
@st.cache_data(ttl=1800)
def get_klines(symbol, interval='1h', limit=720):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df.sort_index()

df = get_klines(symbol)

# --- Análisis técnico ---
def analyze(df):
    df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['buy'] = (df['rsi'] < rsi_buy) & (df['close'] > df['sma50']) & (df['macd'] > df['macd_signal'])
    df['sell'] = (df['rsi'] > rsi_sell) & (df['close'] < df['sma50']) & (df['macd'] < df['macd_signal'])
    return df

df = analyze(df)

# --- Backtesting ---
def backtest(df, initial_capital):
    capital = initial_capital
    in_position = False
    entry_price = 0
    trades = []
    returns = []
    peak = capital
    max_drawdown = 0

    for i, row in df.iterrows():
        if row['buy'] and not in_position:
            entry_price = row['close']
            entry_time = i
            in_position = True
        elif row['sell'] and in_position:
            exit_price = row['close']
            profit = (exit_price - entry_price) / entry_price
            capital *= 1 + profit
            returns.append(profit)
            trades.append({
                'entry_time': entry_time,
                'exit_time': i,
                'entry': entry_price,
                'exit': exit_price,
                'profit_pct': round(profit * 100, 2),
                'capital': round(capital, 2)
            })
            peak = max(peak, capital)
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
            in_position = False

    win_rate = round(100 * sum([1 for r in returns if r > 0]) / len(returns), 2) if returns else 0
    sharpe_ratio = round(np.mean(returns) / np.std(returns) * np.sqrt(24 * 30), 2) if len(returns) > 1 else 0

    return round(capital, 2), trades, round(max_drawdown * 100, 2), win_rate, sharpe_ratio

final_balance, trades, max_dd, win_rate, sharpe = backtest(df, capital)

# --- Mostrar métricas ---
col1, col2, col3 = st.columns(3)
col1.metric("💰 Capital final", f"${final_balance}")
col2.metric("📊 Nº operaciones", len(trades))
col3.metric("📉 Máx. drawdown", f"{max_dd} %")

col4, col5 = st.columns(2)
col4.metric("✅ Win rate", f"{win_rate} %")
col5.metric("📈 Sharpe Ratio", f"{sharpe}")

# --- Gráfico ---
def plot_chart(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['close'], label='Precio', color='blue')
    ax.plot(df.index, df['sma50'], label='SMA50', linestyle='--', color='orange')
    ax.plot(df.index, df['ema20'], label='EMA20', linestyle='--', color='purple')
    ax.plot(df.index, df['bb_upper'], label='BB Upper', linestyle='dotted', color='gray')
    ax.plot(df.index, df['bb_lower'], label='BB Lower', linestyle='dotted', color='gray')
    ax.scatter(df[df['buy']].index, df[df['buy']]['close'], label='Buy', marker='^', color='green')
    ax.scatter(df[df['sell']].index, df[df['sell']]['close'], label='Sell', marker='v', color='red')
    ax.set_title(f"Señales en {symbol_name}")
    ax.set_ylabel("Precio USDT")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

plot_chart(df)

if trades:
    st.subheader("📄 Operaciones ejecutadas")
    st.dataframe(pd.DataFrame(trades).sort_values(by='exit_time'))

st.subheader("📈 Últimos datos")
st.dataframe(df[['close', 'sma50', 'ema20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'buy', 'sell']].tail(10))
