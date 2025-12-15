import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Nifty 50 RSI Divergence Scanner",
    page_icon="📈",
    layout="wide"
)

# Nifty 50 symbols
NIFTY_50_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
    'TECHM.NS', 'TMPV.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'INDUSINDBK.NS', 'TATASTEEL.NS', 'DRREDDY.NS',
    'JSWSTEEL.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HINDALCO.NS',
    'HEROMOTOCO.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'BPCL.NS', 'SBILIFE.NS',
    'TATACONSUM.NS', 'BAJAJ-AUTO.NS', 'LTIM.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS'
]

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_peaks_troughs(data, order=3):
    """Find local peaks and troughs with lower order for better detection"""
    price_peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    price_troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    rsi_peaks = argrelextrema(data['RSI'].values, np.greater, order=order)[0]
    rsi_troughs = argrelextrema(data['RSI'].values, np.less, order=order)[0]
    return price_peaks, price_troughs, rsi_peaks, rsi_troughs

def calculate_divergence_strength(price_1, price_2, rsi_1, rsi_2, div_type):
    """Calculate divergence strength score"""
    score = 0
    
    # Price change magnitude
    price_change_pct = abs((price_2 - price_1) / price_1 * 100)
    if price_change_pct > 5:
        score += 30
    elif price_change_pct > 3:
        score += 20
    elif price_change_pct > 1:
        score += 10
    
    # RSI divergence magnitude
    rsi_divergence = abs(rsi_2 - rsi_1)
    if rsi_divergence > 15:
        score += 30
    elif rsi_divergence > 10:
        score += 20
    elif rsi_divergence > 5:
        score += 10
    
    # RSI extremes bonus
    if div_type == 'Bullish':
        if rsi_2 < 30:
            score += 20
        elif rsi_2 < 40:
            score += 10
    else:
        if rsi_2 > 70:
            score += 20
        elif rsi_2 > 60:
            score += 10
    
    # Base score
    score += 20
    return min(score, 100)

def detect_bullish_divergence(data, lookback=50):
    """Detect bullish divergence"""
    price_peaks, price_troughs, rsi_peaks, rsi_troughs = find_peaks_troughs(data)
    divergences = []
    
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        recent_price_troughs = [i for i in price_troughs if i >= max(0, len(data) - lookback)]
        recent_rsi_troughs = [i for i in rsi_troughs if i >= max(0, len(data) - lookback)]
        
        if len(recent_price_troughs) >= 2 and len(recent_rsi_troughs) >= 2:
            pt1, pt2 = recent_price_troughs[-2], recent_price_troughs[-1]
            rt1, rt2 = recent_rsi_troughs[-2], recent_rsi_troughs[-1]
            
            # Price making lower low, RSI making higher low
            if (data['Close'].iloc[pt2] < data['Close'].iloc[pt1] and 
                data['RSI'].iloc[rt2] > data['RSI'].iloc[rt1]):
                
                strength = calculate_divergence_strength(
                    data['Close'].iloc[pt1], data['Close'].iloc[pt2],
                    data['RSI'].iloc[rt1], data['RSI'].iloc[rt2], 'Bullish'
                )
                
                divergences.append({'type': 'Bullish', 'strength': strength})
    
    return divergences

def detect_bearish_divergence(data, lookback=50):
    """Detect bearish divergence"""
    price_peaks, price_troughs, rsi_peaks, rsi_troughs = find_peaks_troughs(data)
    divergences = []
    
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        recent_price_peaks = [i for i in price_peaks if i >= max(0, len(data) - lookback)]
        recent_rsi_peaks = [i for i in rsi_peaks if i >= max(0, len(data) - lookback)]
        
        if len(recent_price_peaks) >= 2 and len(recent_rsi_peaks) >= 2:
            pp1, pp2 = recent_price_peaks[-2], recent_price_peaks[-1]
            rp1, rp2 = recent_rsi_peaks[-2], recent_rsi_peaks[-1]
            
            # Price making higher high, RSI making lower high
            if (data['Close'].iloc[pp2] > data['Close'].iloc[pp1] and 
                data['RSI'].iloc[rp2] < data['RSI'].iloc[rp1]):
                
                strength = calculate_divergence_strength(
                    data['Close'].iloc[pp1], data['Close'].iloc[pp2],
                    data['RSI'].iloc[rp1], data['RSI'].iloc[rp2], 'Bearish'
                )
                
                divergences.append({'type': 'Bearish', 'strength': strength})
    
    return divergences

def analyze_single_timeframe(symbol, period, interval):
    """Analyze stock for one timeframe"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        if len(data) < 30:
            return None
        
        # Calculate RSI
        data['RSI'] = calculate_rsi(data)
        data = data.dropna()
        
        # Detect divergences
        bullish_div = detect_bullish_divergence(data)
        bearish_div = detect_bearish_divergence(data)
        
        result = {
            'has_bullish': len(bullish_div) > 0,
            'has_bearish': len(bearish_div) > 0,
            'bullish_strength': max([d['strength'] for d in bullish_div]) if bullish_div else 0,
            'bearish_strength': max([d['strength'] for d in bearish_div]) if bearish_div else 0,
            'current_rsi': data['RSI'].iloc[-1]
        }
        
        return result
        
    except Exception as e:
        return None

def analyze_stock(symbol):
    """Analyze stock across 2 timeframes: 15m (5d) and 1h (1mo)"""
    
    # 15-minute for 5 days (more data for better peak detection)
    tf_15m = analyze_single_timeframe(symbol, '5d', '15m')
    
    # 1-hour for 1 month (more data for better peak detection)
    tf_1h = analyze_single_timeframe(symbol, '1mo', '1h')
    
    # Check if we got data
    if tf_15m is None and tf_1h is None:
        return None  # No data available
    
    # Get current price (use adjusted close when market is closed)
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1d')
        # Try to get adjusted close first, fallback to regular close
        if 'Close' in hist.columns and len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = 0
    except:
        current_price = 0
    
    # Calculate combined score
    score = 0
    signal = "NEUTRAL"
    details = []
    
    if tf_15m and tf_1h:
        # Check for bullish signals
        if tf_15m['has_bullish'] or tf_1h['has_bullish']:
            score = (tf_15m['bullish_strength'] + tf_1h['bullish_strength']) / 2
            
            if tf_15m['has_bullish'] and tf_1h['has_bullish']:
                score += 20  # Bonus for both timeframes agreeing
                signal = "STRONG BUY"
            else:
                signal = "BUY"
            
            if tf_15m['has_bullish']:
                details.append(f"15m: {tf_15m['bullish_strength']:.0f}")
            if tf_1h['has_bullish']:
                details.append(f"1h: {tf_1h['bullish_strength']:.0f}")
        
        # Check for bearish signals
        elif tf_15m['has_bearish'] or tf_1h['has_bearish']:
            score = (tf_15m['bearish_strength'] + tf_1h['bearish_strength']) / 2
            
            if tf_15m['has_bearish'] and tf_1h['has_bearish']:
                score += 20  # Bonus for both timeframes agreeing
                signal = "STRONG SELL"
            else:
                signal = "SELL"
            
            if tf_15m['has_bearish']:
                details.append(f"15m: {tf_15m['bearish_strength']:.0f}")
            if tf_1h['has_bearish']:
                details.append(f"1h: {tf_1h['bearish_strength']:.0f}")
    
    # Return stocks with signals
    if signal != "NEUTRAL":
        return {
            'symbol': symbol.replace('.NS', ''),
            'price': current_price,
            'score': min(score, 100),
            'signal': signal,
            'details': ', '.join(details) if details else '-',
            'rsi_15m': tf_15m['current_rsi'] if tf_15m else None,
            'rsi_1h': tf_1h['current_rsi'] if tf_1h else None
        }
    
    # Return empty dict for "no signal but has data"
    return {}

# Main App
st.title("📈 Nifty 50 RSI Divergence Scanner")
st.markdown("**Scanning 15-min (5 days) + 1-hour (1 month) timeframes**")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("🔍 **Timeframes:** 15-minute interval (5 days) + 1-hour interval (1 month)")
with col2:
    if st.button("🔄 Scan Now", type="primary", width="stretch"):
        st.cache_data.clear()

st.divider()

# Scanning
with st.spinner("Scanning Nifty 50 stocks..."):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    no_signal_count = 0
    no_data_count = 0
    total = len(NIFTY_50_SYMBOLS)
    
    for i, symbol in enumerate(NIFTY_50_SYMBOLS):
        status_text.text(f"Analyzing {symbol.replace('.NS', '')}... ({i+1}/{total})")
        result = analyze_stock(symbol)
        
        if result is None:
            # No data available at all
            no_data_count += 1
        elif result == {}:
            # Data available but no divergence signal
            no_signal_count += 1
        else:
            # Has a signal (BUY or SELL)
            results.append(result)
            
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    status_text.empty()

# Separate BUY and SELL signals
buy_signals = [r for r in results if 'BUY' in r['signal']]
sell_signals = [r for r in results if 'SELL' in r['signal']]

# Sort by score
buy_signals.sort(key=lambda x: x['score'], reverse=True)
sell_signals.sort(key=lambda x: x['score'], reverse=True)

# Display results
st.success(f"✅ Scan completed at {datetime.now().strftime('%H:%M:%S')}")

# Metrics with all counters
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🟢 BUY Signals", len(buy_signals))
col2.metric("🔴 SELL Signals", len(sell_signals))
col3.metric("⚪ No Signal", no_signal_count)
col4.metric("❌ No Data", no_data_count)
col5.metric("📊 Total Scanned", total)

st.divider()

# Display tables side by side
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🟢 BUY Signals")
    if buy_signals:
        df_buy = pd.DataFrame(buy_signals)
        df_buy['Rank'] = range(1, len(df_buy) + 1)
        df_buy = df_buy[['Rank', 'symbol', 'score', 'signal', 'price', 'details']]
        df_buy.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
        df_buy['Score'] = df_buy['Score'].round(1)
        df_buy['Price (₹)'] = df_buy['Price (₹)'].round(2)
        st.dataframe(df_buy, width="stretch", hide_index=True)
    else:
        st.info("No BUY signals detected")

with col_right:
    st.subheader("🔴 SELL Signals")
    if sell_signals:
        df_sell = pd.DataFrame(sell_signals)
        df_sell['Rank'] = range(1, len(df_sell) + 1)
        df_sell = df_sell[['Rank', 'symbol', 'score', 'signal', 'price', 'details']]
        df_sell.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
        df_sell['Score'] = df_sell['Score'].round(1)
        df_sell['Price (₹)'] = df_sell['Price (₹)'].round(2)
        st.dataframe(df_sell, width="stretch", hide_index=True)
    else:
        st.info("No SELL signals detected")

st.divider()
st.caption("💡 **Score Guide:** 80+ = Very Strong | 60-79 = Strong | 40-59 = Moderate | Below 40 = Weak")
st.caption("📌 **Signal Types:** STRONG BUY/SELL = Both timeframes agree | BUY/SELL = One timeframe shows divergence")
