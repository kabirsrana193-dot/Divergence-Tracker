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
    'TECHM.NS', 'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
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
    """Find local peaks and troughs"""
    price_peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    price_troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    rsi_peaks = argrelextrema(data['RSI'].values, np.greater, order=order)[0]
    rsi_troughs = argrelextrema(data['RSI'].values, np.less, order=order)[0]
    return price_peaks, price_troughs, rsi_peaks, rsi_troughs

def calculate_divergence_strength(price_1, price_2, rsi_1, rsi_2, div_type):
    """Calculate divergence strength score"""
    score = 0
    price_change_pct = abs((price_2 - price_1) / price_1 * 100)
    if price_change_pct > 5:
        score += 30
    elif price_change_pct > 3:
        score += 20
    elif price_change_pct > 1:
        score += 10
    
    rsi_divergence = abs(rsi_2 - rsi_1)
    if rsi_divergence > 15:
        score += 30
    elif rsi_divergence > 10:
        score += 20
    elif rsi_divergence > 5:
        score += 10
    
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
    
    score += 20
    return min(score, 100)

def detect_bullish_divergence(data, lookback=50):
    """Detect bullish divergence"""
    price_peaks, price_troughs, rsi_peaks, rsi_troughs = find_peaks_troughs(data)
    divergences = []
    
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        recent_price_troughs = [i for i in price_troughs if i >= len(data) - lookback]
        recent_rsi_troughs = [i for i in rsi_troughs if i >= len(data) - lookback]
        
        if len(recent_price_troughs) >= 2 and len(recent_rsi_troughs) >= 2:
            pt1, pt2 = recent_price_troughs[-2], recent_price_troughs[-1]
            rt1, rt2 = recent_rsi_troughs[-2], recent_rsi_troughs[-1]
            
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
        recent_price_peaks = [i for i in price_peaks if i >= len(data) - lookback]
        recent_rsi_peaks = [i for i in rsi_peaks if i >= len(data) - lookback]
        
        if len(recent_price_peaks) >= 2 and len(recent_rsi_peaks) >= 2:
            pp1, pp2 = recent_price_peaks[-2], recent_price_peaks[-1]
            rp1, rp2 = recent_rsi_peaks[-2], recent_rsi_peaks[-1]
            
            if (data['Close'].iloc[pp2] > data['Close'].iloc[pp1] and 
                data['RSI'].iloc[rp2] < data['RSI'].iloc[rp1]):
                
                strength = calculate_divergence_strength(
                    data['Close'].iloc[pp1], data['Close'].iloc[pp2],
                    data['RSI'].iloc[rp1], data['RSI'].iloc[rp2], 'Bearish'
                )
                
                divergences.append({'type': 'Bearish', 'strength': strength})
    
    return divergences

@st.cache_data(ttl=300)  # Cache for 5 minutes
def analyze_single_timeframe(symbol, period, interval):
    """Analyze stock for one timeframe"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        if len(data) < 30:  # Reduced minimum data requirement
            return None
        
        data['RSI'] = calculate_rsi(data)
        data = data.dropna()
        
        if len(data) < 20:  # Check again after dropna
            return None
        
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

def analyze_stock_combined(symbol, progress_bar=None, status_text=None):
    """Analyze stock across multiple timeframes"""
    if status_text:
        status_text.text(f"Analyzing {symbol.replace('.NS', '')}...")
    
    # Intraday
    tf_15m = analyze_single_timeframe(symbol, '5d', '15m')
    tf_1h = analyze_single_timeframe(symbol, '1mo', '1h')
    
    # Longer-term
    tf_5d = analyze_single_timeframe(symbol, '5d', '1d')
    tf_3mo = analyze_single_timeframe(symbol, '3mo', '1d')
    
    # Get current price
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
    except:
        current_price = 0
    
    # Calculate INTRADAY score
    intraday_score = 0
    intraday_signal = "NEUTRAL"
    intraday_details = []
    
    if tf_15m and tf_1h:
        if tf_15m['has_bullish'] or tf_1h['has_bullish']:
            intraday_score = (tf_15m['bullish_strength'] + tf_1h['bullish_strength']) / 2
            
            if tf_15m['has_bullish'] and tf_1h['has_bullish']:
                intraday_score += 20
                intraday_signal = "STRONG BUY"
            else:
                intraday_signal = "BUY"
                
            intraday_details.append(f"15m: {tf_15m['bullish_strength']:.0f}" if tf_15m['has_bullish'] else "15m: -")
            intraday_details.append(f"1h: {tf_1h['bullish_strength']:.0f}" if tf_1h['has_bullish'] else "1h: -")
        
        elif tf_15m['has_bearish'] or tf_1h['has_bearish']:
            intraday_score = (tf_15m['bearish_strength'] + tf_1h['bearish_strength']) / 2
            
            if tf_15m['has_bearish'] and tf_1h['has_bearish']:
                intraday_score += 20
                intraday_signal = "STRONG SELL"
            else:
                intraday_signal = "SELL"
                
            intraday_details.append(f"15m: {tf_15m['bearish_strength']:.0f}" if tf_15m['has_bearish'] else "15m: -")
            intraday_details.append(f"1h: {tf_1h['bearish_strength']:.0f}" if tf_1h['has_bearish'] else "1h: -")
    
    # Calculate LONGER-TERM score
    longer_score = 0
    longer_signal = "NEUTRAL"
    longer_details = []
    
    if tf_5d and tf_3mo:
        if tf_5d['has_bullish'] or tf_3mo['has_bullish']:
            longer_score = (tf_5d['bullish_strength'] + tf_3mo['bullish_strength']) / 2
            
            if tf_5d['has_bullish'] and tf_3mo['has_bullish']:
                longer_score += 20
                longer_signal = "STRONG BUY"
            else:
                longer_signal = "BUY"
                
            longer_details.append(f"5d: {tf_5d['bullish_strength']:.0f}" if tf_5d['has_bullish'] else "5d: -")
            longer_details.append(f"3mo: {tf_3mo['bullish_strength']:.0f}" if tf_3mo['has_bullish'] else "3mo: -")
        
        elif tf_5d['has_bearish'] or tf_3mo['has_bearish']:
            longer_score = (tf_5d['bearish_strength'] + tf_3mo['bearish_strength']) / 2
            
            if tf_5d['has_bearish'] and tf_3mo['has_bearish']:
                longer_score += 20
                longer_signal = "STRONG SELL"
            else:
                longer_signal = "SELL"
                
            longer_details.append(f"5d: {tf_5d['bearish_strength']:.0f}" if tf_5d['has_bearish'] else "5d: -")
            longer_details.append(f"3mo: {tf_3mo['bearish_strength']:.0f}" if tf_3mo['has_bearish'] else "3mo: -")
    
    if intraday_signal != "NEUTRAL" or longer_signal != "NEUTRAL":
        return {
            'symbol': symbol.replace('.NS', ''),
            'price': current_price,
            'intraday_score': min(intraday_score, 100),
            'intraday_signal': intraday_signal,
            'intraday_details': ', '.join(intraday_details),
            'longer_score': min(longer_score, 100),
            'longer_signal': longer_signal,
            'longer_details': ', '.join(longer_details)
        }
    
    return None

def get_signal_color(signal):
    """Get color for signal"""
    if "BUY" in signal:
        return "🟢"
    elif "SELL" in signal:
        return "🔴"
    return "⚪"

# Main App
st.title("📈 Nifty 50 RSI Divergence Scanner")
st.markdown("**Real-time divergence analysis across multiple timeframes**")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown("**Intraday:** 15-min + 1-hour")
with col2:
    st.markdown("**Longer-term:** Daily 5d + 3mo")
with col3:
    if st.button("🔄 Scan Now", type="primary", use_container_width=True):
        st.rerun()

st.divider()

# Scanning
with st.spinner("Scanning Nifty 50 stocks..."):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(NIFTY_50_SYMBOLS)
    
    for i, symbol in enumerate(NIFTY_50_SYMBOLS):
        result = analyze_stock_combined(symbol, progress_bar, status_text)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    status_text.empty()

# Separate signals
intraday_buys = [r for r in results if 'BUY' in r['intraday_signal']]
intraday_sells = [r for r in results if 'SELL' in r['intraday_signal']]
longer_buys = [r for r in results if 'BUY' in r['longer_signal']]
longer_sells = [r for r in results if 'SELL' in r['longer_signal']]

# Sort by score
intraday_buys.sort(key=lambda x: x['intraday_score'], reverse=True)
intraday_sells.sort(key=lambda x: x['intraday_score'], reverse=True)
longer_buys.sort(key=lambda x: x['longer_score'], reverse=True)
longer_sells.sort(key=lambda x: x['longer_score'], reverse=True)

# Display results
st.success(f"✅ Scan completed at {datetime.now().strftime('%H:%M:%S')}")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Intraday BUYs", len(intraday_buys))
col2.metric("Intraday SELLs", len(intraday_sells))
col3.metric("Longer-term BUYs", len(longer_buys))
col4.metric("Longer-term SELLs", len(longer_sells))

st.divider()

# Tabs for different views
tab1, tab2 = st.tabs(["📍 Intraday Signals", "📈 Longer-term Signals"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🟢 BUY Signals")
        if intraday_buys:
            df_buy = pd.DataFrame(intraday_buys)
            df_buy['Rank'] = range(1, len(df_buy) + 1)
            df_buy = df_buy[['Rank', 'symbol', 'intraday_score', 'intraday_signal', 'price', 'intraday_details']]
            df_buy.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df_buy['Score'] = df_buy['Score'].round(1)
            df_buy['Price (₹)'] = df_buy['Price (₹)'].round(2)
            st.dataframe(df_buy, use_container_width=True, hide_index=True)
        else:
            st.info("No intraday BUY signals detected")
    
    with col_right:
        st.subheader("🔴 SELL Signals")
        if intraday_sells:
            df_sell = pd.DataFrame(intraday_sells)
            df_sell['Rank'] = range(1, len(df_sell) + 1)
            df_sell = df_sell[['Rank', 'symbol', 'intraday_score', 'intraday_signal', 'price', 'intraday_details']]
            df_sell.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df_sell['Score'] = df_sell['Score'].round(1)
            df_sell['Price (₹)'] = df_sell['Price (₹)'].round(2)
            st.dataframe(df_sell, use_container_width=True, hide_index=True)
        else:
            st.info("No intraday SELL signals detected")

with tab2:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🟢 BUY Signals")
        if longer_buys:
            df_buy = pd.DataFrame(longer_buys)
            df_buy['Rank'] = range(1, len(df_buy) + 1)
            df_buy = df_buy[['Rank', 'symbol', 'longer_score', 'longer_signal', 'price', 'longer_details']]
            df_buy.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df_buy['Score'] = df_buy['Score'].round(1)
            df_buy['Price (₹)'] = df_buy['Price (₹)'].round(2)
            st.dataframe(df_buy, use_container_width=True, hide_index=True)
        else:
            st.info("No longer-term BUY signals detected")
    
    with col_right:
        st.subheader("🔴 SELL Signals")
        if longer_sells:
            df_sell = pd.DataFrame(longer_sells)
            df_sell['Rank'] = range(1, len(df_sell) + 1)
            df_sell = df_sell[['Rank', 'symbol', 'longer_score', 'longer_signal', 'price', 'longer_details']]
            df_sell.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df_sell['Score'] = df_sell['Score'].round(1)
            df_sell['Price (₹)'] = df_sell['Price (₹)'].round(2)
            st.dataframe(df_sell, use_container_width=True, hide_index=True)
        else:
            st.info("No longer-term SELL signals detected")

st.divider()
st.caption("💡 Tip: Scores 80+ indicate very strong setups with agreement across multiple timeframes")
