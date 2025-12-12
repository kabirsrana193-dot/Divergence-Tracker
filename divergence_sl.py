import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
import time
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_stock_data(symbol, start_date, end_date):
    """Download stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data is None or len(data) == 0:
            return None
            
        return data
    except Exception as e:
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_peaks_troughs(data, order=5):
    """Find local peaks and troughs"""
    if len(data) < order * 2 + 1:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
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

def detect_bullish_divergence(data, lookback=30):
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

def detect_bearish_divergence(data, lookback=30):
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

def analyze_stock_combined(symbol):
    """Analyze stock - simplified version using date ranges"""
    try:
        today = datetime.now()
        
        # Download all data at once - 6 months for all timeframes
        end_date = today
        start_date = today - timedelta(days=180)
        
        data = download_stock_data(symbol, start_date, end_date)
        
        if data is None or len(data) < 50:
            return None
        
        # Calculate RSI
        data['RSI'] = calculate_rsi(data)
        data = data.dropna()
        
        if len(data) < 20:
            return None
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Analyze SHORT-TERM (last 10 days)
        short_data = data.tail(10) if len(data) >= 10 else data
        short_bullish = detect_bullish_divergence(short_data, lookback=10)
        short_bearish = detect_bearish_divergence(short_data, lookback=10)
        
        # Analyze MEDIUM-TERM (last 30 days)
        medium_data = data.tail(30) if len(data) >= 30 else data
        medium_bullish = detect_bullish_divergence(medium_data, lookback=30)
        medium_bearish = detect_bearish_divergence(medium_data, lookback=30)
        
        # Analyze LONG-TERM (last 90 days)
        long_data = data.tail(90) if len(data) >= 90 else data
        long_bullish = detect_bullish_divergence(long_data, lookback=50)
        long_bearish = detect_bearish_divergence(long_data, lookback=50)
        
        # Calculate SHORT-TERM score
        short_score = 0
        short_signal = "NEUTRAL"
        short_details = []
        
        if short_bullish or medium_bullish:
            short_score = (
                (max([d['strength'] for d in short_bullish]) if short_bullish else 0) +
                (max([d['strength'] for d in medium_bullish]) if medium_bullish else 0)
            ) / 2
            
            if short_bullish and medium_bullish:
                short_score += 20
                short_signal = "STRONG BUY"
            else:
                short_signal = "BUY"
                
            if short_bullish:
                short_details.append(f"10d: {max([d['strength'] for d in short_bullish]):.0f}")
            if medium_bullish:
                short_details.append(f"30d: {max([d['strength'] for d in medium_bullish]):.0f}")
        
        elif short_bearish or medium_bearish:
            short_score = (
                (max([d['strength'] for d in short_bearish]) if short_bearish else 0) +
                (max([d['strength'] for d in medium_bearish]) if medium_bearish else 0)
            ) / 2
            
            if short_bearish and medium_bearish:
                short_score += 20
                short_signal = "STRONG SELL"
            else:
                short_signal = "SELL"
                
            if short_bearish:
                short_details.append(f"10d: {max([d['strength'] for d in short_bearish]):.0f}")
            if medium_bearish:
                short_details.append(f"30d: {max([d['strength'] for d in medium_bearish]):.0f}")
        
        # Calculate LONG-TERM score
        long_score = 0
        long_signal = "NEUTRAL"
        long_details = []
        
        if long_bullish:
            long_score = max([d['strength'] for d in long_bullish])
            long_signal = "STRONG BUY" if long_score > 70 else "BUY"
            long_details.append(f"90d: {long_score:.0f}")
        
        elif long_bearish:
            long_score = max([d['strength'] for d in long_bearish])
            long_signal = "STRONG SELL" if long_score > 70 else "SELL"
            long_details.append(f"90d: {long_score:.0f}")
        
        if short_signal != "NEUTRAL" or long_signal != "NEUTRAL":
            return {
                'symbol': symbol.replace('.NS', ''),
                'price': current_price,
                'intraday_score': min(short_score, 100),
                'intraday_signal': short_signal,
                'intraday_details': ', '.join(short_details) if short_details else '-',
                'longer_score': min(long_score, 100),
                'longer_signal': long_signal,
                'longer_details': ', '.join(long_details) if long_details else '-'
            }
        
        return None
        
    except Exception as e:
        if 'errors' in st.session_state:
            st.session_state.errors.append(f"{symbol}: {str(e)}")
        return None

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'scan_stats' not in st.session_state:
    st.session_state.scan_stats = {}

# Main App
st.title("📈 Nifty 50 RSI Divergence Scanner")
st.markdown("**Real-time divergence analysis across multiple timeframes**")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown("**Short-term:** 10-day + 30-day")
with col2:
    st.markdown("**Long-term:** 90-day analysis")
with col3:
    if st.button("🔄 Scan Now", type="primary", use_container_width=True):
        st.session_state.scan_results = None
        st.session_state.last_scan_time = None
        st.session_state.errors = []
        st.session_state.scan_stats = {}
        st.cache_data.clear()
        st.rerun()

st.divider()

# Run scan if no results
if st.session_state.scan_results is None:
    st.info("🔄 Scanning stocks... This may take 1-2 minutes.")
    
    with st.spinner("Analyzing Nifty 50 stocks..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total = len(NIFTY_50_SYMBOLS)
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(NIFTY_50_SYMBOLS):
            status_text.text(f"📊 Analyzing {symbol.replace('.NS', '')}... ({i+1}/{total})")
            result = analyze_stock_combined(symbol)
            if result:
                results.append(result)
                successful += 1
            else:
                failed += 1
            progress_bar.progress((i + 1) / total)
            time.sleep(0.1)  # Small delay
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.scan_results = results
        st.session_state.last_scan_time = datetime.now()
        st.session_state.scan_stats = {
            'total': total,
            'successful': successful,
            'failed': failed,
            'signals_found': len(results)
        }

# Use cached results
results = st.session_state.scan_results if st.session_state.scan_results else []

# Display scan statistics
if st.session_state.scan_stats:
    stats = st.session_state.scan_stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Stocks Scanned", stats['total'])
    col2.metric("✅ Successful", stats['successful'])
    col3.metric("❌ Failed", stats['failed'])
    col4.metric("🎯 Signals Found", stats['signals_found'])
    st.divider()

# Display errors if any
if st.session_state.errors and len(st.session_state.errors) > 0:
    with st.expander(f"⚠️ Debug Info - {len(st.session_state.errors)} Errors"):
        for err in st.session_state.errors[:20]:
            st.text(err)

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
if st.session_state.last_scan_time:
    st.success(f"✅ Scan completed at {st.session_state.last_scan_time.strftime('%H:%M:%S')} • Results cached for 1 hour")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Short-term BUYs", len(intraday_buys))
col2.metric("Short-term SELLs", len(intraday_sells))
col3.metric("Long-term BUYs", len(longer_buys))
col4.metric("Long-term SELLs", len(longer_sells))

st.divider()

# Tabs for different views
tab1, tab2 = st.tabs(["📍 Short-term Signals", "📈 Long-term Signals"])

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
            st.info("No short-term BUY signals detected")
    
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
            st.info("No short-term SELL signals detected")

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
            st.info("No long-term BUY signals detected")
    
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
            st.info("No long-term SELL signals detected")

st.divider()
st.caption("💡 Tip: Scores 80+ indicate very strong divergence setups")
st.caption("📊 Data is cached for 1 hour to improve performance and reduce API calls")
