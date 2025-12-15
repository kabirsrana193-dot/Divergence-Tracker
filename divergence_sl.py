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
    page_title="Nifty 50 Technical Scanner",
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

# ============================================================================
# RSI DIVERGENCE FUNCTIONS
# ============================================================================
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

def analyze_stock_rsi(symbol):
    """Analyze stock across 2 timeframes: 15m (5d) and 1h (1mo)"""
    
    # 15-minute for 5 days
    tf_15m = analyze_single_timeframe(symbol, '5d', '15m')
    
    # 1-hour for 1 month
    tf_1h = analyze_single_timeframe(symbol, '1mo', '1h')
    
    # Check if we got data
    if tf_15m is None and tf_1h is None:
        return None
    
    # Get current price
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
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
                score += 20
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
                score += 20
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
    
    return {}

# ============================================================================
# WILLIAMS %R FUNCTIONS
# ============================================================================
def calculate_williams_r(data, period=14):
    """Calculate Williams %R indicator"""
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
    return williams_r

def analyze_williams_r_single(symbol, period, interval):
    """Analyze Williams %R for a single timeframe"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        if len(data) < 20:
            return None
        
        # Calculate Williams %R
        data['WilliamsR'] = calculate_williams_r(data, period=14)
        data = data.dropna()
        
        if len(data) == 0:
            return None
        
        current_wr = data['WilliamsR'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Determine zone
        zone = None
        zone_type = None
        
        # Extreme reversal zones
        if -5 <= current_wr <= 0:
            zone = "EXTREME OVERBOUGHT"
            zone_type = "extreme"
        elif -100 <= current_wr <= -95:
            zone = "EXTREME OVERSOLD"
            zone_type = "extreme"
        # Normal zones
        elif -20 <= current_wr < -5:
            zone = "OVERBOUGHT"
            zone_type = "normal"
        elif -95 < current_wr <= -80:
            zone = "OVERSOLD"
            zone_type = "normal"
        
        return {
            'williams_r': current_wr,
            'price': current_price,
            'zone': zone,
            'zone_type': zone_type
        }
        
    except Exception as e:
        return None

def analyze_williams_r(symbol):
    """Analyze Williams %R on both 1h and 15m charts"""
    # 1-hour analysis
    wr_1h = analyze_williams_r_single(symbol, '5d', '1h')
    
    # 15-minute analysis
    wr_15m = analyze_williams_r_single(symbol, '5d', '15m')
    
    if not wr_1h:
        return None
    
    result = {
        'symbol': symbol.replace('.NS', ''),
        'price': wr_1h['price'],
        'williams_r_1h': wr_1h['williams_r'],
        'zone_1h': wr_1h['zone'],
        'zone_type_1h': wr_1h['zone_type'],
        'williams_r_15m': wr_15m['williams_r'] if wr_15m else None,
        'zone_15m': wr_15m['zone'] if wr_15m else None,
        'zone_type_15m': wr_15m['zone_type'] if wr_15m else None,
        'double_confirmation': False
    }
    
    # Check for double confirmation (both timeframes in extreme zones)
    if wr_15m and wr_1h['zone_type'] == 'extreme' and wr_15m['zone_type'] == 'extreme':
        if wr_1h['zone'] == wr_15m['zone']:
            result['double_confirmation'] = True
    
    # Only return if 1h has a zone (15m alone doesn't matter)
    if wr_1h['zone']:
        return result
    
    return None

# ============================================================================
# MAIN APP
# ============================================================================
st.title("📈 Nifty 50 Technical Scanner")
st.markdown("**RSI Divergence + Williams %R Analysis**")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("🔍 **RSI:** 15-min (5d) + 1-hour (1mo) | **Williams %R:** 1H (5d) + 15M (5d)")
with col2:
    if st.button("🔄 Scan Now", type="primary", use_container_width=True):
        st.session_state.scanned = False
        st.cache_data.clear()
        st.rerun()

st.divider()

# Initialize session state
if 'scanned' not in st.session_state:
    st.session_state.scanned = False

# Run scan
if not st.session_state.scanned:
    with st.spinner("Scanning Nifty 50 stocks..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        rsi_results = []
        wr_results = []
        no_signal_count = 0
        no_data_count = 0
        total = len(NIFTY_50_SYMBOLS)
        
        for i, symbol in enumerate(NIFTY_50_SYMBOLS):
            status_text.text(f"Analyzing {symbol.replace('.NS', '')}... ({i+1}/{total})")
            
            # RSI Divergence
            rsi_result = analyze_stock_rsi(symbol)
            if rsi_result is None:
                no_data_count += 1
            elif rsi_result == {}:
                no_signal_count += 1
            else:
                rsi_results.append(rsi_result)
            
            # Williams %R
            wr_result = analyze_williams_r(symbol)
            if wr_result:
                wr_results.append(wr_result)
            
            progress_bar.progress((i + 1) / total)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.rsi_results = rsi_results
        st.session_state.wr_results = wr_results
        st.session_state.no_signal_count = no_signal_count
        st.session_state.no_data_count = no_data_count
        st.session_state.scan_time = datetime.now()
        st.session_state.scanned = True

# Display scan completion
if st.session_state.scanned:
    st.success(f"✅ Scan completed at {st.session_state.scan_time.strftime('%H:%M:%S')}")

# Tabs
tab1, tab2 = st.tabs(["📊 RSI Divergence", "📉 Williams %R"])

# ============================================================================
# TAB 1: RSI DIVERGENCE
# ============================================================================
with tab1:
    rsi_results = st.session_state.get('rsi_results', [])
    
    # Separate BUY and SELL signals
    buy_signals = [r for r in rsi_results if 'BUY' in r['signal']]
    sell_signals = [r for r in rsi_results if 'SELL' in r['signal']]
    
    # Sort by score
    buy_signals.sort(key=lambda x: x['score'], reverse=True)
    sell_signals.sort(key=lambda x: x['score'], reverse=True)
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🟢 BUY", len(buy_signals))
    col2.metric("🔴 SELL", len(sell_signals))
    col3.metric("⚪ No Signal", st.session_state.get('no_signal_count', 0))
    col4.metric("❌ No Data", st.session_state.get('no_data_count', 0))
    col5.metric("📊 Total", len(NIFTY_50_SYMBOLS))
    
    st.markdown("---")
    
    # Display tables
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🟢 BUY Signals")
        if buy_signals:
            df = pd.DataFrame(buy_signals)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'score', 'signal', 'price', 'details']]
            df.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df['Score'] = df['Score'].round(1)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No BUY signals detected")
    
    with col_right:
        st.subheader("🔴 SELL Signals")
        if sell_signals:
            df = pd.DataFrame(sell_signals)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'score', 'signal', 'price', 'details']]
            df.columns = ['Rank', 'Stock', 'Score', 'Signal', 'Price (₹)', 'Timeframes']
            df['Score'] = df['Score'].round(1)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No SELL signals detected")
    
    st.caption("💡 **Score:** 80+ = Very Strong | 60-79 = Strong | 40-59 = Moderate")
    st.caption("📌 **Signals:** STRONG = Both timeframes agree | Normal = One timeframe")

# ============================================================================
# TAB 2: WILLIAMS %R
# ============================================================================
with tab2:
    wr_results = st.session_state.get('wr_results', [])
    
    # Separate by zone type (based on 1h)
    extreme_overbought_1h = [r for r in wr_results if r.get('zone_1h') == "EXTREME OVERBOUGHT"]
    extreme_oversold_1h = [r for r in wr_results if r.get('zone_1h') == "EXTREME OVERSOLD"]
    normal_overbought_1h = [r for r in wr_results if r.get('zone_1h') == "OVERBOUGHT"]
    normal_oversold_1h = [r for r in wr_results if r.get('zone_1h') == "OVERSOLD"]
    
    # Separate by 15m zones
    extreme_overbought_15m = [r for r in wr_results if r.get('zone_15m') == "EXTREME OVERBOUGHT"]
    extreme_oversold_15m = [r for r in wr_results if r.get('zone_15m') == "EXTREME OVERSOLD"]
    normal_overbought_15m = [r for r in wr_results if r.get('zone_15m') == "OVERBOUGHT"]
    normal_oversold_15m = [r for r in wr_results if r.get('zone_15m') == "OVERSOLD"]
    
    # Double confirmation
    double_confirm = [r for r in wr_results if r.get('double_confirmation', False)]
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔴 1H Extreme OB", len(extreme_overbought_1h))
    col2.metric("🟢 1H Extreme OS", len(extreme_oversold_1h))
    col3.metric("🟠 1H Overbought", len(normal_overbought_1h))
    col4.metric("🟢 1H Oversold", len(normal_oversold_1h))
    col5.metric("⚡ Double Confirm", len(double_confirm))
    
    st.markdown("---")
    
    # DOUBLE CONFIRMATION SECTION
    if len(double_confirm) > 0:
        st.subheader("🎯 DOUBLE CONFIRMATION - Both 1H & 15M Extreme!")
        st.markdown("**⚡ Highest probability - Both timeframes in extreme zones**")
        
        double_ob = [r for r in double_confirm if r.get('zone_1h') == "EXTREME OVERBOUGHT"]
        double_os = [r for r in double_confirm if r.get('zone_1h') == "EXTREME OVERSOLD"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 DOUBLE EXTREME OVERBOUGHT**")
            if double_ob:
                df = pd.DataFrame(double_ob)
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'symbol', 'williams_r_1h', 'williams_r_15m', 'price']]
                df.columns = ['Rank', 'Stock', 'W%R 1H', 'W%R 15M', 'Price (₹)']
                df['W%R 1H'] = df['W%R 1H'].round(2)
                df['W%R 15M'] = df['W%R 15M'].round(2)
                df['Price (₹)'] = df['Price (₹)'].round(2)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No double confirmation overbought")
        
        with col2:
            st.markdown("**🟢 DOUBLE EXTREME OVERSOLD**")
            if double_os:
                df = pd.DataFrame(double_os)
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'symbol', 'williams_r_1h', 'williams_r_15m', 'price']]
                df.columns = ['Rank', 'Stock', 'W%R 1H', 'W%R 15M', 'Price (₹)']
                df['W%R 1H'] = df['W%R 1H'].round(2)
                df['W%R 15M'] = df['W%R 15M'].round(2)
                df['Price (₹)'] = df['Price (₹)'].round(2)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No double confirmation oversold")
        
        st.markdown("---")
    
    # 1H EXTREME ZONES
    st.subheader("⚡ 1H Extreme Reversal Zones")
    st.markdown("**Primary signals from 1-hour chart**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 EXTREME OVERBOUGHT (-5 to 0)**")
        st.markdown("*Strong resistance - Likely bounce DOWN*")
        if extreme_overbought_1h:
            df = pd.DataFrame(extreme_overbought_1h)
            df['Rank'] = range(1, len(df) + 1)
            df['⚡'] = df['double_confirmation'].apply(lambda x: '⚡' if x else '')
            df = df[['Rank', '⚡', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', '⚡', 'Stock', 'W%R 1H', 'Price (₹)']
            df['W%R 1H'] = df['W%R 1H'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in 1H extreme overbought zone")
    
    with col2:
        st.markdown("**🟢 EXTREME OVERSOLD (-95 to -100)**")
        st.markdown("*Strong support - Likely bounce UP*")
        if extreme_oversold_1h:
            df = pd.DataFrame(extreme_oversold_1h)
            df['Rank'] = range(1, len(df) + 1)
            df['⚡'] = df['double_confirmation'].apply(lambda x: '⚡' if x else '')
            df = df[['Rank', '⚡', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', '⚡', 'Stock', 'W%R 1H', 'Price (₹)']
            df['W%R 1H'] = df['W%R 1H'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in 1H extreme oversold zone")
    
    st.markdown("---")
    
    # 1H NORMAL ZONES
    st.subheader("📊 1H Normal Overbought/Oversold Zones")
    st.markdown("**Potential reversal areas from 1-hour chart**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟠 OVERBOUGHT (-20 to -5)**")
        st.markdown("*Potential reversal down*")
        if normal_overbought_1h:
            df = pd.DataFrame(normal_overbought_1h)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'W%R 1H', 'Price (₹)']
            df['W%R 1H'] = df['W%R 1H'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in 1H overbought zone")
    
    with col2:
        st.markdown("**🟢 OVERSOLD (-95 to -80)**")
        st.markdown("*Potential reversal up*")
        if normal_oversold_1h:
            df = pd.DataFrame(normal_oversold_1h)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'W%R 1H', 'Price (₹)']
            df['W%R 1H'] = df['W%R 1H'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in 1H oversold zone")
    
    st.markdown("---")
    
    # 15M EXTREME ZONES
  col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔴 15M EXTREME OVERBOUGHT (-5 to 0)**")
    if extreme_overbought_15m:
        df = pd.DataFrame(extreme_overbought_15m)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'symbol', 'williams_r_15m', 'price']]
        df.columns = ['Rank', 'Stock', 'W%R 15M', 'Price (₹)']
        df['W%R 15M'] = df['W%R 15M'].round(2)
        df['Price (₹)'] = df['Price (₹)'].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks in 15M extreme overbought zone")

with col2:
    st.markdown("**🟢 15M EXTREME OVERSOLD (-95 to -100)**")
    if extreme_oversold_15m:
        df = pd.DataFrame(extreme_oversold_15m)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'symbol', 'williams_r_15m', 'price']]
        df.columns = ['Rank', 'Stock', 'W%R 15M', 'Price (₹)']
        df['W%R 15M'] = df['W%R 15M'].round(2)
        df['Price (₹)'] = df['Price (₹)'].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks in 15M extreme oversold zone")

st.markdown("---")

# 15M NORMAL ZONES
st.subheader("📍 15M Normal Zones (Reference Only)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🟠 15M OVERBOUGHT (-20 to -5)**")
    if normal_overbought_15m:
        df = pd.DataFrame(normal_overbought_15m)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'symbol', 'williams_r_15m', 'price']]
        df.columns = ['Rank', 'Stock', 'W%R 15M', 'Price (₹)']
        df['W%R 15M'] = df['W%R 15M'].round(2)
        df['Price (₹)'] = df['Price (₹)'].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks in 15M overbought zone")

with col2:
    st.markdown("**🟢 15M OVERSOLD (-95 to -80)**")
    if normal_oversold_15m:
        df = pd.DataFrame(normal_oversold_15m)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'symbol', 'williams_r_15m', 'price']]
        df.columns = ['Rank', 'Stock', 'W%R 15M', 'Price (₹)']
        df['W%R 15M'] = df['W%R 15M'].round(2)
        df['Price (₹)'] = df['Price (₹)'].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks in 15M oversold zone")

st.caption("💡 **Williams %R:** Period=14 | 1H (primary) + 15M (confirmation)")
st.caption("⚡ **Double Confirmation:** Both 1H & 15M in extreme zones - highest probability")
st.caption("📍 **15M zones:** Reference only - look for ⚡ symbol in 1H extreme zones for best signals")
st.divider()
