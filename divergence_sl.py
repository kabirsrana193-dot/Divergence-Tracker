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
    page_title="Nifty 200 Technical Scanner",
    page_icon="📈",
    layout="wide"
)

# Nifty 200 symbols (partial list - you'll need to add all 200)
NIFTY_200_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
    'TECHM.NS', 'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'INDUSINDBK.NS', 'TATASTEEL.NS', 'DRREDDY.NS',
    'JSWSTEEL.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HINDALCO.NS',
    'HEROMOTOCO.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'BPCL.NS', 'SBILIFE.NS',
    'TATACONSUM.NS', 'BAJAJ-AUTO.NS', 'LTIM.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS',
    # Add more Nifty 200 stocks here...
]

# ============================================================================
# RSI DIVERGENCE FUNCTIONS
# ============================================================================
def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_peaks_troughs(data, order=3):
    """Find peaks and troughs"""
    if len(data) < order * 2 + 1:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    price_peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    price_troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    rsi_peaks = argrelextrema(data['RSI'].values, np.greater, order=order)[0]
    rsi_troughs = argrelextrema(data['RSI'].values, np.less, order=order)[0]
    return price_peaks, price_troughs, rsi_peaks, rsi_troughs

def calculate_divergence_strength(price_1, price_2, rsi_1, rsi_2, div_type):
    """Calculate strength score"""
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
        recent_price_troughs = [i for i in price_troughs if i >= max(0, len(data) - lookback)]
        recent_rsi_troughs = [i for i in rsi_troughs if i >= max(0, len(data) - lookback)]
        
        if len(recent_price_troughs) >= 2 and len(recent_rsi_troughs) >= 2:
            pt1, pt2 = recent_price_troughs[-2], recent_price_troughs[-1]
            rt1, rt2 = recent_rsi_troughs[-2], recent_rsi_troughs[-1]
            
            if (data['Close'].iloc[pt2] < data['Close'].iloc[pt1] and 
                data['RSI'].iloc[rt2] > data['RSI'].iloc[rt1]):
                
                strength = calculate_divergence_strength(
                    data['Close'].iloc[pt1], data['Close'].iloc[pt2],
                    data['RSI'].iloc[rt1], data['RSI'].iloc[rt2], 'Bullish'
                )
                divergences.append({'strength': strength})
    
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
            
            if (data['Close'].iloc[pp2] > data['Close'].iloc[pp1] and 
                data['RSI'].iloc[rp2] < data['RSI'].iloc[rp1]):
                
                strength = calculate_divergence_strength(
                    data['Close'].iloc[pp1], data['Close'].iloc[pp2],
                    data['RSI'].iloc[rp1], data['RSI'].iloc[rp2], 'Bearish'
                )
                divergences.append({'strength': strength})
    
    return divergences

def analyze_rsi_divergence(symbol):
    """Analyze RSI divergence on 1h and 4h"""
    try:
        # 1 hour data
        data_1h = yf.Ticker(symbol).history(period='5d', interval='1h')
        # 4 hour data  
        data_4h = yf.Ticker(symbol).history(period='60d', interval='1h')
        # Resample to 4h
        data_4h = data_4h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(data_1h) < 30 or len(data_4h) < 30:
            return None
        
        # Calculate RSI
        data_1h['RSI'] = calculate_rsi(data_1h)
        data_4h['RSI'] = calculate_rsi(data_4h)
        data_1h = data_1h.dropna()
        data_4h = data_4h.dropna()
        
        # Detect divergences
        bull_1h = detect_bullish_divergence(data_1h)
        bear_1h = detect_bearish_divergence(data_1h)
        bull_4h = detect_bullish_divergence(data_4h)
        bear_4h = detect_bearish_divergence(data_4h)
        
        # Calculate scores
        score_1h = 0
        signal_1h = "NEUTRAL"
        if bull_1h:
            score_1h = max([d['strength'] for d in bull_1h])
            signal_1h = "BUY"
        elif bear_1h:
            score_1h = max([d['strength'] for d in bear_1h])
            signal_1h = "SELL"
        
        score_4h = 0
        signal_4h = "NEUTRAL"
        if bull_4h:
            score_4h = max([d['strength'] for d in bull_4h])
            signal_4h = "BUY"
        elif bear_4h:
            score_4h = max([d['strength'] for d in bear_4h])
            signal_4h = "SELL"
        
        # Combined signal
        combined_score = (score_1h + score_4h) / 2
        combined_signal = "NEUTRAL"
        
        if signal_1h == signal_4h and signal_1h != "NEUTRAL":
            combined_signal = f"STRONG {signal_1h}"
            combined_score += 20
        elif signal_1h != "NEUTRAL":
            combined_signal = signal_1h
        elif signal_4h != "NEUTRAL":
            combined_signal = signal_4h
        
        if combined_signal != "NEUTRAL":
            return {
                'symbol': symbol.replace('.NS', ''),
                'price': data_1h['Close'].iloc[-1],
                'signal': combined_signal,
                'score': min(combined_score, 100),
                '1h': signal_1h,
                '4h': signal_4h
            }
        
        return None
    except:
        return None

# ============================================================================
# WILLIAMS %R FUNCTIONS
# ============================================================================
def calculate_williams_r(data, period=14):
    """Calculate Williams %R"""
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
    return williams_r

def analyze_williams_r(symbol):
    """Analyze Williams %R on 1h chart"""
    try:
        # Fetch 1 hour data
        data = yf.Ticker(symbol).history(period='5d', interval='1h')
        
        if len(data) < 20:
            return None
        
        # Calculate Williams %R
        data['WilliamsR'] = calculate_williams_r(data, period=14)
        data = data.dropna()
        
        if len(data) == 0:
            return None
        
        current_wr = data['WilliamsR'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Check zones
        zone = None
        signal_type = None
        
        # Extreme reversal zones
        if -5 <= current_wr <= 0:
            zone = "EXTREME OVERBOUGHT"
            signal_type = "extreme"
        elif -100 <= current_wr <= -95:
            zone = "EXTREME OVERSOLD"
            signal_type = "extreme"
        # Normal zones
        elif -20 <= current_wr <= 0:
            zone = "OVERBOUGHT"
            signal_type = "normal"
        elif -100 <= current_wr <= -80:
            zone = "OVERSOLD"
            signal_type = "normal"
        
        if zone:
            return {
                'symbol': symbol.replace('.NS', ''),
                'price': current_price,
                'williams_r': current_wr,
                'zone': zone,
                'signal_type': signal_type
            }
        
        return None
    except:
        return None

# ============================================================================
# MAIN APP
# ============================================================================
st.title("📈 Nifty 200 Technical Scanner")
st.markdown("**RSI Divergence + Williams %R Analysis**")

if st.button("🔄 Scan All Stocks", type="primary"):
    st.session_state.clear()

st.divider()

# Scan stocks
if 'rsi_results' not in st.session_state:
    with st.spinner("Scanning Nifty 200 stocks..."):
        progress = st.progress(0)
        status = st.empty()
        
        rsi_results = []
        wr_results = []
        total = len(NIFTY_200_SYMBOLS)
        
        for i, symbol in enumerate(NIFTY_200_SYMBOLS):
            status.text(f"Analyzing {symbol.replace('.NS', '')} ({i+1}/{total})")
            
            # RSI Divergence
            rsi_result = analyze_rsi_divergence(symbol)
            if rsi_result:
                rsi_results.append(rsi_result)
            
            # Williams %R
            wr_result = analyze_williams_r(symbol)
            if wr_result:
                wr_results.append(wr_result)
            
            progress.progress((i + 1) / total)
        
        progress.empty()
        status.empty()
        
        st.session_state.rsi_results = rsi_results
        st.session_state.wr_results = wr_results
        st.session_state.scan_time = datetime.now()

# Display results
if 'scan_time' in st.session_state:
    st.success(f"✅ Scan completed at {st.session_state.scan_time.strftime('%H:%M:%S')}")

# Tabs
tab1, tab2 = st.tabs(["📊 RSI Divergence", "📉 Williams %R"])

# ============================================================================
# TAB 1: RSI DIVERGENCE
# ============================================================================
with tab1:
    rsi_results = st.session_state.get('rsi_results', [])
    
    buys = [r for r in rsi_results if 'BUY' in r['signal']]
    sells = [r for r in rsi_results if 'SELL' in r['signal']]
    
    buys.sort(key=lambda x: x['score'], reverse=True)
    sells.sort(key=lambda x: x['score'], reverse=True)
    
    col1, col2 = st.columns(2)
    col1.metric("BUY Signals", len(buys))
    col2.metric("SELL Signals", len(sells))
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🟢 BUY Signals")
        if buys:
            df = pd.DataFrame(buys)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'score', 'signal', '1h', '4h', 'price']]
            df.columns = ['Rank', 'Stock', 'Score', 'Signal', '1H', '4H', 'Price (₹)']
            df['Score'] = df['Score'].round(1)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No BUY signals detected")
    
    with col_right:
        st.subheader("🔴 SELL Signals")
        if sells:
            df = pd.DataFrame(sells)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'score', 'signal', '1h', '4h', 'price']]
            df.columns = ['Rank', 'Stock', 'Score', 'Signal', '1H', '4H', 'Price (₹)']
            df['Score'] = df['Score'].round(1)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No SELL signals detected")

# ============================================================================
# TAB 2: WILLIAMS %R
# ============================================================================
with tab2:
    wr_results = st.session_state.get('wr_results', [])
    
    # Separate by signal type
    extreme = [r for r in wr_results if r['signal_type'] == 'extreme']
    normal = [r for r in wr_results if r['signal_type'] == 'normal']
    
    extreme_overbought = [r for r in extreme if 'OVERBOUGHT' in r['zone']]
    extreme_oversold = [r for r in extreme if 'OVERSOLD' in r['zone']]
    normal_overbought = [r for r in normal if 'OVERBOUGHT' in r['zone']]
    normal_oversold = [r for r in normal if 'OVERSOLD' in r['zone']]
    
    st.subheader("⚡ Extreme Reversal Zones (-5 to 0 | -95 to -100)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 EXTREME OVERBOUGHT (-5 to 0)** - Strong resistance, likely bounce DOWN")
        if extreme_overbought:
            df = pd.DataFrame(extreme_overbought)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r', 'price']]
            df.columns = ['Rank', 'Stock', 'Williams %R', 'Price (₹)']
            df['Williams %R'] = df['Williams %R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in extreme overbought zone")
    
    with col2:
        st.markdown("**🟢 EXTREME OVERSOLD (-95 to -100)** - Strong support, likely bounce UP")
        if extreme_oversold:
            df = pd.DataFrame(extreme_oversold)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r', 'price']]
            df.columns = ['Rank', 'Stock', 'Williams %R', 'Price (₹)']
            df['Williams %R'] = df['Williams %R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in extreme oversold zone")
    
    st.markdown("---")
    st.subheader("📊 Normal Overbought/Oversold Zones (-20 to 0 | -80 to -100)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟠 OVERBOUGHT (-20 to 0)** - Potential reversal down")
        if normal_overbought:
            df = pd.DataFrame(normal_overbought)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r', 'price']]
            df.columns = ['Rank', 'Stock', 'Williams %R', 'Price (₹)']
            df['Williams %R'] = df['Williams %R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in overbought zone")
    
    with col2:
        st.markdown("**🟢 OVERSOLD (-80 to -100)** - Potential reversal up")
        if normal_oversold:
            df = pd.DataFrame(normal_oversold)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r', 'price']]
            df.columns = ['Rank', 'Stock', 'Williams %R', 'Price (₹)']
            df['Williams %R'] = df['Williams %R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in oversold zone")

st.divider()
st.caption("💡 RSI: 1H + 4H charts | Williams %R: 1H chart only | Period: 14")
