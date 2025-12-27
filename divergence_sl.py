import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Nifty 200 Technical Scanner",
    page_icon="📈",
    layout="wide"
)

# Nifty 200 symbols
NIFTY_200_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
    'TECHM.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'INDUSINDBK.NS', 'TATASTEEL.NS', 'DRREDDY.NS',
    'JSWSTEEL.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HINDALCO.NS',
    'HEROMOTOCO.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'BPCL.NS', 'SBILIFE.NS',
    'TATACONSUM.NS', 'BAJAJ-AUTO.NS', 'LTIM.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS',
    'TATAMOTORS.NS', 'UPL.NS', 'SHREECEM.NS', 'HAVELLS.NS', 'PIDILITIND.NS',
    'IOC.NS', 'VEDL.NS', 'GAIL.NS', 'ZOMATO.NS', 'PAYTM.NS',
    'TRENT.NS', 'DMART.NS', 'MRF.NS', 'LUPIN.NS', 'TORNTPHARM.NS',
    'BIOCON.NS', 'AUROPHARMA.NS', 'DLF.NS', 'YESBANK.NS', 'BANKBARODA.NS',
    'PNB.NS', 'CANBK.NS', 'UNIONBANK.NS', 'INDIANB.NS', 'FEDERALBNK.NS',
    'IDFCFIRSTB.NS', 'AUBANK.NS', 'INDIGO.NS', 'ADANIGREEN.NS', 'ATGL.NS',
    'ADANIPOWER.NS', 'GODREJCP.NS', 'TVSMOTOR.NS', 'ACC.NS', 'AMBUJACEM.NS',
    'BERGEPAINT.NS', 'BANDHANBNK.NS', 'ASHOKLEY.NS', 'BOSCHLTD.NS', 'ABB.NS',
    'SIEMENS.NS', 'BEL.NS', 'HAL.NS', 'DIXON.NS', 'POLYCAB.NS',
    'VOLTAS.NS', 'CROMPTON.NS', 'MOTHERSON.NS', 'LICHSGFIN.NS', 'PFC.NS',
    'RECLTD.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'SAIL.NS', 'PIIND.NS',
    'SRF.NS', 'CONCOR.NS', 'IRCTC.NS', 'MAXHEALTH.NS', 'FORTIS.NS',
    'MANKIND.NS', 'ZYDUSLIFE.NS', 'IEX.NS', 'AWL.NS', 'MARICO.NS',
    'DABUR.NS', 'COLPAL.NS', 'VBL.NS', 'TATAELXSI.NS', 'COFORGE.NS',
    'PERSISTENT.NS', 'LTTS.NS', 'MPHASIS.NS', 'NAUKRI.NS', 'PAGEIND.NS',
    'ICICIGI.NS', 'ICICIPRULI.NS', 'PRESTIGE.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS',
    'BHARATFORG.NS', 'CUMMINSIND.NS', 'APOLLOTYRE.NS', 'ESCORTS.NS', 'MUTHOOTFIN.NS',
    'CHOLAFIN.NS', 'NATIONALUM.NS', 'HINDZINC.NS', 'TATACHEM.NS', 'BALRAMCHIN.NS',
    'DALBHARAT.NS', 'JKCEMENT.NS', 'NH.NS', 'LAURUSLABS.NS', 'GRANULES.NS',
    'NATCOPHARM.NS', 'GLENMARK.NS', 'PVRINOX.NS', 'DEEPAKNTR.NS', 'AARTIIND.NS',
    'INDIACEM.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'BALKRISIND.NS', 'CEAT.NS',
    'JKTYRE.NS', 'SONACOMS.NS', 'IRFC.NS', 'RVNL.NS', 'KPITTECH.NS',
    'TATAPOWER.NS', 'LICI.NS', 'ADANIENSOL.NS', 'JIOFIN.NS', 'JSWENERGY.NS',
    'SBICARD.NS', 'IDEA.NS', 'ABBOTINDIA.NS', 'ALKEM.NS', 'PETRONET.NS',
    'ASTRAL.NS', 'CGPOWER.NS', 'CHOLAHLDNG.NS', 'CRISIL.NS',
    'DELHIVERY.NS', 'LODHA.NS', 'GUJGASLTD.NS', 'HINDPETRO.NS', 'INDUSTOWER.NS',
    'JUBLFOOD.NS', 'KANSAINER.NS', 'KEI.NS', 'LTF.NS',
    'MFSL.NS', 'OFSS.NS', 'PHOENIX.NS', 'POLICYBZR.NS',
    'ZYDUSWELL.NS', 'PEL.NS', 'PGHH.NS', 'PVR.NS',
    'RAMCOCEM.NS', 'SOLARINDS.NS',
    'SUNTV.NS', 'SUPREMEIND.NS', 'TORNTPOWER.NS', 
    'UBL.NS', 'WHIRLPOOL.NS'
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
        
        data['RSI'] = calculate_rsi(data)
        data = data.dropna()
        
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
    """Analyze stock across 2 timeframes: 1h and 2h"""
    
    # 1-hour 
    tf_1h = analyze_single_timeframe(symbol, '5d', '1h')
    
    # 2-hour (fetch more data and resample)
    try:
        stock = yf.Ticker(symbol)
        data_2h = stock.history(period='10d', interval='1h')
        if len(data_2h) >= 30:
            # Resample to 2h
            data_2h = data_2h.resample('2H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            if len(data_2h) >= 30:
                data_2h['RSI'] = calculate_rsi(data_2h)
                data_2h = data_2h.dropna()
                
                bullish_div = detect_bullish_divergence(data_2h)
                bearish_div = detect_bearish_divergence(data_2h)
                
                tf_2h = {
                    'has_bullish': len(bullish_div) > 0,
                    'has_bearish': len(bearish_div) > 0,
                    'bullish_strength': max([d['strength'] for d in bullish_div]) if bullish_div else 0,
                    'bearish_strength': max([d['strength'] for d in bearish_div]) if bearish_div else 0,
                    'current_rsi': data_2h['RSI'].iloc[-1]
                }
            else:
                tf_2h = None
        else:
            tf_2h = None
    except:
        tf_2h = None
    
    if tf_1h is None and tf_2h is None:
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
    
    if tf_1h and tf_2h:
        if tf_1h['has_bullish'] or tf_2h['has_bullish']:
            score = (tf_1h['bullish_strength'] + tf_2h['bullish_strength']) / 2
            
            if tf_1h['has_bullish'] and tf_2h['has_bullish']:
                score += 20
                signal = "STRONG BUY"
            else:
                signal = "BUY"
            
            if tf_1h['has_bullish']:
                details.append(f"1h: {tf_1h['bullish_strength']:.0f}")
            if tf_2h['has_bullish']:
                details.append(f"2h: {tf_2h['bullish_strength']:.0f}")
        
        elif tf_1h['has_bearish'] or tf_2h['has_bearish']:
            score = (tf_1h['bearish_strength'] + tf_2h['bearish_strength']) / 2
            
            if tf_1h['has_bearish'] and tf_2h['has_bearish']:
                score += 20
                signal = "STRONG SELL"
            else:
                signal = "SELL"
            
            if tf_1h['has_bearish']:
                details.append(f"1h: {tf_1h['bearish_strength']:.0f}")
            if tf_2h['has_bearish']:
                details.append(f"2h: {tf_2h['bearish_strength']:.0f}")
    
    if signal != "NEUTRAL":
        return {
            'symbol': symbol.replace('.NS', ''),
            'price': current_price,
            'score': min(score, 100),
            'signal': signal,
            'details': ', '.join(details) if details else '-',
            'rsi_1h': tf_1h['current_rsi'] if tf_1h else None,
            'rsi_2h': tf_2h['current_rsi'] if tf_2h else None
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
        
        data['WilliamsR'] = calculate_williams_r(data, period=14)
        data = data.dropna()
        
        if len(data) == 0:
            return None
        
        current_wr = data['WilliamsR'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Determine zone
        zone = None
        zone_type = None
        
        if -5 <= current_wr <= 0:
            zone = "EXTREME OVERBOUGHT"
            zone_type = "extreme"
        elif -100 <= current_wr <= -95:
            zone = "EXTREME OVERSOLD"
            zone_type = "extreme"
        elif -20 <= current_wr < -5:
            zone = "OVERBOUGHT"
            zone_type = "normal"
        elif -95 < current_wr <= -80:
            zone = "OVERSOLD"
            zone_type = "normal"
        
        # Check how long in zone (count consecutive candles)
        hours_in_zone = 0
        if zone:
            for i in range(len(data) - 1, -1, -1):
                wr_val = data['WilliamsR'].iloc[i]
                if zone_type == "extreme":
                    if "OVERBOUGHT" in zone and -5 <= wr_val <= 0:
                        hours_in_zone += 1
                    elif "OVERSOLD" in zone and -100 <= wr_val <= -95:
                        hours_in_zone += 1
                    else:
                        break
                else:  # normal
                    if "OVERBOUGHT" in zone and -20 <= wr_val < -5:
                        hours_in_zone += 1
                    elif "OVERSOLD" in zone and -95 < wr_val <= -80:
                        hours_in_zone += 1
                    else:
                        break
        
        return {
            'williams_r': current_wr,
            'price': current_price,
            'zone': zone,
            'zone_type': zone_type,
            'hours_in_zone': hours_in_zone
        }
        
    except Exception as e:
        return None

def analyze_williams_r(symbol):
    """Analyze Williams %R on both 1h and 2h charts"""
    # 1-hour analysis
    wr_1h = analyze_williams_r_single(symbol, '5d', '1h')
    
    # 2-hour analysis (fetch more data and resample)
    try:
        stock = yf.Ticker(symbol)
        data_2h = stock.history(period='10d', interval='1h')
        if len(data_2h) >= 20:
            data_2h = data_2h.resample('2H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            if len(data_2h) >= 20:
                data_2h['WilliamsR'] = calculate_williams_r(data_2h, period=14)
                data_2h = data_2h.dropna()
                
                current_wr = data_2h['WilliamsR'].iloc[-1]
                current_price = data_2h['Close'].iloc[-1]
                
                zone = None
                zone_type = None
                
                if -5 <= current_wr <= 0:
                    zone = "EXTREME OVERBOUGHT"
                    zone_type = "extreme"
                elif -100 <= current_wr <= -95:
                    zone = "EXTREME OVERSOLD"
                    zone_type = "extreme"
                elif -20 <= current_wr < -5:
                    zone = "OVERBOUGHT"
                    zone_type = "normal"
                elif -95 < current_wr <= -80:
                    zone = "OVERSOLD"
                    zone_type = "normal"
                
                hours_in_zone_2h = 0
                if zone:
                    for i in range(len(data_2h) - 1, -1, -1):
                        wr_val = data_2h['WilliamsR'].iloc[i]
                        if zone_type == "extreme":
                            if "OVERBOUGHT" in zone and -5 <= wr_val <= 0:
                                hours_in_zone_2h += 2  # 2 hours per candle
                            elif "OVERSOLD" in zone and -100 <= wr_val <= -95:
                                hours_in_zone_2h += 2
                            else:
                                break
                        else:
                            if "OVERBOUGHT" in zone and -20 <= wr_val < -5:
                                hours_in_zone_2h += 2
                            elif "OVERSOLD" in zone and -95 < wr_val <= -80:
                                hours_in_zone_2h += 2
                            else:
                                break
                
                wr_2h = {
                    'williams_r': current_wr,
                    'price': current_price,
                    'zone': zone,
                    'zone_type': zone_type,
                    'hours_in_zone': hours_in_zone_2h
                }
            else:
                wr_2h = None
        else:
            wr_2h = None
    except:
        wr_2h = None
    
    if not wr_1h:
        return None
    
    if not wr_1h['zone']:
        return None
    
    result = {
        'symbol': symbol.replace('.NS', ''),
        'price': wr_1h['price'],
        'williams_r_1h': wr_1h['williams_r'],
        'zone_1h': wr_1h['zone'],
        'zone_type_1h': wr_1h['zone_type'],
        'hours_in_zone_1h': wr_1h['hours_in_zone'],
        'williams_r_2h': wr_2h['williams_r'] if wr_2h else None,
        'zone_2h': wr_2h['zone'] if wr_2h else None,
        'zone_type_2h': wr_2h['zone_type'] if wr_2h else None,
        'hours_in_zone_2h': wr_2h['hours_in_zone'] if wr_2h else 0,
        'double_confirmation': False,
        'extended_stay': False
    }
    
    # Double confirmation (both timeframes in extreme zones)
    if wr_2h and wr_1h['zone_type'] == 'extreme' and wr_2h['zone_type'] == 'extreme':
        if wr_1h['zone'] == wr_2h['zone']:
            result['double_confirmation'] = True
    
    # Extended stay (3+ hours in zone on 1h chart)
    if wr_1h['hours_in_zone'] >= 3:
        result['extended_stay'] = True
    
    return result

# ============================================================================
# HEATMAP FUNCTIONS
# ============================================================================
def get_stock_heatmap_data(symbol):
    """Get comprehensive data for heatmap visualization"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='5d', interval='1h')
        
        if len(data) < 20:
            return None
        
        # Calculate indicators
        data['RSI'] = calculate_rsi(data)
        data['WilliamsR'] = calculate_williams_r(data)
        data = data.dropna()
        
        if len(data) == 0:
            return None
        
        # Get last 30 candles for heatmap
        recent_data = data.tail(30)
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'timestamps': recent_data.index.tolist(),
            'rsi': recent_data['RSI'].tolist(),
            'williams_r': recent_data['WilliamsR'].tolist(),
            'close': recent_data['Close'].tolist()
        }
    except Exception as e:
        return None

def create_heatmap(selected_stocks):
    """Create heatmap visualization for selected stocks"""
    if not selected_stocks:
        return None
    
    # Fetch data for selected stocks
    heatmap_data = []
    for symbol in selected_stocks:
        data = get_stock_heatmap_data(symbol)
        if data:
            heatmap_data.append(data)
    
    if not heatmap_data:
        return None
    
    # Create subplots - one row per stock, 3 columns (Price, RSI, Williams %R)
    fig = go.Figure()
    
    # Determine common time axis
    max_len = max(len(d['timestamps']) for d in heatmap_data)
    time_indices = list(range(max_len))
    
    for idx, stock_data in enumerate(heatmap_data):
        symbol = stock_data['symbol']
        
        # Pad data if needed
        n = len(stock_data['rsi'])
        rsi_vals = stock_data['rsi']
        wr_vals = stock_data['williams_r']
        
        # Create heatmap matrix for this stock (1 row with multiple metrics)
        # We'll stack RSI and Williams %R as two separate rows
        y_labels = [f"{symbol} RSI", f"{symbol} W%R"]
        
        # Normalize values for heatmap coloring
        # RSI: 0-100 scale
        # Williams %R: -100 to 0, convert to 0-100 for visualization
        wr_normalized = [(w + 100) for w in wr_vals]
        
        # Add RSI heatmap
        fig.add_trace(go.Heatmap(
            z=[rsi_vals],
            x=time_indices[:n],
            y=[f"{symbol} RSI"],
            colorscale=[
                [0, '#d62728'],      # Red for oversold (0-30)
                [0.3, '#ff7f0e'],    # Orange
                [0.5, '#2ca02c'],    # Green for neutral (40-60)
                [0.7, '#ff7f0e'],    # Orange
                [1, '#d62728']       # Red for overbought (70-100)
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(
                title="RSI",
                x=1.02,
                len=0.3,
                y=0.85 - (idx * 0.15)
            ),
            showscale=(idx == 0),
            hovertemplate=f'{symbol} RSI: %{{z:.1f}}<extra></extra>'
        ))
        
        # Add Williams %R heatmap
        fig.add_trace(go.Heatmap(
            z=[wr_normalized],
            x=time_indices[:n],
            y=[f"{symbol} W%R"],
            colorscale=[
                [0, '#d62728'],      # Red for oversold (-100)
                [0.2, '#ff7f0e'],    # Orange
                [0.5, '#2ca02c'],    # Green for neutral
                [0.8, '#ff7f0e'],    # Orange
                [1, '#d62728']       # Red for overbought (0)
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(
                title="W%R",
                x=1.08,
                len=0.3,
                y=0.85 - (idx * 0.15)
            ),
            showscale=(idx == 0),
            customdata=[[w] for w in wr_vals],
            hovertemplate=f'{symbol} W%R: %{{customdata[0]:.1f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Technical Indicators Heatmap (Last 30 Hours)",
        xaxis_title="Time (Hours Ago)",
        height=150 * len(heatmap_data) * 2,
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117'
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
st.title("📈 Nifty 200 Technical Scanner")
st.markdown("**RSI Divergence + Williams %R Analysis**")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("🔍 **RSI:** 1H + 2H | **Williams %R:** 1H + 2H")
with col2:
    if st.button("🔄 Scan Now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# Initialize session state
if 'scanned' not in st.session_state:
    st.session_state.scanned = False

# Run scan
if not st.session_state.scanned:
    with st.spinner("Scanning Nifty 200 stocks..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        rsi_results = []
        wr_results = []
        no_signal_count = 0
        no_data_count = 0
        total = len(NIFTY_200_SYMBOLS)
        
        for i, symbol in enumerate(NIFTY_200_SYMBOLS):
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 RSI Divergence", "📉 Williams %R", "⏱️ Extended Stay Zones", "🔥 Heatmap"])

# ============================================================================
# TAB 1: RSI DIVERGENCE
# ============================================================================
with tab1:
    rsi_results = st.session_state.get('rsi_results', [])
    
    buy_signals = [r for r in rsi_results if 'BUY' in r['signal']]
    sell_signals = [r for r in rsi_results if 'SELL' in r['signal']]
    
    buy_signals.sort(key=lambda x: x['score'], reverse=True)
    sell_signals.sort(key=lambda x: x['score'], reverse=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🟢 BUY", len(buy_signals))
    col2.metric("🔴 SELL", len(sell_signals))
    col3.metric("⚪ No Signal", st.session_state.get('no_signal_count', 0))
    col4.metric("❌ No Data", st.session_state.get('no_data_count', 0))
    col5.metric("📊 Total", len(NIFTY_200_SYMBOLS))
    
    st.markdown("---")
    
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
            st.info("No BUY signals")
    
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
            st.info("No SELL signals")
    
    st.caption("💡 Score: 80+ = Very Strong | 60-79 = Strong | 40-59 = Moderate")

# ============================================================================
# TAB 2: WILLIAMS %R
# ============================================================================
with tab2:
    wr_results = st.session_state.get('wr_results', [])
    
    extreme_ob_1h = [r for r in wr_results if r.get('zone_1h') == "EXTREME OVERBOUGHT"]
    extreme_os_1h = [r for r in wr_results if r.get('zone_1h') == "EXTREME OVERSOLD"]
    normal_ob_1h = [r for r in wr_results if r.get('zone_1h') == "OVERBOUGHT"]
    normal_os_1h = [r for r in wr_results if r.get('zone_1h') == "OVERSOLD"]
    double_confirm = [r for r in wr_results if r.get('double_confirmation', False)]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔴 Extreme OB", len(extreme_ob_1h))
    col2.metric("🟢 Extreme OS", len(extreme_os_1h))
    col3.metric("🟠 Overbought", len(normal_ob_1h))
    col4.metric("🟢 Oversold", len(normal_os_1h))
    col5.metric("⚡ Double", len(double_confirm))
    
    st.markdown("---")
    
    # Double confirmation
    if double_confirm:
        st.subheader("🎯 DOUBLE CONFIRMATION (1H + 2H Extreme)")
        col1, col2 = st.columns(2)
        
        double_ob = [r for r in double_confirm if r.get('zone_1h') == "EXTREME OVERBOUGHT"]
        double_os = [r for r in double_confirm if r.get('zone_1h') == "EXTREME OVERSOLD"]
        
        with col1:
            st.markdown("**🔴 EXTREME OVERBOUGHT**")
            if double_ob:
                df = pd.DataFrame(double_ob)
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'symbol', 'williams_r_1h', 'williams_r_2h', 'price']]
                df.columns = ['Rank', 'Stock', '1H W%R', '2H W%R', 'Price (₹)']
                df['1H W%R'] = df['1H W%R'].round(2)
                df['2H W%R'] = df['2H W%R'].round(2)
                df['Price (₹)'] = df['Price (₹)'].round(2)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No double OB")
        
        with col2:
            st.markdown("**🟢 EXTREME OVERSOLD**")
            if double_os:
                df = pd.DataFrame(double_os)
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'symbol', 'williams_r_1h', 'williams_r_2h', 'price']]
                df.columns = ['Rank', 'Stock', '1H W%R', '2H W%R', 'Price (₹)']
                df['1H W%R'] = df['1H W%R'].round(2)
                df['2H W%R'] = df['2H W%R'].round(2)
                df['Price (₹)'] = df['Price (₹)'].round(2)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No double OS")
        
        st.markdown("---")
    
    # 1H Extreme zones
    st.subheader("⚡ 1H Extreme Zones")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 EXTREME OVERBOUGHT (-5 to 0)**")
        if extreme_ob_1h:
            df = pd.DataFrame(extreme_ob_1h)
            df['Rank'] = range(1, len(df) + 1)
            df['⚡'] = df['double_confirmation'].apply(lambda x: '⚡' if x else '')
            df = df[['Rank', '⚡', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', '⚡', 'Stock', 'W%R', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No extreme OB")
    
    with col2:
        st.markdown("**🟢 EXTREME OVERSOLD (-95 to -100)**")
        if extreme_os_1h:
            df = pd.DataFrame(extreme_os_1h)
            df['Rank'] = range(1, len(df) + 1)
            df['⚡'] = df['double_confirmation'].apply(lambda x: '⚡' if x else '')
            df = df[['Rank', '⚡', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', '⚡', 'Stock', 'W%R', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No extreme OS")
    
    st.markdown("---")
    
    # 1H Normal zones
    st.subheader("📊 1H Normal Zones")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟠 OVERBOUGHT (-20 to -5)**")
        if normal_ob_1h:
            df = pd.DataFrame(normal_ob_1h)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'W%R', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No OB")
    
    with col2:
        st.markdown("**🟢 OVERSOLD (-95 to -80)**")
        if normal_os_1h:
            df = pd.DataFrame(normal_os_1h)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'williams_r_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'W%R', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No OS")

# ============================================================================
# TAB 3: EXTENDED STAY ZONES
# ============================================================================
with tab3:
    wr_results = st.session_state.get('wr_results', [])
    
    extended_stay = [r for r in wr_results if r.get('extended_stay', False)]
    
    extended_ob = [r for r in extended_stay if 'OVERBOUGHT' in r.get('zone_1h', '')]
    extended_os = [r for r in extended_stay if 'OVERSOLD' in r.get('zone_1h', '')]
    
    extended_ob.sort(key=lambda x: x['hours_in_zone_1h'], reverse=True)
    extended_os.sort(key=lambda x: x['hours_in_zone_1h'], reverse=True)
    
    st.subheader("⏱️ Stocks in Zones for 3+ Hours")
    st.markdown("**These stocks have been stuck in overbought/oversold zones - potential for strong reversal**")
    
    col1, col2 = st.columns(2)
    col1.metric("🔴 Extended Overbought", len(extended_ob))
    col2.metric("🟢 Extended Oversold", len(extended_os))
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔴 OVERBOUGHT 3+ Hours")
        st.markdown("*Prolonged overbought - reversal likely*")
        if extended_ob:
            df = pd.DataFrame(extended_ob)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'hours_in_zone_1h', 'williams_r_1h', 'zone_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'Hours', 'W%R', 'Zone', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in extended overbought")
    
    with col_right:
        st.subheader("🟢 OVERSOLD 3+ Hours")
        st.markdown("*Prolonged oversold - bounce likely*")
        if extended_os:
            df = pd.DataFrame(extended_os)
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'symbol', 'hours_in_zone_1h', 'williams_r_1h', 'zone_1h', 'price']]
            df.columns = ['Rank', 'Stock', 'Hours', 'W%R', 'Zone', 'Price (₹)']
            df['W%R'] = df['W%R'].round(2)
            df['Price (₹)'] = df['Price (₹)'].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in extended oversold")
    
    st.caption("💡 **Extended Stay:** Stocks that stayed in zone for multiple hours are building energy for reversal")
    st.caption("⚡ **Higher hours = Higher probability** of reversal when breakout happens")

# ============================================================================
# TAB 4: HEATMAP
# ============================================================================
with tab4:
    st.subheader("🔥 Technical Indicators Heatmap")
    st.markdown("**Select up to 8 stocks to visualize RSI and Williams %R heatmaps**")
    
    # Create list of stock names without .NS
    stock_names = [s.replace('.NS', '') for s in NIFTY_200_SYMBOLS]
    
    # Multi-select dropdown
    selected = st.multiselect(
        "Select stocks (max 8):",
        options=stock_names,
        default=[],
        max_selections=8,
        help="Choose stocks to compare their RSI and Williams %R patterns"
    )
    
    if selected:
        # Convert back to .NS format
        selected_symbols = [s + '.NS' for s in selected]
        
        with st.spinner(f"Loading heatmap data for {len(selected)} stocks..."):
            fig = create_heatmap(selected_symbols)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("**🎨 Color Guide:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **RSI:**
                    - 🟢 Green (40-60): Neutral zone
                    - 🟠 Orange (30-40, 60-70): Caution zones  
                    - 🔴 Red (<30, >70): Oversold/Overbought
                    """)
                with col2:
                    st.markdown("""
                    **Williams %R:**
                    - 🟢 Green (-60 to -40): Neutral zone
                    - 🟠 Orange (-80 to -60, -40 to -20): Caution zones
                    - 🔴 Red (<-80, >-20): Oversold/Overbought  
                    """)
            else:
                st.error("Unable to load heatmap data. Please try again.")
    else:
        st.info("👆 Select stocks from the dropdown above to view heatmap")
        
        # Show some suggested stocks
        st.markdown("**💡 Suggested stocks to analyze:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Large Cap:**")
            st.markdown("• RELIANCE\n• TCS\n• HDFCBANK\n• INFY")
        
        with col2:
            st.markdown("**Mid Cap:**")
            st.markdown("• DIXON\n• POLYCAB\n• TRENT\n• ZOMATO")
        
        with col3:
            st.markdown("**Bank/Financial:**")
            st.markdown("• ICICIBANK\n• AXISBANK\n• BAJFINANCE")

st.divider()
