import logging
from dhanhq import dhanhq
import credentials
from test_functions import test_authentication, get_latest_expiry_date
import pandas as pd
import os
import time
from datetime import datetime, time as dt_time
import threading
from collections import deque
import numpy as np
import requests  # Added for direct API calls

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading Configuration
INDEX = 'SENSEX'
SECURITY_ID = 1
EXCHANGE = 'BSE_FNO'
PRODUCT = 'INTRADAY'
TIMEFRAME = '5min'

# Risk Management
STOP_LOSS_PERCENT = 15  # 15% risk per trade
TARGET_POINTS = 200     # 200 points target on premium
MAX_TRADES_PER_DAY = 3

# Trading Hours
ALGO_START_TIME = dt_time(9, 16)    # 9:16 AM
ALGO_END_TIME = dt_time(15, 30)     # 3:30 PM
AUTO_SQUARE_OFF_TIME = dt_time(15, 14)  # 3:14 PM

# EMA Configuration
EMA_SHORT = 9
EMA_LONG = 21
PRICE_HISTORY_SIZE = 2000  # Store last 2000 candles for EMA calculation

# Use absolute path for CSV file
INSTRUMENTS_CSV = os.path.join(os.path.dirname(__file__), 'dhan_instruments.csv')

class SensexOptionAlgo:
    def __init__(self):
        self.dhan = test_authentication(credentials.DHAN_CLIENT_ID, credentials.DHAN_ACCESS_TOKEN)
        self.is_running = False
        self.trades_today = 0
        self.current_position = None
        self.price_history = deque(maxlen=PRICE_HISTORY_SIZE)
        self.ema_9_values = deque(maxlen=10)  # Keep last 10 EMA values for crossover detection
        self.ema_21_values = deque(maxlen=10)  # Keep last 10 EMA values for crossover detection
        self.last_signal_time = None
        self.options_df = None
        self.daily_trades = []  # Store daily trade history
        self.option_chain_data = None  # Store full option chain data
        self.nearest_expiry = None  # Store nearest expiry date
        
        # Signal tracking for EMA crossovers
        self.last_ema_crossover_signal = None  # 'bullish' or 'bearish' or None
        self.last_ema_crossover_time = None
        self.signal_acted_upon = False  # Track if we acted on the current signal
        self.previous_ema_9 = None
        self.previous_ema_21 = None
        
        # Daily P&L tracking
        self.daily_pnl = 0
        self.total_brokerage = 0
        
        # Price data caching to reduce API calls
        self.cached_option_data = None
        self.last_option_chain_fetch = 0
        
        # Initialize options data
        self._load_options_data()
    
    def _load_options_data(self):
        """Load and filter SENSEX options data for nearest expiry using Dhan API"""
        try:
            base_url = "https://api.dhan.co"
            headers = {
                "access-token": credentials.DHAN_ACCESS_TOKEN,
                "client-id": credentials.DHAN_CLIENT_ID,
                "Content-Type": "application/json"
            }

            # Step 1: Get expiry list
            expiry_url = f"{base_url}/v2/optionchain/expirylist"
            payload = {
                "UnderlyingScrip": SECURITY_ID,
                "UnderlyingSeg": "BSE_EQ"  # SENSEX underlying segment
            }

            logger.info("Fetching SENSEX expiry dates...")
            expiry_response = requests.post(expiry_url, headers=headers, json=payload, timeout=10)
            expiry_response.raise_for_status()
            expiry_data = expiry_response.json()

            if not expiry_data.get('data'):
                raise Exception(f"Failed to get expiry list: {expiry_data}")

            # Get nearest expiry
            nearest_expiry = expiry_data['data'][0]
            self.nearest_expiry = nearest_expiry
            logger.info(f"Nearest expiry: {nearest_expiry}")
            
            # Add delay between API calls
            time.sleep(2)

            # Step 2: Get option chain for nearest expiry
            option_url = f"{base_url}/v2/optionchain"
            option_payload = {
                "UnderlyingScrip": SECURITY_ID,
                "UnderlyingSeg": "BSE_EQ",
                "Expiry": nearest_expiry
            }

            logger.info(f"Fetching option chain for expiry {nearest_expiry}...")
            option_response = requests.post(option_url, headers=headers, json=option_payload, timeout=10)
            option_response.raise_for_status()
            option_data = option_response.json()

            if not option_data.get('data'):
                raise Exception(f"Failed to get option chain: {option_data}")

            # Store the full option chain data
            self.option_chain_data = option_data['data']
            
            # Create a simplified DataFrame for compatibility
            options_list = []
            for strike, option_info in self.option_chain_data.get('oc', {}).items():
                # CE options
                ce_data = option_info.get('ce', {})
                if ce_data:
                    options_list.append({
                        'TRADING_SYMBOL': f'SENSEX{strike}CE',
                        'STRIKE_PRICE': float(strike),
                        'OPTION_TYPE': 'CE',
                        'SEM_SMST_SECURITY_ID': ce_data.get('security_id', f'CE_{strike}'),
                        'LOT_SIZE': ce_data.get('lot_size', 10)
                    })
                
                # PE options
                pe_data = option_info.get('pe', {})
                if pe_data:
                    options_list.append({
                        'TRADING_SYMBOL': f'SENSEX{strike}PE',
                        'STRIKE_PRICE': float(strike),
                        'OPTION_TYPE': 'PE',
                        'SEM_SMST_SECURITY_ID': pe_data.get('security_id', f'PE_{strike}'),
                        'LOT_SIZE': pe_data.get('lot_size', 10)
                    })
            
            self.options_df = pd.DataFrame(options_list)
            
            if self.options_df.empty:
                raise Exception(f"No SENSEX options found for nearest expiry {nearest_expiry}")
            
            logger.info(f"Loaded {len(self.options_df)} SENSEX options for expiry {nearest_expiry}")
            
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
            raise
    
    def calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average
        prices: list or deque of price values
        period: EMA period (9 or 21)
        """
        if len(prices) < period:
            return None
        
        prices_list = list(prices)
        multiplier = 2 / (period + 1)
        
        # Start with SMA for the first calculation
        sma = sum(prices_list[:period]) / period
        ema = sma
        
        # Calculate EMA for remaining values
        for price in prices_list[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def get_sensex_price(self):
        """
        Fetch current SENSEX spot price using working dhanhq method
        """
        try:
            logger.info("🔍 Fetching SENSEX price...")
            # Use the working approach from spot-algo.py
            sensex_spot_price = self.dhan.ticker_data({'IDX_I': [51]})
            logger.debug(f"Raw API response: {sensex_spot_price}")
            
            if sensex_spot_price.get('status') == 'success':
                sensex_price = sensex_spot_price.get('data', {}).get('data', {}).get('IDX_I', {}).get('51', {}).get('last_price')
                
                if sensex_price and sensex_price > 0:
                    price_data = {
                        'last_price': float(sensex_price),
                        'open': float(sensex_price),  # Using current as fallback
                        'high': float(sensex_price),  
                        'low': float(sensex_price),   
                        'volume': 1000000
                    }
                    
                    logger.info(f"✅ SENSEX Price fetched: ₹{price_data['last_price']:.2f}")
                    return price_data
                else:
                    logger.warning(f"❌ SENSEX price is 0 or invalid: {sensex_price}")
            else:
                logger.warning(f"❌ Failed to get SENSEX data: {sensex_spot_price.get('remarks', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching SENSEX price: {e}")
            
        return None
    
    def update_price_history(self, price_data):
        """
        Update price history and calculate EMAs
        price_data: dict with OHLC information
        """
        if not price_data or 'last_price' not in price_data:
            logger.error("❌ Invalid price data received")
            return False
        
        current_price = price_data['last_price']
        current_time = datetime.now()
        
        logger.info(f"📊 Updating price history: ₹{current_price:.2f} at {current_time.strftime('%H:%M:%S')}")
        
        # Add to price history
        self.price_history.append({
            'time': current_time,
            'price': current_price,
            'open': price_data.get('open', current_price),
            'high': price_data.get('high', current_price),
            'low': price_data.get('low', current_price),
            'volume': price_data.get('volume', 0)
        })
        
        logger.debug(f"📈 Price history size: {len(self.price_history)} entries")
        
        # Deque automatically handles size limit with maxlen, no need to manually trim
        logger.debug(f"🗂️ Price history size: {len(self.price_history)} entries")
        
        # Calculate EMAs if we have enough data
        prices = [entry['price'] for entry in self.price_history]
        
        logger.info(f"🧮 Calculating EMAs with {len(prices)} price points...")
        
        # Calculate 9-period EMA
        ema_9 = None
        if len(prices) >= EMA_SHORT:
            ema_9 = self.calculate_ema(prices, EMA_SHORT)
            logger.info(f"📊 9-EMA calculated: ₹{ema_9:.2f}")
        else:
            logger.info(f"⏳ Need {EMA_SHORT - len(prices)} more prices for 9-EMA")
        
        # Calculate 21-period EMA  
        ema_21 = None
        if len(prices) >= EMA_LONG:
            ema_21 = self.calculate_ema(prices, EMA_LONG)
            logger.info(f"📊 21-EMA calculated: ₹{ema_21:.2f}")
        else:
            logger.info(f"⏳ Need {EMA_LONG - len(prices)} more prices for 21-EMA")
        
        # Store EMAs
        if ema_9 is not None:
            self.ema_9_values.append(ema_9)
            # Keep last 10 values - deque automatically handles maxlen
        
        if ema_21 is not None:
            self.ema_21_values.append(ema_21)
            # Keep last 10 values - deque automatically handles maxlen
        
        # Log EMA comparison if both are available
        if ema_9 is not None and ema_21 is not None:
            if ema_9 > ema_21:
                trend = "🟢 BULLISH"
                logger.info(f"📈 EMA Trend: {trend} (9-EMA: ₹{ema_9:.2f} > 21-EMA: ₹{ema_21:.2f})")
            else:
                trend = "🔴 BEARISH" 
                logger.info(f"📉 EMA Trend: {trend} (9-EMA: ₹{ema_9:.2f} < 21-EMA: ₹{ema_21:.2f})")
        
        ema9_str = f"{ema_9:.2f}" if ema_9 is not None else "N/A"
        ema21_str = f"{ema_21:.2f}" if ema_21 is not None else "N/A"
        logger.debug(f"Updated EMAs - 9 EMA: {ema9_str}, 21 EMA: {ema21_str}")
        
        return True
    
    def get_current_candle_info(self):
        """
        Analyze current candle to determine if it's green or red
        Returns: dict with candle info or None
        """
        if len(self.price_history) < 1:
            return None
        
        current_candle = self.price_history[-1]
        
        open_price = current_candle['open']
        close_price = current_candle['close']
        high_price = current_candle['high']
        low_price = current_candle['low']
        
        is_green = close_price > open_price
        is_red = close_price < open_price
        
        candle_info = {
            'is_green': is_green,
            'is_red': is_red,
            'open': open_price,
            'close': close_price,
            'high': high_price,
            'low': low_price,
            'body_size': abs(close_price - open_price),
            'timestamp': current_candle['timestamp']
        }
        
        return candle_info
    
    def is_trading_time(self):
        """Check if current time is within trading hours"""
        current_time = datetime.now().time()
        return ALGO_START_TIME <= current_time <= ALGO_END_TIME
    
    def should_auto_square_off(self):
        """Check if it's time for auto square-off"""
        current_time = datetime.now().time()
        return current_time >= AUTO_SQUARE_OFF_TIME
    
    def can_place_new_trade(self):
        """Check if we can place a new trade (within daily limits and no existing position)"""
        return (self.trades_today < MAX_TRADES_PER_DAY and 
                self.current_position is None and 
                self.is_trading_time() and 
                not self.should_auto_square_off())
    
    def reset_ema_signals(self):
        """Reset EMA signals after a trade exits to wait for fresh crossover"""
        self.last_ema_crossover_signal = None
        self.last_ema_crossover_time = None
        self.signal_acted_upon = False
        logger.info("EMA signals reset - waiting for fresh crossover signal")
    
    def detect_ema_crossover(self):
        """
        Detect EMA crossover and return signal type
        Returns: 'bullish', 'bearish', or None
        Only returns signal if it's a NEW crossover (not acted upon)
        """
        if len(self.ema_9_values) < 2 or len(self.ema_21_values) < 2:
            logger.debug("🔍 Insufficient EMA data for crossover detection")
            return None
        
        current_ema_9 = self.ema_9_values[-1]
        current_ema_21 = self.ema_21_values[-1]
        
        logger.info(f"🔍 Checking EMA crossover - Current: 9EMA=₹{current_ema_9:.2f}, 21EMA=₹{current_ema_21:.2f}")
        
        if self.previous_ema_9 is None or self.previous_ema_21 is None:
            self.previous_ema_9 = self.ema_9_values[-2] if len(self.ema_9_values) >= 2 else None
            self.previous_ema_21 = self.ema_21_values[-2] if len(self.ema_21_values) >= 2 else None
            logger.info(f"📊 Setting previous EMAs - Prev: 9EMA=₹{self.previous_ema_9:.2f if self.previous_ema_9 else 'N/A'}, 21EMA=₹{self.previous_ema_21:.2f if self.previous_ema_21 else 'N/A'}")
            return None
        
        logger.info(f"📊 Previous EMAs - 9EMA=₹{self.previous_ema_9:.2f}, 21EMA=₹{self.previous_ema_21:.2f}")
        
        # Check for bullish crossover: 9 EMA crosses above 21 EMA
        if (self.previous_ema_9 <= self.previous_ema_21 and 
            current_ema_9 > current_ema_21):
            
            logger.info("🚨 BULLISH CROSSOVER DETECTED!")
            logger.info(f"   Previous: 9EMA(₹{self.previous_ema_9:.2f}) <= 21EMA(₹{self.previous_ema_21:.2f})")
            logger.info(f"   Current:  9EMA(₹{current_ema_9:.2f}) > 21EMA(₹{current_ema_21:.2f})")
            
            # Only signal if this is a new crossover (not already acted upon)
            if (self.last_ema_crossover_signal != 'bullish' or 
                self.signal_acted_upon):
                
                self.last_ema_crossover_signal = 'bullish'
                self.last_ema_crossover_time = datetime.now()
                self.signal_acted_upon = False
                logger.info("✅ NEW BULLISH Signal generated")
                
                # Update previous values for next iteration
                self.previous_ema_9 = current_ema_9
                self.previous_ema_21 = current_ema_21
                
                return 'bullish'
            else:
                logger.info("⏸️ Bullish crossover already acted upon")
        
        # Check for bearish crossover: 21 EMA crosses above 9 EMA  
        elif (self.previous_ema_9 >= self.previous_ema_21 and 
              current_ema_9 < current_ema_21):
            
            logger.info("🚨 BEARISH CROSSOVER DETECTED!")
            logger.info(f"   Previous: 9EMA(₹{self.previous_ema_9:.2f}) >= 21EMA(₹{self.previous_ema_21:.2f})")
            logger.info(f"   Current:  9EMA(₹{current_ema_9:.2f}) < 21EMA(₹{current_ema_21:.2f})")
            
            # Only signal if this is a new crossover (not already acted upon)
            if (self.last_ema_crossover_signal != 'bearish' or 
                self.signal_acted_upon):
                
                self.last_ema_crossover_signal = 'bearish'
                self.last_ema_crossover_time = datetime.now()
                self.signal_acted_upon = False
                logger.info("✅ NEW BEARISH Signal generated")
                
                # Update previous values for next iteration
                self.previous_ema_9 = current_ema_9
                self.previous_ema_21 = current_ema_21
                
                return 'bearish'
            else:
                logger.info("⏸️ Bearish crossover already acted upon")
        else:
            logger.debug("📊 No crossover detected - EMAs maintaining trend")
        
        # Update previous values for next iteration
        self.previous_ema_9 = current_ema_9
        self.previous_ema_21 = current_ema_21
        
        return None
        
        return None
    
    def mark_signal_as_acted_upon(self):
        """Mark the current EMA signal as acted upon to prevent re-entry"""
        self.signal_acted_upon = True
        logger.info(f"Signal marked as acted upon: {self.last_ema_crossover_signal}")
    
    def has_valid_fresh_signal(self, signal_type):
        """
        Check if we have a valid fresh signal that hasn't been acted upon
        signal_type: 'bullish' or 'bearish'
        """
        return (self.last_ema_crossover_signal == signal_type and 
                not self.signal_acted_upon)
    
    def validate_entry_conditions(self, crossover_signal):
        """
        Validate entry conditions based on EMA crossover and candle analysis
        crossover_signal: 'bullish' or 'bearish'
        Returns: 'CALL' for call entry, 'PUT' for put entry, or None
        """
        if not crossover_signal or len(self.ema_9_values) < 1:
            return None
        
        candle_info = self.get_current_candle_info()
        if not candle_info:
            return None
        
        current_ema_9 = self.ema_9_values[-1]
        close_price = candle_info['close']
        
        # CALL ENTRY CONDITIONS:
        # 1. 9 EMA crossed above 21 EMA (bullish crossover)
        # 2. Green candle (close > open)  
        # 3. Green candle closes above 9 EMA
        if (crossover_signal == 'bullish' and 
            candle_info['is_green'] and 
            close_price > current_ema_9):
            
            logger.info(f"CALL Entry Conditions Met:")
            logger.info(f"  - Bullish EMA crossover: ✓")
            logger.info(f"  - Green candle: ✓ (Close: {close_price}, Open: {candle_info['open']})")
            logger.info(f"  - Close above 9 EMA: ✓ (Close: {close_price}, 9 EMA: {current_ema_9:.2f})")
            return 'CALL'
        
        # PUT ENTRY CONDITIONS:
        # 1. 21 EMA crossed above 9 EMA (bearish crossover)
        # 2. Red candle (close < open)
        # 3. Red candle closes below 9 EMA  
        elif (crossover_signal == 'bearish' and 
              candle_info['is_red'] and 
              close_price < current_ema_9):
            
            logger.info(f"PUT Entry Conditions Met:")
            logger.info(f"  - Bearish EMA crossover: ✓")
            logger.info(f"  - Red candle: ✓ (Close: {close_price}, Open: {candle_info['open']})")
            logger.info(f"  - Close below 9 EMA: ✓ (Close: {close_price}, 9 EMA: {current_ema_9:.2f})")
            return 'PUT'
        
        else:
            # Log why conditions weren't met
            if crossover_signal == 'bullish':
                logger.debug(f"CALL conditions not met:")
                logger.debug(f"  - Green candle: {candle_info['is_green']}")
                logger.debug(f"  - Close above 9 EMA: {close_price > current_ema_9} "
                           f"({close_price} vs {current_ema_9:.2f})")
            elif crossover_signal == 'bearish':
                logger.debug(f"PUT conditions not met:")
                logger.debug(f"  - Red candle: {candle_info['is_red']}")
                logger.debug(f"  - Close below 9 EMA: {close_price < current_ema_9} "
                           f"({close_price} vs {current_ema_9:.2f})")
        
        return None
    
    def find_nearest_strike_option(self, option_type, current_price):
        """
        Find the nearest strike price option (ATM)
        option_type: 'CE' for Call, 'PE' for Put
        current_price: Current SENSEX price
        """
        try:
            # Filter options by type
            options = self.options_df[self.options_df['OPTION_TYPE'] == option_type].copy()
            
            if options.empty:
                logger.error(f"No {option_type} options found")
                return None
            
            # Calculate distance from current price to each strike
            options['strike_distance'] = abs(options['STRIKE_PRICE'] - current_price)
            
            # Find the nearest strike (ATM)
            nearest_option = options.loc[options['strike_distance'].idxmin()]
            
            logger.info(f"Selected {option_type} option: {nearest_option['TRADING_SYMBOL']} "
                       f"(Strike: {nearest_option['STRIKE_PRICE']}, Distance: {nearest_option['strike_distance']:.0f})")
            
            return nearest_option
            
        except Exception as e:
            logger.error(f"Error finding nearest strike option: {e}")
            return None
    
    def place_option_order(self, option_type, current_price):
        """
        Place buy order for nearest strike option (Mock implementation for now)
        option_type: 'CE' for Call, 'PE' for Put
        current_price: Current SENSEX price for strike selection
        """
        try:
            # Find nearest strike option
            option_data = self.find_nearest_strike_option(option_type, current_price)
            if not option_data:
                logger.error(f"Could not find suitable {option_type} option")
                return None
            
            # Refresh option chain to get latest prices
            if not self.refresh_option_chain():
                logger.error("Could not refresh option chain data")
                return None
            
            # Get current option price from refreshed data
            current_premium = self.get_option_current_price(option_type, option_data['STRIKE_PRICE'])
            if not current_premium or current_premium <= 0:
                logger.error(f"Invalid option premium: {current_premium}")
                return None

            # Calculate quantity (for simplicity, using 1 lot)
            quantity = int(option_data.get('LOT_SIZE', 10))
            
            # Calculate stop loss price (15% below entry)
            stop_loss_price = current_premium * (1 - STOP_LOSS_PERCENT / 100)
            target_price = current_premium + TARGET_POINTS
            
            # Mock order placement for now (replace with actual Dhan API call later)
            logger.info("🚧 MOCK ORDER PLACEMENT - Would place real order here")
            order_id = f"MOCK_{datetime.now().strftime('%H%M%S')}_{current_premium:.0f}"
            
            # Store position information
            self.current_position = {
                'order_id': order_id,
                'option_type': option_type,
                'symbol': option_data['TRADING_SYMBOL'],
                'strike_price': option_data['STRIKE_PRICE'],
                'security_id': option_data['SEM_SMST_SECURITY_ID'],
                'entry_price': current_premium,
                'stop_loss_price': stop_loss_price,
                'target_price': target_price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'instrument_token': f"BSE_FNO:{option_data['SEM_SMST_SECURITY_ID']}"
            }
            
            self.trades_today += 1
            
            logger.info(f"✅ {option_type} MOCK ORDER PLACED!")
            logger.info(f"   Symbol: {option_data['TRADING_SYMBOL']}")
            logger.info(f"   Strike: {option_data['STRIKE_PRICE']}")
            logger.info(f"   Entry Premium: ₹{current_premium}")
            logger.info(f"   Stop Loss: ₹{stop_loss_price:.2f}")
            logger.info(f"   Target: ₹{target_price:.2f}")
            logger.info(f"   Quantity: {quantity}")
            logger.info(f"   Mock Order ID: {order_id}")
            
            return self.current_position
                
        except Exception as e:
            logger.error(f"Error placing option order: {e}")
            return None
            
    def get_option_current_price(self, option_type, strike_price):
        """
        Get current option price from cached option chain data
        """
        try:
            if not self.option_chain_data:
                logger.warning("No option chain data available")
                return None
            
            strike_str = str(int(strike_price))
            option_info = self.option_chain_data.get('oc', {}).get(strike_str, {})
            
            if option_type == 'CE':
                ce_data = option_info.get('ce', {})
                if ce_data:
                    return ce_data.get('last_price', 0)
            elif option_type == 'PE':
                pe_data = option_info.get('pe', {})
                if pe_data:
                    return pe_data.get('last_price', 0)
            
            logger.warning(f"No price data found for {option_type} {strike_price}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting option price: {e}")
            return None
    
    def refresh_option_chain(self):
        """Refresh option chain data to get latest prices"""
        try:
            base_url = "https://api.dhan.co"
            headers = {
                "access-token": credentials.DHAN_ACCESS_TOKEN,
                "client-id": credentials.DHAN_CLIENT_ID,
                "Content-Type": "application/json"
            }

            option_url = f"{base_url}/v2/optionchain"
            payload = {
                "UnderlyingScrip": SECURITY_ID,
                "UnderlyingSeg": "BSE_EQ",
                "Expiry": self.nearest_expiry
            }

            response = requests.post(option_url, headers=headers, json=payload)
            response.raise_for_status()
            
            option_data = response.json()
            
            if option_data.get('status') == 'success' and 'data' in option_data:
                self.option_chain_data = option_data['data']
                return True
            else:
                logger.error(f"Failed to refresh option chain: {option_data.get('remarks', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing option chain: {e}")
            return False
    
    def check_position_pnl(self):
        """
        Check current position P&L and trigger SL/Target if needed
        """
        if not self.current_position:
            return
        
        try:
            # Refresh option chain to get current prices
            if not self.refresh_option_chain():
                logger.warning("Could not refresh option chain data for P&L check")
                return
            
            # Get current option price from refreshed data
            current_premium = self.get_option_current_price(
                self.current_position['option_type'], 
                self.current_position['strike_price']
            )
            
            if not current_premium or current_premium <= 0:
                logger.warning(f"Invalid current premium: {current_premium}")
                return
            
            entry_price = self.current_position['entry_price']
            stop_loss_price = self.current_position['stop_loss_price']
            target_price = self.current_position['target_price']
            
            pnl_points = current_premium - entry_price
            pnl_percent = (pnl_points / entry_price) * 100
            
            logger.debug(f"Position PnL: ₹{pnl_points:.2f} ({pnl_percent:.2f}%) "
                        f"[Current: ₹{current_premium}, Entry: ₹{entry_price}]")
            
            # Check stop loss condition
            if current_premium <= stop_loss_price:
                logger.warning(f"🔴 STOP LOSS HIT! Current: ₹{current_premium}, SL: ₹{stop_loss_price:.2f}")
                self.exit_trade("Stop Loss Hit")
                return
            
            # Check target condition
            if current_premium >= target_price:
                logger.info(f"🎯 TARGET ACHIEVED! Current: ₹{current_premium}, Target: ₹{target_price:.2f}")
                self.exit_trade("Target Achieved")
                return
                
        except Exception as e:
            logger.error(f"Error checking position P&L: {e}")
    
    def exit_trade(self, reason="Manual Exit"):
        """
        Exit current trade and reset monitoring (Mock implementation for now)
        """
        if not self.current_position:
            logger.warning("No active position to exit")
            return
        
        try:
            logger.info(f"🔄 Exiting trade: {reason}")
            
            # Get current premium for P&L calculation using option chain
            current_premium = 0
            try:
                if self.refresh_option_chain():
                    current_premium = self.get_option_current_price(
                        self.current_position['option_type'], 
                        self.current_position['strike_price']
                    )
                if not current_premium or current_premium <= 0:
                    current_premium = self.current_position['entry_price']  # Fallback
            except:
                logger.warning("Could not fetch exit premium for P&L calculation")
                current_premium = self.current_position['entry_price']  # Fallback
            
            # Mock exit order placement
            logger.info("🚧 MOCK EXIT ORDER - Would place real sell order here")
            exit_order_id = f"MOCK_EXIT_{datetime.now().strftime('%H%M%S')}_{current_premium:.0f}"
            
            # Calculate trade P&L
            entry_premium = self.current_position['entry_price']
            exit_premium = current_premium
            quantity = self.current_position['quantity']
            
            trade_pnl = (exit_premium - entry_premium) * quantity
            self.daily_pnl += trade_pnl
            
            # Log trade summary
            entry_time = self.current_position['entry_time']
            exit_time = datetime.now()
            duration = exit_time - entry_time
            
            # Store trade record
            trade_record = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration': duration,
                'symbol': self.current_position['symbol'],
                'option_type': self.current_position['option_type'],
                'strike_price': self.current_position['strike_price'],
                'entry_premium': entry_premium,
                'exit_premium': exit_premium,
                'quantity': quantity,
                'pnl': trade_pnl,
                'exit_reason': reason,
                'entry_order_id': self.current_position['order_id'],
                'exit_order_id': exit_order_id
            }
            
            self.daily_trades.append(trade_record)
            
            logger.info(f"✅ MOCK TRADE EXITED SUCCESSFULLY!")
            logger.info(f"   Symbol: {self.current_position['symbol']}")
            logger.info(f"   Entry Time: {entry_time.strftime('%H:%M:%S')}")
            logger.info(f"   Exit Time: {exit_time.strftime('%H:%M:%S')}")
            logger.info(f"   Duration: {duration}")
            logger.info(f"   Entry Premium: ₹{entry_premium}")
            logger.info(f"   Exit Premium: ₹{exit_premium}")
            logger.info(f"   Trade P&L: ₹{trade_pnl:.2f}")
            logger.info(f"   Exit Reason: {reason}")
            logger.info(f"   Mock Exit Order ID: {exit_order_id}")
            logger.info(f"   Trades Today: {self.trades_today}/{MAX_TRADES_PER_DAY}")
            logger.info(f"   Daily P&L: ₹{self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error exiting trade: {e}")
        
        finally:
            # Clean up position and reset signals
            self.current_position = None
            self.reset_ema_signals()  # Reset signals after trade exit
            logger.info("🔄 Position cleared and monitoring reset for fresh signals")
    
    def print_daily_summary(self):
        """
        Print daily trading summary
        """
        print(f"\n{'='*70}")
        print(f"📊 DAILY TRADING SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*70}")
        
        if not self.daily_trades:
            print("📝 No trades executed today")
        else:
            print(f"📈 Total Trades: {len(self.daily_trades)}")
            print(f"💰 Daily P&L: ₹{self.daily_pnl:.2f}")
            
            winning_trades = [t for t in self.daily_trades if t['pnl'] > 0]
            losing_trades = [t for t in self.daily_trades if t['pnl'] < 0]
            
            print(f"✅ Winning Trades: {len(winning_trades)}")
            print(f"❌ Losing Trades: {len(losing_trades)}")
            
            if len(self.daily_trades) > 0:
                win_rate = (len(winning_trades) / len(self.daily_trades)) * 100
                print(f"📊 Win Rate: {win_rate:.1f}%")
            
            print(f"\n{'Trade Details:'}")
            print(f"{'='*70}")
            
            for i, trade in enumerate(self.daily_trades, 1):
                pnl_status = "✅" if trade['pnl'] > 0 else "❌" if trade['pnl'] < 0 else "⚪"
                duration_str = f"{trade['duration'].seconds//3600:02d}:{(trade['duration'].seconds//60)%60:02d}:{trade['duration'].seconds%60:02d}"
                
                print(f"{i}. {pnl_status} {trade['option_type']} {trade['symbol']}")
                print(f"   Time: {trade['entry_time'].strftime('%H:%M:%S')} - {trade['exit_time'].strftime('%H:%M:%S')} ({duration_str})")
                print(f"   Premium: ₹{trade['entry_premium']:.2f} → ₹{trade['exit_premium']:.2f}")
                print(f"   P&L: ₹{trade['pnl']:.2f} | Reason: {trade['exit_reason']}")
                print()
        
        print(f"{'='*70}\n")
    
    def monitor_signals(self):
        """
        Main monitoring loop - fetch prices, update EMAs, check signals and manage positions
        """
        try:
            logger.info("=" * 60)
            logger.info("🔄 Starting monitoring cycle...")
            
            # Fetch current SENSEX price
            logger.debug("Step 1: Fetching SENSEX price...")
            price_data = self.get_sensex_price()
            if not price_data:
                logger.warning("❌ Could not fetch SENSEX price data - skipping cycle")
                return
            
            logger.debug(f"Got price data: {price_data}")
            current_sensex_price = price_data['last_price']
            logger.info(f"📈 Current SENSEX: ₹{current_sensex_price:.2f}")
            
            # Update price history and EMAs
            logger.debug("Step 2: Updating price history and EMAs...")
            if not self.update_price_history(price_data):
                logger.warning("❌ Failed to update price history - skipping cycle")
                return
            
            logger.info("✅ Price history and EMAs updated")
            
            # If we have an active position, check P&L first
            if self.current_position:
                logger.info("Step 3: Checking existing position P&L...")
                self.check_position_pnl()
                # If position was exited due to SL/Target, return early
                if not self.current_position:
                    logger.info("Position exited - ending cycle")
                    return
            
            # Check if we have sufficient data for analysis
            if len(self.ema_9_values) < 2 or len(self.ema_21_values) < 2:
                logger.info(f"⏳ Need more data - 9EMA: {len(self.ema_9_values)}/2, 21EMA: {len(self.ema_21_values)}/2")
                return
            
            logger.info("✅ Sufficient EMA data available for signal analysis")
            
            # Only look for new trades if we don't have an active position
            if not self.current_position and self.can_place_new_trade():
                logger.info("Step 4: Looking for trading signals...")
                
                # Detect EMA crossover
                crossover_signal = self.detect_ema_crossover()
                
                if crossover_signal:
                    logger.info(f"🎯 Crossover signal detected: {crossover_signal.upper()}")
                    # Validate complete entry conditions (EMA + Candle)
                    entry_type = self.validate_entry_conditions(crossover_signal)
                    
                    if entry_type:
                        logger.info(f"🚨 COMPLETE TRADE SIGNAL: {entry_type}")
                        
                        # Mark signal as acted upon to prevent re-entry
                        self.mark_signal_as_acted_upon()
                        
                        # Place actual trade
                        option_type = 'CE' if entry_type == 'CALL' else 'PE'
                        logger.info(f"📋 Placing {option_type} order at SENSEX price ₹{current_sensex_price:.2f}")
                        position = self.place_option_order(option_type, current_sensex_price)
                        
                        if position:
                            logger.info(f"🎯 Trade executed successfully!")
                        else:
                            logger.error(f"❌ Failed to place {entry_type} order")
                            # Reset signal so we can try again on next iteration
                            self.signal_acted_upon = False
                    else:
                        logger.info(f"⏸️ Crossover detected but candle conditions not met")
                else:
                    logger.info("📊 No crossover signals detected")
            elif self.current_position:
                logger.info("📊 Position active - monitoring for exit signals only")
            elif not self.can_place_new_trade():
                logger.info("🚫 Daily trade limit reached - monitoring only")
            
            # Check for auto square-off if position exists
            if self.current_position and self.should_auto_square_off():
                logger.info("⏰ Auto square-off time reached (3:14 PM)")
                self.exit_trade("Auto Square-off at 3:14 PM")
                
            logger.info("🔄 Monitoring cycle completed")
            logger.info("=" * 60)
                
        except Exception as e:
            logger.error(f"❌ Error in signal monitoring: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def print_status_summary(self):
        """
        Print current algorithm status
        """
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n{'='*60}")
        print(f"📊 SENSEX OPTION ALGO STATUS - {current_time}")
        print(f"{'='*60}")
        
        # Market status
        if self.is_trading_time():
            print(f"🟢 Market Status: ACTIVE")
        else:
            print(f"🔴 Market Status: CLOSED")
        
        # Trade limits
        print(f"📈 Trades Today: {self.trades_today}/{MAX_TRADES_PER_DAY}")
        
        # Current position
        if self.current_position:
            pos = self.current_position
            duration = datetime.now() - pos['entry_time']
            duration_str = f"{duration.seconds//3600:02d}:{(duration.seconds//60)%60:02d}:{duration.seconds%60:02d}"
            
            print(f"💼 Active Position: {pos['option_type']} - {pos['symbol']}")
            print(f"   Entry Time: {pos['entry_time'].strftime('%H:%M:%S')}")
            print(f"   Duration: {duration_str}")
            print(f"   Entry Premium: ₹{pos['entry_price']:.2f}")
            print(f"   Stop Loss: ₹{pos['stop_loss_price']:.2f}")
            print(f"   Target: ₹{pos['target_price']:.2f}")
        else:
            print(f"💼 Active Position: None")
        
        # EMA status
        if len(self.ema_9_values) > 0 and len(self.ema_21_values) > 0:
            ema9 = self.ema_9_values[-1]
            ema21 = self.ema_21_values[-1]
            if ema9 is not None and ema21 is not None:
                trend = "BULLISH" if ema9 > ema21 else "BEARISH"
                print(f"📊 EMAs: 9EMA=₹{ema9:.2f}, 21EMA=₹{ema21:.2f} ({trend})")
            else:
                print(f"📊 EMAs: Calculating...")
        else:
            print(f"📊 EMAs: Calculating...")
        
        # Signal status
        if self.last_ema_crossover_signal:
            signal_time = self.last_ema_crossover_time.strftime('%H:%M:%S') if self.last_ema_crossover_time else 'N/A'
            status = "ACTED" if self.signal_acted_upon else "PENDING"
            print(f"📡 Last Signal: {self.last_ema_crossover_signal.upper()} at {signal_time} ({status})")
        else:
            print(f"📡 Last Signal: None")
        
        print(f"{'='*60}\n")
    
    def start_monitoring(self):
        """
        Start the monitoring loop with improved status reporting
        """
        self.is_running = True
        logger.info("🚀 Starting SENSEX Option Algorithm monitoring...")
        
        # Print initial status
        self.print_status_summary()
        
        status_counter = 0
        
        while self.is_running:
            try:
                if self.is_trading_time():
                    self.monitor_signals()
                    
                    # Print status summary every 20 iterations (1 minute at 3s intervals)
                    status_counter += 1
                    if status_counter >= 20:
                        self.print_status_summary()
                        status_counter = 0
                    
                    # Wait 3 seconds minimum as required by Dhan API
                    time.sleep(3)
                    
                else:
                    if datetime.now().time() < ALGO_START_TIME:
                        logger.info(f"⏳ Waiting for market open at {ALGO_START_TIME}")
                        time.sleep(30)  # Check every 30 seconds before market open
                    else:
                        logger.info("🏁 Market closed - stopping algorithm")
                        break
                        
            except KeyboardInterrupt:
                logger.info("🛑 Algorithm stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)
        
        # Final status and cleanup
        if self.current_position:
            logger.warning("⚠️ Active position detected at shutdown")
            self.print_status_summary()
            
            # Ask user if they want to exit the position
            try:
                response = input("\nActive position found. Exit position? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    self.exit_trade("Manual exit at shutdown")
            except:
                logger.info("Auto-exiting position at shutdown")
                self.exit_trade("Auto exit at shutdown")
        
        # Print final daily summary
        if self.daily_trades:
            self.print_daily_summary()
        
        self.is_running = False
        logger.info("🔴 Algorithm monitoring stopped")
    
    def stop_monitoring(self):
        """
        Stop the monitoring loop
        """
        self.is_running = False
        logger.info("Stopping algorithm monitoring...")

if __name__ == "__main__":
    print("🚀 Initializing SENSEX Option Algorithm...")
    try:
        algo = SensexOptionAlgo()
        print("✅ Algorithm initialized successfully.")
        print(f"⏰ Trading Hours: {ALGO_START_TIME} - {ALGO_END_TIME}")
        print(f"⏱️ Auto Square-off Time: {AUTO_SQUARE_OFF_TIME}")
        print(f"📊 Maximum Trades Per Day: {MAX_TRADES_PER_DAY}")
        print(f"⚠️ Risk Management: {STOP_LOSS_PERCENT}% SL, {TARGET_POINTS} points target")
        print(f"📈 EMA Configuration: {EMA_SHORT}-period and {EMA_LONG}-period")
        print(f"⏱️ Monitoring Frequency: Every 3 seconds")
        
        print(f"\n{'='*60}")
        print("📋 TRADING RULES")
        print(f"{'='*60}")
        print("1. Maximum 1 trade allowed at a time")
        print("2. EMA signals reset after each trade exit")
        print("3. Fresh crossover signals required for new trades")
        print("4. No re-entry on old/acted-upon signals")
        print("5. Auto square-off at 3:14 PM")
        print("6. Algorithm runs from 9:16 AM to 3:30 PM")
        
        print(f"\n{'='*60}")
        print("🎯 ENTRY CONDITIONS")
        print(f"{'='*60}")
        print(f"CALL ENTRY:")
        print(f"  ✓ 9 EMA crosses above 21 EMA (Bullish crossover)")
        print(f"  ✓ Green candle (Close > Open)")
        print(f"  ✓ Green candle closes above 9 EMA")
        print(f"\nPUT ENTRY:")
        print(f"  ✓ 21 EMA crosses above 9 EMA (Bearish crossover)")
        print(f"  ✓ Red candle (Close < Open)")
        print(f"  ✓ Red candle closes below 9 EMA")
        
        if algo.dhan:
            print(f"\n✅ Dhan Authentication: SUCCESS")
            print(f"📊 Options Loaded: {len(algo.options_df)} SENSEX contracts available")
            
            print(f"\n{'='*60}")
            print("� STARTING ALGORITHM")
            print(f"{'='*60}")
            
            # Start monitoring automatically
            try:
                current_time = datetime.now().time()
                if ALGO_START_TIME <= current_time <= ALGO_END_TIME:
                    print(f"🚀 Starting algorithm monitoring...")
                    algo.start_monitoring()
                else:
                    if current_time < ALGO_START_TIME:
                        print(f"⏰ Market not yet open. Monitoring will start at {ALGO_START_TIME}")
                    else:
                        print(f"⏰ Market closed. Monitoring will start tomorrow at {ALGO_START_TIME}")
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                
        else:
            print(f"\n❌ Dhan Authentication: FAILED")
            print("❌ Please check your credentials in credentials.py")
            
    except Exception as e:
        print(f"❌ Failed to initialize algorithm: {e}")
        logger.error(f"Initialization error: {e}")
        
        # Print troubleshooting tips
        print(f"\n🔧 TROUBLESHOOTING:")
        print("1. Check internet connection")
        print("2. Verify Dhan credentials in credentials.py")
        print("3. Ensure dhan_instruments.csv exists and is up to date")
        print("4. Check if market is open for options trading")