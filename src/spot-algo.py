from dhanhq import dhanhq
import credentials
from test_functions import test_authentication, fetch_prices, get_latest_expiry_date
import time

dhan = test_authentication(credentials.DHAN_CLIENT_ID, credentials.DHAN_ACCESS_TOKEN)

def get_latest_sensex_price(dhan):
    print("Hello Algo Traders, Fetching latest SENSEX price...")
    sensex_spot_price = dhan.ticker_data({'IDX_I': [51]})
    print(sensex_spot_price, 'sensex_spot_price')
    sensex_price = sensex_spot_price.get('data', {}).get('data', {}).get('IDX_I', {}).get('51', {}).get('last_price')
    print(f"SENSEX last price: {sensex_price}")
    return sensex_price

def fetch_sensex_atm_options(dhan):
    print("Fetching SENSEX ATM options for nearest expiry...")
    
    # Get current SENSEX spot price
    current_price = get_latest_sensex_price(dhan)
    
    if current_price:
        # Round to nearest 100 to get ATM strike (SENSEX options typically have 100 point intervals)
        atm_strike = round(current_price / 100) * 100
        print(f"ATM Strike: {atm_strike}")
        
        # For SENSEX 80900 options expiring today (2025-07-29)
        # SENSEX 80900 CE: sec_id = 1140065
        # SENSEX 80900 PE: sec_id = 1140425
        expiry_date = "2025-07-29"
        print(f"Using expiry: {expiry_date}")
        
        # Fetch ATM Call option price (80900 CE)
        print("Fetching SENSEX 80900 CE prices...")
        call_prices = fetch_prices(dhan, 826411, 'BSE', expiry_date)
        if call_prices:
            print(f"SENSEX {atm_strike} CE prices: {call_prices}")
        
        # Add delay to avoid rate limiting
        time.sleep(6)
        
        # Fetch ATM Put option price (80900 PE)
        print("Fetching SENSEX 80900 PE prices...")
        put_prices = fetch_prices(dhan, 826411, 'BSE', expiry_date)
        if put_prices:
            print(f"SENSEX {atm_strike} PE prices: {put_prices}")
    else:
        print("Could not fetch SENSEX spot price")

# Execute the function
fetch_sensex_atm_options(dhan)
