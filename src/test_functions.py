from dhanhq import dhanhq

def test_authentication(client_id, access_token):
    # Test authentication with valid credentials
    dhan = dhanhq(client_id, access_token)
    return dhan

def get_latest_expiry_date(dhan, sec_id, exchange):
    expiry_response = dhan.expiry_list(sec_id, exchange)
    print(f"Expiry date for {exchange}: {expiry_response}")
    
    # Check if the API request was successful
    if expiry_response.get('status') == 'success':
        expiry_dates_list = expiry_response['data']['data']
        if expiry_dates_list and len(expiry_dates_list) > 0:
            nearest_expiry_api_str = expiry_dates_list[0]
            print(f"Nearest expiry date: {nearest_expiry_api_str}")
            return nearest_expiry_api_str
        else:
            print("No expiry dates available")
            return None
    else:
        print(f"API request failed: {expiry_response.get('data', {}).get('data', 'Unknown error')}")
        return None


def fetch_prices(dhan, sec_id, exchange, expiry_date):
    # Fetch the latest prices for the given security ID and exchange
    print(f"Fetching price for sec_id: {sec_id}, exchange: {exchange}")
    
    # For BSE options, use BSE_FNO format
    if exchange == 'BSE':
        price_response = dhan.ticker_data({'BSE_FNO': [sec_id]})
        print(f"Price response for {exchange}: {price_response}")
        
        # Check if the API request was successful
        if price_response.get('status') == 'success':
            option_data = price_response.get('data', {}).get('data', {}).get('BSE_FNO', {}).get(str(sec_id), {})
            if option_data:
                return option_data
            else:
                print(f"No data found for security ID {sec_id}")
                return None
        else:
            print(f"API request failed: {price_response.get('remarks', 'Unknown error')}")
            return None
    else:
        print(f"Exchange {exchange} not implemented yet")
        return None