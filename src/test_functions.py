from dhanhq import dhanhq
import credentials

def test_authentication(client_id, access_token):
    # Test authentication with valid credentials
    dhan = dhanhq(client_id= client_id, access_token= access_token)
    print(dhan.get_fund_limits())

test_authentication(credentials.DHAN_CLIENT_ID, credentials.DHAN_ACCESS_TOKEN)