# %% imports
from __future__ import print_function
import webbrowser
import json
import logging
import configparser
import sys
import requests
from rauth import OAuth1Service
from logging.handlers import RotatingFileHandler
import pandas as pd
config = configparser.ConfigParser()
config_path=r"C:\Users\Liam Morrow\Documents\Onedrive\Python scripts\02 Snippets\config.ini"
config.read(config_path)
CONSUMER_KEY = config.get('DEFAULT', 'CONSUMER_KEY')
CONSUMER_SECRET = config.get('DEFAULT', 'CONSUMER_SECRET')

# logger settings
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("python_client.log", maxBytes=5 * 1024 * 1024, backupCount=3)
FORMAT = "%(asctime)-15s %(message)s"
fmt = logging.Formatter(FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)

# %% Oauth
def oauth():
    """Allows user authorization for the sample application with OAuth 1"""
    etrade = OAuth1Service(
        name="etrade",
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        request_token_url="https://api.etrade.com/oauth/request_token",
        access_token_url="https://api.etrade.com/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url="https://api.etrade.com")

    # Step 1: Get OAuth 1 request token and secret
    request_token, request_token_secret = etrade.get_request_token(
        params={"oauth_callback": "oob", "format": "json"})

    # Step 2: Go through the authentication flow. Login to E*TRADE.
    # After you login, the page will provide a text code to enter.
    authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)
    webbrowser.open(authorize_url)
    text_code = input("Please accept agreement and enter text code from browser: ")

    # Step 3: Exchange the authorized request token for an authenticated OAuth 1 session
    session = etrade.get_auth_session(request_token,
                                  request_token_secret,
                                  params={"oauth_verifier": text_code})
    base_url = "https://api.etrade.com"

    return (session, base_url)

# %% Accounts
class Accounts:
    def __init__(self, session, base_url):
        """
        Initialize Accounts object with session and account information

        :param session: authenticated session
        """
        self.session = session
        self.account = {}
        self.base_url = base_url

    def account_list(self):
        """
        Calls account list API to retrieve a list of the user's E*TRADE accounts

        :param self:Passes in parameter authenticated session
        """
        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/list.json"

        # Make API call for GET request
        response = self.session.get(url, header_auth=True)
        logger.debug("Request Header: %s", response.request.headers)
        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            accounts = data["AccountListResponse"]["Accounts"]["Account"]
            return accounts

        else:
            # Handle errors
            logger.debug("Response Body: %s", response.text)
            if response is not None and response.headers['Content-Type'] == 'application/json' \
                    and "Error" in response.json() and "message" in response.json()["Error"] \
                    and response.json()["Error"]["message"] is not None:
                print("Error: " + response.json()["Error"]["message"])
            else:
                print("Error: AccountList API service error")

    def portfolio(self,acct_id_key):
        """
        Call portfolio API to retrieve a list of positions held in the specified account

        :param self: Passes in parameter authenticated session and information on selected account
        """

        # URL for the API endpoint
        url =self.base_url + "/v1/accounts/" + acct_id_key + "/portfolio.json"

        # Make API call for GET request
        response = self.session.get(url, header_auth=True)
        logger.debug("Request Header: %s", response.request.headers)

        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            portfolio = data["PortfolioResponse"]["AccountPortfolio"]
            return portfolio


        #     if data is not None and "PortfolioResponse" in data and "AccountPortfolio" in data["PortfolioResponse"]:
        #         # Display balance information
        #         for acctPortfolio in data["PortfolioResponse"]["AccountPortfolio"]:
        #             if acctPortfolio is not None and "Position" in acctPortfolio:
        #                 for position in acctPortfolio["Position"]:
        #                     print_str = ""
        #                     if position is not None and "symbolDescription" in position:
        #                         print_str = print_str + "Symbol: " + str(position["symbolDescription"])
        #                     if position is not None and "quantity" in position:
        #                         print_str = print_str + " | " + "Quantity #: " + str(position["quantity"])
        #                     if position is not None and "Quick" in position and "lastTrade" in position["Quick"]:
        #                         print_str = print_str + " | " + "Last Price: " \
        #                                     + str('${:,.2f}'.format(position["Quick"]["lastTrade"]))
        #                     if position is not None and "pricePaid" in position:
        #                         print_str = print_str + " | " + "Price Paid: " \
        #                                     + str('${:,.2f}'.format(position["pricePaid"]))
        #                     if position is not None and "totalGain" in position:
        #                         print_str = print_str + " | " + "Total Gain: " \
        #                                     + str('${:,.2f}'.format(position["totalGain"]))
        #                     if position is not None and "marketValue" in position:
        #                         print_str = print_str + " | " + "Value: " \
        #                                     + str('${:,.2f}'.format(position["marketValue"]))
        #                     print(print_str)
        #             else:
        #                 print("None")
        #     else:
        #         # Handle errors
        #         logger.debug("Response Body: %s", response.text)
        #         if response is not None and "headers" in response and "Content-Type" in response.headers \
        #                 and response.headers['Content-Type'] == 'application/json' \
        #                 and "Error" in response.json() and "message" in response.json()["Error"] \
        #                 and response.json()["Error"]["message"] is not None:
        #             print("Error: " + response.json()["Error"]["message"])
        #         else:
        #             print("Error: Portfolio API service error")
        # elif response is not None and response.status_code == 204:
        #     print("None")

        else:
            # Handle errors
            logger.debug("Response Body: %s", response.text)
            if response is not None and "headers" in response and "Content-Type" in response.headers \
                    and response.headers['Content-Type'] == 'application/json' \
                    and "Error" in response.json() and "message" in response.json()["Error"] \
                    and response.json()["Error"]["message"] is not None:
                print("Error: " + response.json()["Error"]["message"])
            else:
                print("Error: Portfolio API service error")




# %% Oauth
ouath_method=oauth()
session = ouath_method[0]
base_url = ouath_method[1]

# %% Pull Accounts
accounts = Accounts(session,base_url)
acct = accounts.account_list()
acct_df = pd.DataFrame(acct)
acct_id_keys = acct_df['accountIdKey']
acct_id_keys


# %% check for valid accounts
#check for valid accounts
valid_accounts = []
for i in range(0,len(acct_id_keys)):
    try:
        selection = accounts.portfolio(acct_id_keys[i])[0]
        valid_accounts.append(selection)
        print("Valid")
        print(selection)
    except TypeError:
        pass 


# %% 




# %% 

valid_accounts[0]

# %% Pull Accounts
#positions
accounts.portfolio(acct_id_keys[0])[0]['Position'][0]['positionId']


# %% Pull Accounts
pd.DataFrame(valid_accounts)

acct_id_keys
acct_df


# %%
