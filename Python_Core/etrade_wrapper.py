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
class Etrade_Connect:
    def __init__(self):
        pass


    def oauth(self):
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
            acct_df = pd.DataFrame(accounts)
            return acct_df

        else:
            # Handle errors
            logger.debug("Response Body: %s", response.text)
            if response is not None and response.headers['Content-Type'] == 'application/json' \
                    and "Error" in response.json() and "message" in response.json()["Error"] \
                    and response.json()["Error"]["message"] is not None:
                print("Error: " + response.json()["Error"]["message"])
            else:
                print("Error: AccountList API service error")

    def fetch_portfolio(self,acct_id_key):
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

    def portfolio_dataframe(self,acct_id_keys,accounts):
        """Function used to parse data from API into a Pandas Dataframe format. 

        Args:
            acct_id_keys (str):Takes a list of account id from the account_list function to pull the positions.  

        Returns:
            [Dataframe]: Returns a filled dataframe with stock positions sorted by account. Postions have a number of data items (price paid, qty, etc.)
        """
        #Check for valid accounts. Append valid accounts to a list, disregard none valid accounts.
        valid_accounts = []
        for i in range(0,len(acct_id_keys)):
            try:
                selection = accounts.fetch_portfolio(acct_id_keys[i])[0]
                valid_accounts.append(selection)
                print('Valid Account')
            except TypeError:
                pass 

        #setup dictionary to incorporate data from accounts. Dictionary is needed to cycle through the accounts and positions. Each account is a seperate list inside of the dictionary.
        acc_df = {}
        #for loop to go through each account
        for i in range(0,len(valid_accounts)):
            #These are the datapoints we are collecting. Setup lists to hold the data. Range needs to change if more datapoints are added.
            account_id, position, positionId, adjPrevClose, pricePaid, costPerShare, totalCost, totalGain, totalGainPct, marketValue, quantity, positionType, daysGain, daysGainPct, pctOfPortfolio = ([] for i in range(15))

            #for loop to god through each position. 
            for pos in range(0,len(valid_accounts[i]['Position'])):
                account_id.append(valid_accounts[i]['accountId'])
                position.append(valid_accounts[i]['Position'][pos]['symbolDescription'])
                positionId.append(valid_accounts[i]['Position'][pos]['positionId'])
                adjPrevClose.append(valid_accounts[i]['Position'][pos]['adjPrevClose'])
                pricePaid.append(valid_accounts[i]['Position'][pos]['pricePaid'])
                costPerShare.append(valid_accounts[i]['Position'][pos]['costPerShare'])
                totalCost.append(valid_accounts[i]['Position'][pos]['totalCost'])
                totalGain.append(valid_accounts[i]['Position'][pos]['totalGain'])
                totalGainPct.append(valid_accounts[i]['Position'][pos]['totalGainPct'])
                marketValue.append(valid_accounts[i]['Position'][pos]['marketValue'])
                quantity.append(valid_accounts[i]['Position'][pos]['quantity'])
                positionType.append(valid_accounts[i]['Position'][pos]['positionType'])
                daysGain.append(valid_accounts[i]['Position'][pos]['daysGain'])
                daysGainPct.append(valid_accounts[i]['Position'][pos]['daysGainPct'])
                pctOfPortfolio.append(valid_accounts[i]['Position'][pos]['pctOfPortfolio'])
                positionType.append(valid_accounts[i]['Position'][pos]['positionType'])

            #Zip the data into a tuple to create the dataframe. The data and column variables are to make the script more readable.
            data=[]
            data = zip(account_id,position,positionId, adjPrevClose, pricePaid, costPerShare, totalCost, totalGain, totalGainPct, marketValue, quantity, positionType, daysGain, daysGainPct, pctOfPortfolio)
            columns = ('Account ID', 'Position', 'Position ID', 'ADJ Previous Close', 'Price Paid', 'Cost Per Share', 'Total Cost', 'Total Gain', 'Total Gain Pct', 'Market Value', 'Quantity', 'Position Type', 'Days Gain', 'Days Gain Pct', 'Pct of Portfolio')
            #create individual df for each account.
            acc_df[i] = pd.DataFrame(data, columns=(columns))

        #concat the individual df into one master df.
        overall_df = pd.concat(acc_df.values(), ignore_index=True)
        print('DataFrame Complete :)')
        return overall_df

    def balance(self,acct_df):
            """
            Calls account balance API to retrieve the current balance and related details for a specified account

            :param self: Pass in parameters authenticated session and information on selected account
            """

            results_df = {}
            #for loop to go through each account
            data=[]
            accountId , accountType, accountDescription, cashAvailableForInvestment, cashAvailableForWithdrawal, totalAvailableForWithdrawal, settledCashForInvestment, cashBuyingPower, totalAccountValue, netMv, netMvLong, netMvShort = ([] for i in range(12))
            for i in range(0,len(acct_df['accountIdKey'])):
                # URL for the API endpoint
                url = self.base_url + "/v1/accounts/" + acct_df['accountIdKey'][i] + "/balance.json"

                # Add parameters and header information
                params = {"instType": acct_df['institutionType'][i], "realTimeNAV": "true"}
                headers = {"consumerkey": config["DEFAULT"]["CONSUMER_KEY"]}

                # Make API call for GET request
                response = self.session.get(url, header_auth=True, params=params, headers=headers)
                logger.debug("Request url: %s", url)
                logger.debug("Request Header: %s", response.request.headers)
                
                # Handle and parse response
                if response is not None and response.status_code == 200:
                    parsed = json.loads(response.text)
                    logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
                    data = response.json()
                   
                    accountId.append(data['BalanceResponse']['accountId'])
                    accountType.append(data['BalanceResponse']['accountType'])
                    accountDescription.append(data['BalanceResponse']['accountDescription'])
                    cashAvailableForInvestment.append(data['BalanceResponse']['accountType'])
                    cashAvailableForWithdrawal.append(data['BalanceResponse']['Computed']['cashAvailableForWithdrawal'])
                    totalAvailableForWithdrawal.append(data['BalanceResponse']['Computed']['totalAvailableForWithdrawal'])
                    settledCashForInvestment.append(data['BalanceResponse']['Computed']['settledCashForInvestment'])
                    cashBuyingPower.append(data['BalanceResponse']['Computed']['cashBuyingPower'])
                    totalAccountValue.append(data['BalanceResponse']['Computed']['RealTimeValues']['totalAccountValue'])
                    netMv.append(data['BalanceResponse']['Computed']['RealTimeValues']['netMv'])
                    netMvLong.append(data['BalanceResponse']['Computed']['RealTimeValues']['netMvLong'])
                    netMvShort.append(data['BalanceResponse']['Computed']['RealTimeValues']['netMvShort'])

                else:
                    # Handle errors
                    logger.debug("Response Body: %s", response.text)
                    if response is not None and response.headers['Content-Type'] == 'application/json' \
                            and "Error" in response.json() and "message" in response.json()["Error"] \
                            and response.json()["Error"]["message"] is not None:
                        print("Error: " + response.json()["Error"]["message"])
                    else:
                        print("Error: Balance API service error")

                

            else:
                # Handle errors
                logger.debug("Response Body: %s", response.text)
                if response is not None and response.headers['Content-Type'] == 'application/json' \
                        and "Error" in response.json() and "message" in response.json()["Error"] \
                        and response.json()["Error"]["message"] is not None:
                    print("Error: " + response.json()["Error"]["message"])
                else:
                    print("Error: Balance API service error")

            results=[]
            columns=[]
            results = zip(accountId , accountType, accountDescription, cashAvailableForInvestment, cashAvailableForWithdrawal, totalAvailableForWithdrawal, settledCashForInvestment, cashBuyingPower, totalAccountValue, netMv, netMvLong, netMvShort)
            columns = ('Account ID' , 'Account Type', 'Account Description', 'Cash Availble For Investment', 'Cash Available For Withdraw', 'Total Available For Withdraw', 'Settled Cash For Investment', 'Cash BUying Power', 'Total Account Value', 'Net Market Value', 'Net Market Value Long', 'Net Market Value Short')
            #create individual df for each account.
            results_df[i] = pd.DataFrame(results, columns=(columns))

            #concat the individual df into one master df.
            overall_df = pd.concat(results_df.values(), ignore_index=True)
            print('DataFrame Below:')
            return overall_df
                        
    def list_transactions(self,acct_id):
        """
        Calls account balance API to retrieve the current balance and related details for a specified account

        :param self: Pass in parameters authenticated session and information on selected account
        """

        results_df = {}
        data=[]
        accountId,transactionId,transactionType,transactionDate,postDate,amount,description,displaySymbol,quantity,price,settlementCurrency,settlementDate = ([] for i in range(12))
        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/" + str(acct_id) + "/transactions.json"

        # Make API call for GET request
        response = self.session.get(url)
        logger.debug("Request url: %s", url)
        logger.debug("Request Header: %s", response.request.headers)
        
        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            for n in range(0,(len(data['TransactionListResponse']['Transaction']))):
                accountId.append(data['TransactionListResponse']['Transaction'][n]['accountId'])
                transactionId.append(data['TransactionListResponse']['Transaction'][n]['transactionId'])
                transactionType.append(data['TransactionListResponse']['Transaction'][n]['transactionType'])
                transactionDate.append(data['TransactionListResponse']['Transaction'][n]['transactionDate'])
                postDate.append(data['TransactionListResponse']['Transaction'][n]['postDate'])
                amount.append(data['TransactionListResponse']['Transaction'][n]['amount'])
                description.append(data['TransactionListResponse']['Transaction'][n]['description'])
                try:
                    displaySymbol.append(data['TransactionListResponse']['Transaction'][n]['brokerage']['product']['symbol'])
                except:
                    displaySymbol.append('NA')
                quantity.append(data['TransactionListResponse']['Transaction'][n]['brokerage']['quantity'])
                price.append(data['TransactionListResponse']['Transaction'][n]['brokerage']['price'])
                settlementCurrency.append(data['TransactionListResponse']['Transaction'][n]['brokerage']['settlementCurrency'])
                settlementDate.append(data['TransactionListResponse']['Transaction'][n]['brokerage']['settlementDate'])
           
        else:
            logger.debug("Response Body: %s", response)
            print("Error: Quote API service error")

        results=[]
        columns=[]
        results = zip(accountId,transactionId,displaySymbol,description,transactionType,settlementDate,transactionDate,postDate,amount,quantity,price,settlementCurrency)
        columns = ('Account ID','Transaction ID','Symbol','Description','Transaction Type','Transaction Date','Settlement Date','Post Date','Amount','Quantity','Price','Settlement Currency')
        #create individual df for each account.
        results_df = pd.DataFrame(results, columns=(columns))

        print('DataFrame Below:')
        return results_df


# %% Market
class Market:
    def __init__(self, session, base_url):
        self.session = session
        self.base_url = base_url

    def quotes(self,symbols):
        """
        Calls quotes API to provide quote details for equities, options, and mutual funds

        :param self: Passes authenticated session in parameter
        """
        results_df = {}
        data=[]
        dateTime,quoteStatus,securityType,symbol,companyName,ask,askSize,bid,bidSize,changeClose,changeClosePercentage,dividend,eps,estEarnings,exDividendDate,high,high52,lastTrade,low,low52,open_price,openInterest,previousClose,previousDayVolume,primaryExchange,totalVolume,marketCap,sharesOutstanding,nextEarningDate,beta,div_yield,declaredDividend,dividendPayableDate,pe,week52LowDate,week52HiDate,intrinsicValue= ([] for i in range(37))
        # URL for the API endpoint
        url = self.base_url + "/v1/market/quote/" + symbols + ".json"

        # Make API call for GET request
        response = self.session.get(url)
        logger.debug("Request Header: %s", response.request.headers)

        if response is not None and response.status_code == 200:

            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))

            # Handle and parse response
            data = response.json()
            #TODO there needs to be a cleaner way of writting this.
            dateTime.append(data['QuoteResponse']['QuoteData'][0]['dateTime'])
            quoteStatus.append(data['QuoteResponse']['QuoteData'][0]['quoteStatus'])
            securityType.append(data['QuoteResponse']['QuoteData'][0]['Product']['securityType'])
            symbol.append(data['QuoteResponse']['QuoteData'][0]['Product']['symbol'])
            companyName.append(data['QuoteResponse']['QuoteData'][0]['All']['companyName'])
            ask.append(data['QuoteResponse']['QuoteData'][0]['All']['ask'])
            askSize.append(data['QuoteResponse']['QuoteData'][0]['All']['askSize'])
            bid.append(data['QuoteResponse']['QuoteData'][0]['All']['bid'])
            bidSize.append(data['QuoteResponse']['QuoteData'][0]['All']['bidSize'])
            changeClose.append(data['QuoteResponse']['QuoteData'][0]['All']['changeClose'])
            changeClosePercentage.append(data['QuoteResponse']['QuoteData'][0]['All']['changeClosePercentage'])
            dividend.append(data['QuoteResponse']['QuoteData'][0]['All']['dividend'])
            eps.append(data['QuoteResponse']['QuoteData'][0]['All']['eps'])
            estEarnings.append(data['QuoteResponse']['QuoteData'][0]['All']['estEarnings'])
            exDividendDate.append(data['QuoteResponse']['QuoteData'][0]['All']['exDividendDate'])
            high.append(data['QuoteResponse']['QuoteData'][0]['All']['high'])
            high52.append(data['QuoteResponse']['QuoteData'][0]['All']['high52'])
            lastTrade.append(data['QuoteResponse']['QuoteData'][0]['All']['lastTrade'])
            low.append(data['QuoteResponse']['QuoteData'][0]['All']['low'])
            low52.append(data['QuoteResponse']['QuoteData'][0]['All']['low52'])
            open_price.append(data['QuoteResponse']['QuoteData'][0]['All']['open'])
            openInterest.append(data['QuoteResponse']['QuoteData'][0]['All']['openInterest'])
            previousClose.append(data['QuoteResponse']['QuoteData'][0]['All']['previousClose'])
            previousDayVolume.append(data['QuoteResponse']['QuoteData'][0]['All']['previousDayVolume'])
            primaryExchange.append(data['QuoteResponse']['QuoteData'][0]['All']['primaryExchange'])
            totalVolume.append(data['QuoteResponse']['QuoteData'][0]['All']['totalVolume'])
            marketCap.append(data['QuoteResponse']['QuoteData'][0]['All']['marketCap'])
            sharesOutstanding.append(data['QuoteResponse']['QuoteData'][0]['All']['sharesOutstanding'])
            nextEarningDate.append(data['QuoteResponse']['QuoteData'][0]['All']['nextEarningDate'])
            beta.append(data['QuoteResponse']['QuoteData'][0]['All']['beta'])
            div_yield.append(data['QuoteResponse']['QuoteData'][0]['All']['yield'])
            declaredDividend.append(data['QuoteResponse']['QuoteData'][0]['All']['declaredDividend'])
            dividendPayableDate.append(data['QuoteResponse']['QuoteData'][0]['All']['dividendPayableDate'])
            pe.append(data['QuoteResponse']['QuoteData'][0]['All']['pe'])
            week52LowDate.append(data['QuoteResponse']['QuoteData'][0]['All']['week52LowDate'])
            week52HiDate.append(data['QuoteResponse']['QuoteData'][0]['All']['week52HiDate'])
            intrinsicValue.append(data['QuoteResponse']['QuoteData'][0]['All']['intrinsicValue'])

            

        else:
            logger.debug("Response Body: %s", response)
            print("Error: Quote API service error")

        results=[]
        columns=[]
        results = zip(dateTime,quoteStatus,securityType,symbol,companyName,ask,askSize,bid,bidSize,changeClose,changeClosePercentage,dividend,eps,estEarnings,exDividendDate,high,high52,lastTrade,low,low52,open_price,openInterest,previousClose,previousDayVolume,primaryExchange,totalVolume,marketCap,sharesOutstanding,nextEarningDate,beta,div_yield,declaredDividend,dividendPayableDate,pe,week52LowDate,week52HiDate,intrinsicValue)
        columns = ('dateTime','quoteStatus','securityType','symbol','companyName','ask','askSize','bid','bidSize','changeClose','changeClosePercentage','dividend','eps','estEarnings','exDividendDate','high','high52','lastTrade','low','low52','open_price','openInterest','previousClose','previousDayVolume','primaryExchange','totalVolume','marketCap','sharesOutstanding','nextEarningDate','beta','div_yield','declaredDividend','dividendPayableDate','pe','week52LowDate','week52HiDate','intrinsicValue')

        #create individual df for each account.
        results_df = pd.DataFrame(results, columns=(columns))

        print('DataFrame Below:')
        return results_df


# %% Example Cell 
# # %% Oauth
# #Oauth set the session and base url to use in the rest of the calls.
# e_c = Etrade_Connect()
# ouath_method=e_c.oauth()
# session = ouath_method[0]
# base_url = ouath_method[1]

# # %% Account funtions
# #set account class with session and base url
# accounts = Accounts(session,base_url)

# #pull account list to use for other functions
# acct_df= accounts.account_list()

# #seperate just the id keys
# acct_id_keys = acct_df['accountIdKey']

# #sample of the portfolio,balance, and transaction function.
# accounts.portfolio_dataframe(acct_id_keys)
# accounts.balance(acct_df)
# accounts.list_transactions(acct_id_keys[0])
# # %% Market Functions
# #set the session and base for the market class
# mkt = Market(session,base_url)
# #fetch a df with stock info and transpose the results for easy reading.
# mkt.quotes('HP').T
