# %% imports
import webbrowser
import pandas as pd
from pandas import Series, DataFrame
import logging
from requests_oauthlib import OAuth1Session
import jxmlease

from pandas.io.json import read_json

# %% Etrade oauth class
# Set up logging
LOGGER = logging.getLogger(__name__)


class ETradeOAuth(object):
    """ETradeOAuth
       ETrade OAuth 1.0a Wrapper"""

    def __init__(self, consumer_key, consumer_secret, callback_url="oob"):
        """__init__(consumer_key, consumer_secret, callback_url)
           param: consumer_key
           type: str
           description: etrade oauth consumer key
           param: consumer_secret
           type: str
           description: etrade oauth consumer secret
           param: callback_url
           type: str
           description: etrade oauth callback url default oob"""

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.base_url_prod = r"https://api.etrade.com"
        self.base_url_dev = r"https://apisb.etrade.com"
        self.req_token_url = r"https://api.etrade.com/oauth/request_token"
        self.auth_token_url = r"https://us.etrade.com/e/t/etws/authorize"
        self.access_token_url = r"https://api.etrade.com/oauth/access_token"
        self.callback_url = callback_url
        self.access_token = None
        self.resource_owner_key = None

    def get_request_token(self):
        """get_request_token() -> auth url
           some params handled by requests_oauthlib but put in
           doc string for clarity into the API.
           param: oauth_consumer_key
           type: str
           description: the value used by the consumer to identify
                        itself to the service provider.
           param: oauth_timestamp
           type: int
           description: the date and time of the request, in epoch time.
                        must be accurate within five minutes.
           param: oauth_nonce
           type: str
           description: a nonce, as discribed in the authorization guide
                        roughly, an arbitrary or random value that cannot
                        be used again with the same timestamp.
           param: oauth_signature_method
           type: str
           description: the signature method used by the consumer to sign
                        the request. the only supported value is 'HMAC-SHA1'.
           param: oauth_signature
           type: str
           description: signature generated with the shared secret and token
                        secret using the specified oauth_signature_method
                        as described in OAuth documentation.
           param: oauth_callback
           type: str
           description: callback information, as described elsewhere. must
                        always be set to 'oob' whether using a callback or
                        not
           rtype: str
           description: Etrade autherization url"""

        # Set up session
        self.session = OAuth1Session(
            self.consumer_key,
            self.consumer_secret,
            callback_uri=self.callback_url,
            signature_type="AUTH_HEADER",
        )
        # get request token
        self.session.fetch_request_token(self.req_token_url)
        # get authorization url
        # etrade format: url?key&token
        authorization_url = self.session.authorization_url(self.auth_token_url)
        akey = self.session.parse_authorization_response(authorization_url)
        # store oauth_token
        self.resource_owner_key = akey["oauth_token"]
        formated_auth_url = "%s?key=%s&token=%s" % (
            self.auth_token_url,
            self.consumer_key,
            akey["oauth_token"],
        )
        self.verifier_url = formated_auth_url
        LOGGER.debug(formated_auth_url)

        return formated_auth_url

    def get_access_token(self, verifier):
        """get_access_token(verifier) -> access_token
           param: verifier
           type: str
           description: oauth verification code
           rtype: dict
           description: oauth access tokens
           OAuth API paramiters mostly handled by requests_oauthlib
           but illistrated here for clarity.
           param: oauth_consumer_key
           type: str
           description: the value used by the consumer to identify
                        itself to the service provider.
           param: oauth_timestamp
           type: int
           description: the date and time of the request, in epoch time.
                        must be accurate within five minutes.
           param: oauth_nonce
           type: str
           description: a nonce, as discribed in the authorization guide
                        roughly, an arbitrary or random value that cannot
                        be used again with the same timestamp.
           param: oauth_signature_method
           type: str
           description: the signature method used by the consumer to sign
                        the request. the only supported value is 'HMAC-SHA1'.
           param: oauth_signature
           type: str
           description: signature generated with the shared secret and token
                        secret using the specified oauth_signature_method
                        as described in OAuth documentation.
           param: oauth_token
           type: str
           description: the consumer's request token to be exchanged for an
                        access token
           param: oauth_verifier
           type: str
           description: the code received by the user to authenticate with
                        the third-party application"""

        # Set verifier
        self.session._client.client.verifier = verifier
        # Get access token
        self.access_token = self.session.fetch_access_token(
            self.access_token_url)
        LOGGER.debug(self.access_token)

        return self.access_token


class ETradeAccessManager(object):
    """ETradeAccessManager - Renew and revoke ETrade OAuth Access Tokens"""

    def __init__(self, client_key, client_secret, resource_owner_key, resource_owner_secret):
        """__init__(client_key, client_secret)
           param: client_key
           type: str
           description: etrade client key
           param: client_secret
           type: str
           description: etrade client secret
           param: resource_owner_key
           type: str
           description: OAuth authentication token key
           param: resource_owner_secret
           type: str
           description: OAuth authentication token secret"""
        self.client_key = client_key
        self.client_secret = client_secret
        self.resource_owner_key = resource_owner_key
        self.resource_owner_secret = resource_owner_secret
        self.renew_access_token_url = r"https://api.etrade.com/oauth/renew_access_token"
        self.revoke_access_token_url = (
            r"https://api.etrade.com/oauth/revoke_access_token"
        )
        self.session = OAuth1Session(
            self.client_key,
            self.client_secret,
            self.resource_owner_key,
            self.resource_owner_secret,
            signature_type="AUTH_HEADER",
        )

    def renew_access_token(self):
        """renew_access_token() -> bool
           some params handled by requests_oauthlib but put in
           doc string for clarity into the API.
           param: oauth_consumer_key
           type: string
           required: true
           description: the value used by the consumer to identify
                        itself to the service provider.
           param: oauth_timestamp
           type: int
           required: true
           description: the date and time of the request, in epoch time.
                        must be accurate withiin five minutes.
           param: oauth_nonce
           type: str
           required: true
           description: a nonce, as described in the authorization guide
                        roughly, an arbitrary or random value that cannot
                        be used again with the same timestamp.
           param: oauth_signature_method
           type: str
           required: true
           description: the signature method used by the consumer to sign
                        the request. the only supported value is "HMAC-SHA1".
           param: oauth_signature
           type: str
           required: true
           description: signature generated with the shared secret and
                        token secret using the specified oauth_signature_method
                        as described in OAuth documentation.
           param: oauth_token
           type: str
           required: true
           description: the consumer's access token to be renewed."""
        resp = self.session.get(self.renew_access_token_url)
        LOGGER.debug(resp.text)
        resp.raise_for_status()

        return True

    def revoke_access_token(self):
        """revoke_access_token() -> bool
           some params handled by requests_oauthlib but put in
           doc string for clarity into the API.
           param: oauth_consumer_key
           type: string
           required: true
           description: the value used by the consumer to identify
                        itself to the service provider.
           param: oauth_timestamp
           type: int
           required: true
           description: the date and time of the request, in epoch time.
                        must be accurate withiin five minutes.
           param: oauth_nonce
           type: str
           required: true
           description: a nonce, as described in the authorization guide
                        roughly, an arbitrary or random value that cannot
                        be used again with the same timestamp.
           param: oauth_signature_method
           type: str
           required: true
           description: the signature method used by the consumer to sign
                        the request. the only supported value is "HMAC-SHA1".
           param: oauth_signature
           type: str
           required: true
           description: signature generated with the shared secret and
                        token secret using the specified oauth_signature_method
                        as described in OAuth documentation.
           param: oauth_token
           type: str
           required: true
           description: the consumer's access token to be revoked."""
        resp = self.session.get(self.revoke_access_token_url)
        LOGGER.debug(resp.text)
        resp.raise_for_status()

        return True


class ETradeAccounts(object):
    """ETradeAccounts:"""

    def __init__(self, client_key, client_secret, resource_owner_key, resource_owner_secret):
        """__init_()"""
        self.client_key = client_key
        self.client_secret = client_secret
        self.resource_owner_key = resource_owner_key
        self.resource_owner_secret = resource_owner_secret
        self.base_url_prod = r"https://api.etrade.com/v1/accounts"
        self.base_url_dev = r"https://apisb.etrade.com/v1/accounts"
        self.session = OAuth1Session(
            self.client_key,
            self.client_secret,
            self.resource_owner_key,
            self.resource_owner_secret,
            signature_type="AUTH_HEADER",
        )

    def list_accounts(self, dev=True, resp_format="json"):
        """list_account(dev, resp_format)
           param: dev
           type: bool
           description: API enviornment
           param: resp_format
           type: str
           description: Response format
           rformat: json
           rtype: dict
           rformat: other than json
           rtype: str"""

        if dev:
            if resp_format == "json":
                uri = r"list"
                api_url = "%s/%s.%s" % (self.base_url_dev, uri, resp_format)
            elif resp_format == "xml":
                uri = r"list"
                api_url = "%s/%s" % (self.base_url_dev, uri)
        else:
            if resp_format == "json":
                uri = r"list"
                api_url = "%s/%s.%s" % (self.base_url_prod, uri, resp_format)
            elif resp_format == "xml":
                uri = r"list"
                api_url = "%s/%s" % (self.base_url_prod, uri)

        LOGGER.debug(api_url)
        req = self.session.get(api_url)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        else:
            return jxmlease.parse(req.text)

    def get_account_balance(self, account_id, account_type=None, real_time=True, dev=True, resp_format="json",):
        """get_account_balance(dev, resp_format)
        param: account_id
        type: int
        required: true
        description: Numeric account id
        param: dev
        type: bool
        description: API enviornment
        param: resp_format
        type: str
        description: Response format
        rformat: json
        rtype: dict
        rformat: other than json
        rtype: str"""

        uri = "balance"
        payload = {"realTimeNAV": real_time, "instType": "BROKERAGE"}
        if account_type:
            payload["accountType"] = account_type

        if dev:
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_dev,
                    account_id,
                    uri,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_dev, account_id, uri)
        else:
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_prod,
                    account_id,
                    uri,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_prod, account_id, uri)
        LOGGER.debug(api_url)
        req = self.session.get(api_url, params=payload)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        else:
            return jxmlease.parse(req.text)

    def get_account_positions(self, account_id, dev=True, resp_format="json"):
        """get_account_positions(dev, account_id, resp_format) -> resp
        param: account_id
        type: string
        required: true
        description: account id key
        param: dev
        type: bool
        description: API enviornment
        param: resp_format
        type: str
        description: Response format
        rformat: json
        rtype: dict
        rformat: other than json
        rtype: str"""

        if dev:
            api_url = self.base_url_dev
        else:
            api_url = self.base_url_prod

        api_url += "/" + account_id + "/portfolio"
        if resp_format == "json":
            api_url += ".json"

        LOGGER.debug(api_url)
        req = self.session.get(api_url)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return jxmlease.parse(req.text)

    def list_alerts(self, dev=True, resp_format="json"):
        """list_alerts(dev, resp_format) -> resp
        param: dev
        type: bool
        description: API enviornment
        param: resp_format
        type: str
        description: Response format
        rformat: json
        rtype: dict
        rformat: other than json
        rtype: str"""

        if dev:
            uri = r"accounts/sandbox/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s.%s" % (self.base_url_dev, uri, resp_format)
            elif resp_format == "xml":
                api_url = "%s/%s" % (self.base_url_dev, uri)

        else:
            uri = r"accounts/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s.%s" % (self.base_url_prod, uri, resp_format)
            elif resp_format == "xml":
                api_url = "%s/%s" % (self.base_url_prod, uri)

        LOGGER.debug(api_url)
        req = self.session.get(api_url)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return req.text

    def read_alert(self, alert_id, dev=True, resp_format="json"):
        """read_alert(alert_id, dev, resp_format) -> resp
        param: alert_id
        type: int
        description: Numaric alert ID
        param: dev
        type: bool
        description: API enviornment
        param: resp_format
        type: str
        description: Response format
        rformat: json
        rtype: dict
        rformat: other than json
        rtype: str"""

        if dev:
            uri = r"accounts/sandbox/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_dev,
                    uri,
                    alert_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_dev, uri, alert_id)

        else:
            uri = r"accounts/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_prod,
                    uri,
                    alert_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_prod, uri, alert_id)

        LOGGER.debug(api_url)
        req = self.session.get(api_url)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return req.text

    def delete_alert(self, alert_id, dev=True, resp_format="json"):
        """delete_alert(alert_id, dev, resp_format) -> resp
        param: alert_id
        type: int
        description: Numaric alert ID
        param: dev
        type: bool
        description: API enviornment
        param: resp_format
        type: str
        description: Response format
        rformat: json
        rtype: dict
        rformat: other than json
        rtype: str"""

        if dev:
            uri = r"accounts/sandbox/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_dev,
                    uri,
                    alert_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_dev, uri, alert_id)

        else:
            uri = r"accounts/rest/alerts"
            if resp_format == "json":
                api_url = "%s/%s/%s.%s" % (
                    self.base_url_prod,
                    uri,
                    alert_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s" % (self.base_url_prod, uri, alert_id)

        LOGGER.debug(api_url)
        req = self.session.delete(api_url)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return req.text

    def get_transaction_history(self, account_id, dev=True, group="ALL", asset_type="ALL", transaction_type="ALL", ticker_symbol="ALL", resp_format="json", **kwargs):
        """get_transaction_history(account_id, dev, resp_format) -> resp
        param: account_id
        type: int
        required: true
        description: Numeric account ID
        param: group
        type: string
        default: 'ALL'
        description: Possible values are: DEPOSITS, WITHDRAWALS, TRADES.
        param: asset_type
        type: string
        default: 'ALL'
        description: Only allowed if group is TRADES. Possible values are:
                EQ (equities), OPTN (options), MMF (money market funds),
                MF (mutual funds), BOND (bonds). To retrieve all types,
                use ALL or omit this parameter.
        param: transaction_type
        type: string
        default: 'ALL'
        description: Transaction type(s) to include, e.g., check, deposit,
                fee, dividend, etc. A list of types is provided in documentation
        param: ticker_symbol
        type: string
        default: 'ALL'
        description: Only allowed if group is TRADES. A single market symbol,
                e.g., GOOG.
        param: marker
        type: str
        description: Specify the desired starting point of the set
                of items to return. Used for paging.
        param: count
        type: int
        description: The number of orders to return in a response.
                The default is 25. Used for paging.
        description: see ETrade API docs"""

        # add each optional argument not equal to 'ALL' to the uri
        optional_args = [group, asset_type, transaction_type, ticker_symbol]
        optional_uri = ""
        for optional_arg in optional_args:
            if optional_arg.upper() != "ALL":
                optional_uri = "%s/%s" % (optional_uri, optional_arg)
        # Set Env
        if dev:
            # assemble the following:
            # self.base_url_dev: https://etws.etrade.com
            # uri:               /accounts/rest
            # account_id:        /{accountId}
            # format string:     /transactions
            # if not 'ALL' args:
            #   group:              /{Group}
            #   asset_type          /{AssetType}
            #   transaction_type:   /{TransactionType}
            #   ticker_symbol:      /{TickerSymbol}
            # resp_format:       {.json}
            # payload:           kwargs
            #
            uri = r"accounts/sandbox/rest"
            if resp_format == "json":
                api_url = "%s/%s/%s/transactions%s.%s" % (
                    self.base_url_dev,
                    uri,
                    account_id,
                    optional_uri,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s/transactions%s" % (
                    self.base_url_dev,
                    uri,
                    account_id,
                    optional_uri,
                )
        else:
            uri = r"accounts/rest"
            if resp_format == "json":
                api_url = "%s/%s/%s/transactions%s.%s" % (
                    self.base_url_prod,
                    uri,
                    account_id,
                    optional_uri,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s/transactions%s" % (
                    self.base_url_prod,
                    uri,
                    account_id,
                    optional_uri,
                )

        # Build Payload
        payload = kwargs
        LOGGER.debug("payload: %s", payload)

        LOGGER.debug(api_url)
        req = self.session.get(api_url, params=payload)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return req.text

    def get_transaction_details(self, account_id, transaction_id, dev=True, resp_format="json", **kwargs):
        """get_transaction_details(account_id, transaction_id, dev, resp_format) -> resp
        param: account_id
        type: int
        required: true
        description: Numeric account ID
        param: transaction_id
        type: int
        required: true
        description: Numeric transaction ID"""

        # Set Env
        if dev:
            uri = r"accounts/sandbox/rest"
            if resp_format == "json":
                api_url = "%s/%s/%s/transactions/%s.%s" % (
                    self.base_url_dev,
                    uri,
                    account_id,
                    transaction_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s/transactions/%s" % (
                    self.base_url_dev,
                    uri,
                    account_id,
                    transaction_id,
                )
        else:
            uri = r"accounts/rest"
            if resp_format == "json":
                api_url = "%s/%s/%s/transactions/%s.%s" % (
                    self.base_url_prod,
                    uri,
                    account_id,
                    transaction_id,
                    resp_format,
                )
            elif resp_format == "xml":
                api_url = "%s/%s/%s/transactions/%s" % (
                    self.base_url_prod,
                    uri,
                    account_id,
                    transaction_id,
                )

        # Build Payload
        payload = kwargs
        LOGGER.debug("payload: %s", payload)

        LOGGER.debug(api_url)
        req = self.session.get(api_url, params=payload)
        req.raise_for_status()
        LOGGER.debug(req.text)

        if resp_format == "json":
            return req.json()
        return req.text


# %% etrade activate oauth
consumer_key = "22a2edd63c0d6f8164abaeb20b027cb5"
consumer_secret = "f4b49e7a9c95ac97b3e8925228739e691d84568fe8f952596167dcbbbfb81385"

oauth = ETradeOAuth(consumer_key, consumer_secret)
print(oauth.get_request_token())  # Use the printed URL
webbrowser.open_new_tab(oauth.get_request_token())

# %% etrade oauth
verifier_code = input("Enter verification code: ")
tokens = oauth.get_access_token(verifier_code)
print(tokens)

tokens = tokens

# %% etrade oauth

accounts = ETradeAccounts(
    consumer_key,
    consumer_secret,
    tokens['oauth_token'],
    tokens['oauth_token_secret']
)
# %%
# account id really means id key
acct = accounts.list_accounts(dev=True, resp_format="json")
acct_df = pd.DataFrame(acct['AccountListResponse']['Accounts']['Account'])
acct_id_keys = acct_df['accountIdKey']
acct

# %%


# %% get dataframe of accounts
acct_lists = []
for i in acct_id_keys:
    print(i)

    acct_values = accounts.get_account_balance(i)['BalanceResponse']

    acct_dictionary = {'Account_ID': [acct_values['accountId']],'Open Order Cash Funds': [acct_values['Cash']['fundsForOpenOrdersCash']],'Money Market Balance': [acct_values['Cash']['moneyMktBalance']]}

    acct_lists.append(acct_dictionary)
    
acct_bal_df = pd.DataFrame(acct_lists)
acct_bal_df

# %%

#TODO use the df language above to start pulling the needed information.

