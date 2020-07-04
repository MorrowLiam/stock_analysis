# Stock Analysis

## Description:
Stock Analysis to evaluate my personal accounts.

## Index
1. stock_utilities
    - Stock Fetch from yahoo
    - Read From Excel
    - Write to Excel
    - Generate Typ. DataFrame for stock info
    - Scrape SP500 tickers from wikipedia

2. etrade_wrapper
    - Etrade_Connect
    - Accounts
      - account_list
      - fetch_portfolio
      - portfolio_dataframe
      - balance
      - list_transactions
    - Market
      - quotes
    -examples to use the wrapper

3. portfolio_optimization
    - efficient_frontier_models
      - pyfolio_eff_frontier
      - monte_carlo_eff_frontier
      - scipy_eff_frontier
    - plot all of the efficient frontier models




TODO:
- add exp covariance
- add cvxpy efficient scipy_eff_frontier
- complete decompose to prediction for stock_screener
- finalize stock_screener to incorporate other data
- Incorporate CAPM
- Develop comparison chart of allocations v. preferred allocation.
- create another file to run the analysis separately


## License
MIT License

Copyright (c) 2020 Liam Morrow

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
