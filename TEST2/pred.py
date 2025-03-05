from stock_analysis import StockReader

reader = StockReader('2019-01-01', '2023-08-27')

# get bitcoin data in USD
bitcoin = reader.get_bitcoin_data('USD')

# get faang data
fb, aapl, amzn, nflx, goog = (
    reader.get_ticker_data(ticker)
    for ticker in ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOG']
)

# get S&P 500 data
sp = reader.get_index_data('S&P 500')

from stock_analysis.utils import group_stocks, describe_group

faang = group_stocks(
    {
        'Facebook': fb,
        'Apple': aapl,
        'Amazon': amzn,
        'Netflix': nflx,
        'Google': goog
    }
)

# describe the group
describe_group(faang)

from stock_analysis.utils import make_portfolio

faang_portfolio = make_portfolio(faang)

import matplotlib.pyplot as plt
from stock_analysis import StockVisualizer


from stock_analysis import AssetGroupVisualizer

faang_viz = AssetGroupVisualizer(faang)
faang_viz.heatmap(True)

#plt.show()

from stock_analysis import StockModeler
arima_model = StockModeler.arima(nflx, ar=10, i=1, ma=5)
StockModeler.plot_residuals(arima_model)
plt.show()

arima_ax = StockModeler.arima_predictions(
    nflx, arima_model,
    start='2023-01-07', end='2023-08-04',
    title='ARIMA'
)
plt.show()
