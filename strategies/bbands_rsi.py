import ta
import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility
from strategies.strategy import Strategy


class BBRSI(Strategy):
    def __init__(self, maximum_window = 300):
        self.name = "BBRSI"
        self.version = "v1.0"

        Strategy.__init__(self,
                          maximum_window=maximum_window)
    
        self.indicators = dict({
            "rsi": lambda o=None, h=None, l=None, c=None: ta.momentum.rsi(close = c, window = 13),
            "hband": lambda o=None, h=None, l=None, c=None: ta.volatility.bollinger_hband(close = c, window = 30),
            "mband": lambda o=None, h=None, l=None, c=None: ta.volatility.bollinger_mavg(close = c, window = 30),
            "lband": lambda o=None, h=None, l=None, c=None: ta.volatility.bollinger_lband(close = c, window = 30),
            "ema": lambda o=None, h=None, l=None, c=None: ta.momentum._ema(series = c, periods = 200),
        })


    def next(self, current_data):
        current_data = current_data.copy()
        if len(current_data) < self.maximum_window:
            return -1
        
        data: pd.DataFrame = self.calculate_indicators(current_data)
        data_lookback = data.iloc[-3:]

        if (data_lookback["ema"] < data_lookback["hband"]).any() and (data_lookback["ema"] > data_lookback["lband"]).any():
            return -1
        if (data_lookback["rsi"] >= 70).any() and (data_lookback["Close"] > data_lookback["hband"]).any():
            return 0
        if (data_lookback["rsi"] <= 30).any() and (data_lookback["Close"] < data_lookback["lband"]).any():
            return 1
        return -1