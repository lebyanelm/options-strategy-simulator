import ta
import pandas as pd
import numpy as np
import ta.momentum
import ta.trend
import ta.volatility
from strategies.strategy import Strategy
import pickle
import sklearn.svm
import sklearn.preprocessing


class SVCAI(Strategy):
    BACKCANDLES = 15
    CURRENT_DATA = None

    model: sklearn.svm.SVC = None
    scaler: sklearn.preprocessing.StandardScaler = None

    def __init__(self, maximum_window = 20):
        self.name = "SVCAI"
        self.version = "v1.0"

        Strategy.__init__(self,
                          maximum_window=maximum_window)

        self.indicators = dict(
            Bollup = lambda o, h, l, c: ta.volatility.bollinger_hband_indicator(close=c, window=30),
            Bolldown = lambda o, h, l, c: ta.volatility.bollinger_lband_indicator(close=c, window=30),
            Ema200 = lambda o, h, l, c: ta.momentum._ema(series = c, periods = 200),
            RSI = lambda o, h, l, c: ta.momentum.rsi(close=c, window=13),
            ATR = lambda o, h, l, c: ta.volatility.average_true_range(high=h, low=l, close=c, window=14),
            Support = lambda o, h, l, c: l.rolling(window=self.BACKCANDLES).min(),
            Resistance = lambda o, h, l, c: h.rolling(window=self.BACKCANDLES).min()
        )

        """Load a saved classifier"""
        with open(f"./models/EURUSD_1.model", "rb") as model_file:
            self.model = pickle.load(model_file)

        """Load a saved scaler"""
        with open(f"./models/EURUSD_1.scaler", "rb") as scaler_file:
            self.scaler = pickle.load(scaler_file)


    def next(self, current_data):
        self.CURRENT_DATA = current_data.copy()
        if self.CURRENT_DATA.shape[0] < 200:
            return -1
        
        self.CURRENT_DATA = self.calculate_indicators(self.CURRENT_DATA)
        self.CURRENT_DATA.dropna(inplace=True)

        if len(self.CURRENT_DATA) < self.maximum_window:
            return -1
        
        if self.CURRENT_DATA.iloc[-1]["Ema200"] <= self.CURRENT_DATA.iloc[-1]["Bollup"] and self.CURRENT_DATA.iloc[-1]["Ema200"] >= self.CURRENT_DATA.iloc[-1]["Bolldown"]:
            return -1

        input = self.prepare_input()
        if input is None:
            return -1
        else:
            scaled_input = self.scaler.transform([input])
            prediction = self.model.predict(scaled_input).flatten()[0]
            return int(prediction)


    def prepare_input(self):
        processed_inputs = []
        for i in range(0, len(self.CURRENT_DATA), self.BACKCANDLES):
            j = i + self.BACKCANDLES
            group = self.CURRENT_DATA.iloc[i:j][["Support", "Resistance", "Bollup", "Bolldown", "ATR", "Ema200", "RSI"]]
            if len(group) != self.BACKCANDLES:
                break
            else:
                processed_inputs.append(group.to_numpy())
        processed_inputs = np.array(processed_inputs)
        num_samples = processed_inputs.shape[0]
        num_samples_n = processed_inputs.shape[1:]
        height, width = num_samples_n if len(num_samples_n) == 2 else (None, None)
        if height is None and width is None:
            return None
        else:
            return processed_inputs.reshape(num_samples, height * width)[-1]