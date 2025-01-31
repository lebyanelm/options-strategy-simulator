import pandas as pd


class Strategy:
    indicators = []
    def __init__(self, maximum_window: int = 300):
        # Defines tick data to be used by the main frame or the backtester.
        # Thus leaves room for the indicators to be calculated without NA values
        # at the last value of the dataframe.
        self.maximum_window = maximum_window

        if not self.name and not self.version:
            self.name = "Strategy Base"
            self.version = "1.0.0"
        self.name = f"[{self.name} - {self.version}]"

        # Ghost indicators as a placeholder for testing empty strategies.
        self.indicators = list()


    def __str__(self):
        return self.name


    def calculate_indicators(self, current_data: pd.DataFrame):
        current_data = current_data.copy()
        open = current_data["Open"]
        high = current_data["High"]
        low = current_data["Low"]
        close = current_data["Close"]

        for indicator in self.indicators:
            current_data.loc[:, indicator] = self.indicators[indicator](o=open, h=high, l=low, c=close)
        return current_data


    def next(self, current_data: pd.DataFrame):
        return -1