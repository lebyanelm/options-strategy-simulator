

""" Imports """
import logging
import datetime
import time
import os
import colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf
from strategies.strategy import Strategy
from order import Order


class Backtest:
    """
    Backtest.py class alternative for binary options.
    Simulates the environment of IQ Options.
    """
    __default_expiration__ = 15
    __default_interval__ = 1
    __current_data__ = pd.DataFrame([], columns=["open", "high", "low", "close"])
    __balance_trend__ = []
    __current_balance__ = 0
    __day_trades__ = 0
    __current_date__ = datetime.datetime.now().day
    __max_avalanche_count__ = 3
    __maximum_order_count__ = 3
    __allowable_loss__ = 0.1
    __avalanche_amount__ = 0.1
    __avalanche_count__ = 0
    __simulation_start_time__ = None
    __simulation_end_time__ = None
    __simulation_type__ = None
    

    # For statistical purposes
    __initial_balance__ = 0
    __best_trade__ = 0
    __worst_trade__ = 0
    __equity_peak__ = 0
    __equity_least__ = 0


    # Strategies included in the backtest.
    strategies: list[Strategy]
    def __init__(self, active: str, strategies: list[Strategy], capital = 100, type = "offline"):
        """
        :Description:
        - Strategy backtesting class to include a strategy and measure possible returns from the strategy.

        :Arguments:
        - 
        """
        self.simulation_name = active
        self.simulation_data = self.get_local_data(self.simulation_name)
        
        self.strategies = []
        for strategy in strategies:
            self.strategies.append(strategy())

        self.__simulation_type__ = type
        self.__initial_balance__ = capital
        self.__current_balance__ = self.__initial_balance__
        self.__engine__ = TradingEngine()


    def get_balance(self):
        """
        Gets the current available capital of the backtest.

        :Returns:
        - Balance [int]: Balance capital of the backtest.
        """
        return round(self.__current_balance__, 2)


    def get_local_data(self, name: str):
        """
        Reads data locally from data folder with OHLCV.

        :arguments:
        - name: Name of the data file without an extension.

        :returns:
        - candles [df]: Candlesticks data with OHLC (Open, High, Low, Close).
        """
        candles = pd.read_csv(f"./data/{name}.csv")
        candles.columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        candles = candles[["timestamp", "open", "high", "low", "close"]]
        candles["timestamp"] = candles["timestamp"]*1000
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="ms")
        candles.set_index("timestamp", inplace=True)
        return candles


    def get_next_expiry(self):
        """
        Retrieves next expiry time depending on default expiration value.
        Eg. If expiry is 15m, current time is 08:05 returns 10 min.

        :return: None
        """
        now_time_minute = self.__current_data__.index[-1].minute
        return self.DEFAULT_EXPIRATION - now_time_minute % self.DEFAULT_EXPIRATION


    def get_data(self):
        """
        Retrieves data from the current data.

        :Arguments:
        - length [int]: Length of the data to return.

        :Returns:
        - candles [df]: Candlestick data requested.
        """
        if len(self.__current_data__) == 0:
            return pd.DataFrame([], columns=["open", "high", "low", "close"])
        return self.__current_data__[-self.strategies[0].maximum_window:]


    def get_current_day(self) -> datetime.datetime:
        """
        Gets the current day of the the backtest is currently on.

        :Returns:
        date [datetime]: The current date of the backtest.
        """
        if len(self.current_data) != 0:
            current_date = self.current_data.index[-1]
            logging.info(f"Current day is: {current_date}")
            return current_date
        return None


    def get_sleep_time(self, sleep_start: datetime.datetime) -> None:
        """
        Toggles on a flag on the backtest to block any new orders until sleep time ends prefebly after 24hrs.

        :Returns: None
        """
        current_date = self.get_current_day()
        time_delta = datetime.timedelta(hours=23-current_date.hour)
        sleep_time = (sleep_start + time_delta).timestamp()
        logging.info(f"Starting sleep session until: {datetime.datetime.fromtimestamp(sleep_time)}")
        return sleep_time
    

    def find_index_of(self, item):
        result = self.simulation_data[self.simulation_data == item].dropna()
        if len(result):
            return result.index[-1]
        return None


    def find_data_range(self, a: pd.Series, b: pd.Series):
        start_timestamp = self.find_index_of(a)
        end_timestamp = self.find_index_of(b)
        if not start_timestamp or not end_timestamp:
            return None, None
        return start_timestamp, end_timestamp


    def make_plot_snapshot(self, a: pd.Series, b: pd.Series, trade_direction = "call", is_correct = False):
        trade_points_configs = dict(
            color = "blue",
            symbol = "*",
            line_width = 2
        )
        # Get the data range requested by date_start and date_end
        start_timestamp, end_timestamp = self.find_data_range(a, b)
        if not start_timestamp and not end_timestamp:
            return
        all_data = self.simulation_data.loc[start_timestamp : end_timestamp]
        ohlc_data = all_data[all_data.columns[:4]]
        # Define the points of trades
        trade_points = pd.DataFrame(np.full(len(all_data), np.nan), index=all_data.index)
        price_level_start = a["close"]
        price_level_end = b["close"]
        trade_points.loc[start_timestamp] = price_level_start
        trade_points.loc[end_timestamp] = price_level_end
        indicator_data = all_data[all_data.columns[4:]]
        subplots = [mpf.make_addplot(trade_points, scatter=True, markersize=50, marker='o', color=trade_points_configs["color"])]
        for indicator_name in indicator_data.columns:
            excluded_indicators = ["rsi", "mband"]
            if indicator_name not in excluded_indicators:
                subplots.append(mpf.make_addplot(indicator_data[indicator_name], width=trade_points_configs["line_width"]))
        # Define a custom style with opacity for the candlesticks
        custom_style = mpf.make_mpf_style(base_mpf_style='charles', 
                        marketcolors={'candle': {'up': 'green', 'down': 'red'},
                                        'edge': {'up': 'k', 'down': 'k'},
                                        'wick': {'up': 'k', 'down': 'k'},
                                        'ohlc': {'up': 'k', 'down': 'k'},
                                        'volume': {'up': '#1f77b4', 'down': '#1f77b4'},
                                        'vcedge': {'up': '#1f77b4', 'down': '#1f77b4'},
                                        'vcdopcod': False,
                                        'alpha': .5})
        fig, axlist = mpf.plot(ohlc_data,
                        type='candle',
                        style=custom_style,
                        title=f"{'Call order' if trade_direction == 'call' else 'Put order'} on {start_timestamp} to {end_timestamp}, {is_correct}",
                        addplot=subplots,
                        volume=False, returnfig=True)
        axis = axlist[0]
        axis.axhline(y=price_level_start, color=trade_points_configs["color"], linestyle='--', linewidth=trade_points_configs["line_width"], label=f'Start at {price_level_start}')
        axis.axhline(y=price_level_end, color=trade_points_configs["color"], linestyle='--', linewidth=trade_points_configs["line_width"], label=f'End at {price_level_end}')
        fig.savefig(fname=f"./snapshots/{self.simulation_name}_{self.__trades_count__}.png")
        return ohlc_data


    def run(self, start = None, end = None):
        # Allow running from offline/local/online modes
        self.__simulation_start_time__ = time.time()
        sliced_data = self.simulation_data[start : end]
        logging.info(f"Simulation starting with: {len(sliced_data)} candles.")
        for _, row in sliced_data.iterrows():
            rowdf = pd.DataFrame([row])
            if len(self.__current_data__) == 0:
                self.__current_data__ = rowdf
            else:
                self.__current_data__ = pd.concat([self.__current_data__, rowdf])
            now_date: int = self.__current_data__.index[-1].day
            if self.__current_date__ != now_date:
                self.__day_trades__ = 0
                self.__current_date__ = now_date
            self.run_strategies(start = sliced_data.index[0], end = sliced_data.index[-1])
        return self.print_results(
            start = sliced_data.index[0],
            end = sliced_data.index[-1])


    def run_strategies(self, start: pd.Timestamp, end: pd.Timestamp):
        outcomes = []
        for strategy in self.strategies:
            current_data = self.__current_data__.iloc[-strategy.maximum_window:]
            outcomes.append(strategy.next(current_data))
        if len(outcomes):
            if -1 not in outcomes:
                current_outcome = outcomes[0]
                valid = True
                for outcome in outcomes:
                    if outcome != current_outcome:
                        valid = False
                        break
                if not valid:
                    return None
                else:
                    self.prepare_order(current_outcome)
                    # return self.fullfil_order(current_outcome, start = start, end = end)

    def prepare_order(outcome: int):
        
        

    def fullfil_order(self, strategy_outcome: int, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        valid_order_codes = [0, 1]
        if strategy_outcome in valid_order_codes:
            price = self.__current_data__.iloc[-1]
            price_index = self.find_index_of(price)
            if not price_index:
                logging.info("Price index could not be determined.")
                return False
            
            # Make an order
            order = Order(
                type = "call" if strategy_outcome == 1 else "put",
                price = price,
                buy_at = price_index,
                amount = self.get_balance() * self.__allowable_loss__)
            
            # Evaluate the signal
            future_price = self.simulation_data[self.simulation_data.index == order.expire_at].dropna()
            if not future_price.empty and self.__day_trades__ < self.__maximum_order_count__:
                future_price = future_price.iloc[-1]
                profit = round(order.amount * order.profit_ratio, 2)

                self.__trades_count__ += 1
                self.__day_trades__ += 1

                if future_price["close"] == order.purchase_price["close"]:
                    logging.info("Price is break-even.")
                    return

                if self.__avalanche_count__ > self.__max_avalanche_count__:
                    self.__avalanche_count__ = 0
                    self.__allowable_loss__ = self.__avalanche_amount__
                    logging.info("Allowable-loss ratio has been amount reset.")

                if order.type == "call":
                    if future_price["close"] > order.purchase_price["close"]:
                        self.__current_balance__ += profit
                        self.__trades_won__ += 1
                        self.__best_trade__ = profit if self.__best_trade__ < profit else self.__best_trade__
                        logging.info(f"Possible order: {order.type}, profit: {profit}, outcome: won, balance: {self.get_balance()}")
                        self.__allowable_loss__ += self.__avalanche_amount__
                        self.__avalanche_count__ += 1
                        logging.info(f"Allowable-loss ratio increased: {round(self.__allowable_loss__, 2)}")
                        self.make_plot_snapshot(order.purchase_price, future_price, trade_direction="call", is_correct = True)
                    else:
                        self.__current_balance__ -= order.amount
                        self.__trades_lost__ += 1
                        self.__worst_trade__ = order.amount if self.__worst_trade__ > order.amount else self.__worst_trade__
                        logging.info(f"Possible order: {order.type}, profit {profit}, outcome: lost, balance: {self.get_balance()}")
                        self.__allowable_loss__ = self.__avalanche_amount__
                        self.__avalanche_count__ = 0
                        logging.info("Allowable-loss ratio has been reset.")
                        self.make_plot_snapshot(order.purchase_price, future_price, trade_direction="call", is_correct = False)
                else:
                    if future_price["close"] < order.purchase_price["close"]:
                        self.__current_balance__ += profit
                        self.__trades_won__ += 1
                        logging.info(f"Possible order: {order.type}, profit {profit}, outcome: won, balance: {self.get_balance()}")
                        self.__best_trade__ = profit if self.__best_trade__ < profit else self.__best_trade__
                        self.__allowable_loss__ += self.__avalanche_amount__
                        self.__avalanche_count__ += 1
                        logging.info("Allowable-loss ratio increased:", round(self.__allowable_loss__, 2))
                        self.make_plot_snapshot(order.purchase_price, future_price, trade_direction="put", is_correct = True)
                    else:
                        self.__current_balance__ -= order.amount
                        self.__trades_lost__ += 1
                        logging.info(f"Possible order: {order.type}, profit {profit}, outcome: lost, balance: {self.get_balance()}")
                        self.__allowable_loss__ = self.__avalanche_amount__
                        self.__avalanche_count__ = 0
                        logging.info("Allowable loss ratio has been reset.")
                        self.make_plot_snapshot(order.purchase_price, future_price, trade_direction="call", is_correct = False)
                    self.make_plot_snapshot(order.purchase_price, future_price, trade_direction="put")
                    
                if self.__equity_peak__ < self.__current_balance__:
                    self.__equity_peak__ = self.__current_balance__
                self.__balance_trend__.append([self.__current_data__.index[-1], self.__current_balance__])
                
                # Update console results
                self.__simulation_end_time__ = time.time()
                self.print_results(start = start, end = end)


    def update_balance_plot(self):
        # Plot a balance trend graph to show the balance over time
        plot_data = pd.DataFrame(self.__balance_trend__, columns=["Index", "Balance"])
        plot_data.set_index("Index", inplace=True)
        plot = plot_data["Balance"].plot(title=f"Balance trend overtime on: {self.simulation_name}", legend=True)
        plot.figure.savefig(fname=f"Balance Trend - {self.simulation_name}.png")
        plt.close(plot.figure)


    def print_results(self, start: pd.Timestamp, end: pd.Timestamp):
        headers = ["Metric", "Outcome"]

        data_span = str(pd.Timedelta(seconds=(end.timestamp() - start.timestamp())))
        peak_equity = round(self.__equity_peak__, 2)
        final_equity = round(self.get_balance(), 2)
        roi = round(((self.get_balance() - self.__initial_balance__)/self.__initial_balance__) * 100, 2)

        metrics = [
            ["Simulation Name", self.simulation_name],
            ["Data Span", data_span],
            ["Initial Equity", colors.color_print(f"${round(self.__initial_balance__, 2)}", colors.Colors.YELLOW)],
            ["Peak Equity", colors.color_print(f"${peak_equity}", colors.Colors.GREEN)],
            ["Final Equity", colors.color_print(f"${final_equity}", colors.Colors.GREEN if final_equity > self.__initial_balance__ else colors.Colors.RED)],
            ["Return on Investment (ROI)", colors.color_print(f"{roi}%", colors.Colors.GREEN if roi > 0 else colors.Colors.RED)],
            ["Trades Count", colors.color_print(self.__engine__.__trades_count__, colors.Colors.YELLOW)],
            ["Best Trade", colors.color_print(f"${round(self.__best_trade__, 2)}", colors.Colors.GREEN)],
        ]

        if self.__trades_count__:
            metrics.append(["Accuracy Score (%)", f"{round(self.__engine__.__trades_correct__/self.__engine__.__trades_count__*100, 2)}%"])
        else:
            metrics.append(["Accuracy Score (%)", "N/A"])

        simulated_seconds = self.__current_data__.index[-1].timestamp() - self.__current_data__.index[0].timestamp()
        simulation_time = pd.Timedelta(seconds=simulated_seconds)
        metrics.append(["Simulated Time", simulation_time])
        tabular = tabulate(headers=headers, tabular_data=metrics, tablefmt="grid")
        self.clear_console()
        print(tabular)
        return tabular


    def clear_console(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
        
