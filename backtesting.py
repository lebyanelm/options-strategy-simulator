import logging
import colors
import datetime
import os
import datetime as dt
import yfinance as yf
import pandas as pd
from strategies.strategy import Strategy
from optimization import Optimization
from engine import TradingEngine
from order import Order
from tabulate import tabulate


class Backtesting:
    def __init__(self, symbols: list[str] = "EURUSD=X", strategies: list[Strategy] = None, capital = 100, mode = "offline", start_from: dt.datetime = None):
        # Instance variables initialization
        self.strike_time = 900
        self.interval = "1m"
        self.symbols = symbols
        self.balance_trend = []
        self.capital = capital
        self.maximum_win_streak = None
        self.maximum_loss_streak = None
        self.maximum_order_count = None
        self.minimum_entry_minutes = 15
        self.risk_ratio = None
        self.risk_ratio_factor = None
        self.profit_ratio = None
        self.mode = mode
        self.initial_capital = capital
        self.bileteral_orders = False
        self.start = start_from
        self.end = None
        self.progress = None
        
        # Statistical purposes
        self.orders_count = 0
        self.loss_count = 0
        self.win_orders_count = 0
        self.call_orders_count = 0
        self.correct_call_orders_count = 0
        self.put_orders_count = 0
        self.correct_put_orders_count = 0
        self.best_order = 0
        self.worst_order = 0
        self.highest_balance = capital
        self.day_orders_count = 0
        self.current_day = None
        
        # Active orders
        self.active_orders: list[Order] = []
        
        # Data storage
        self.data: dict[str, pd.DataFrame] = {}
        self.__data__: dict[str, pd.DataFrame] = {}
        
        # Initialize strategies
        self.strategies: list[Strategy] = []
        if strategies is not None:
            for strategy in strategies:
                self.strategies.append(strategy())

        # Load data immediately if backtest is meant to be offline
        if self.mode == "offline":
            self.load_data()
            self.apply_strategies()


    def load_data(self):
        delta = dt.timedelta(days=7)
        if self.start is None:
            self.end = dt.datetime.now()
            self.start = self.end - delta
        else:
            self.end = self.start + delta
        for symbol in self.symbols:
            try:
                self.data[symbol] = yf.download(
                    tickers = symbol,
                    interval = self.interval,
                    start = self.start,
                    end = self.end
                )
            except Exception as e:
                logging.info(e)


    def apply_strategies(self):
        logging.debug("Applying strategies to the loaded data.")
        pass


    def run(self, optimizations: list[Optimization] = None):
        if optimizations == None:
            optimizations = [Optimization()]

        if self.mode == "offline":
            for optimization in optimizations:
                logging.debug(f"Using optimization: {optimization.__dict__}")
                self.capital = optimization.capital
                self.initial_capital = self.capital
                self.highest_balance = self.capital
                self.bileteral_orders = optimization.bileteral_orders
                self.interval = optimization.interval
                self.maximum_order_count = optimization.maximun_order_count
                self.maximum_win_streak = optimization.maximum_win_streak
                self.risk_ratio = optimization.risk_ratio
                self.risk_ratio_factor = optimization.risk_ratio_factor
                self.profit_ratio = optimization.profit_ratio
                self.strike_time = optimization.strike_time
                self.maximum_loss_streak = optimization.maximum_loss_streak

                self.orders_count = 0
                self.day_orders_count = 0
                self.put_orders_count = 0
                self.call_orders_count = 0
                self.correct_call_orders_count = 0
                self.correct_put_orders_count = 0
                self.win_orders_count = 0
                self.loss_count = 0
                self.balance_trend = []
                self.active_orders = []
                self.best_order = 0
                self.worst_order = 0
                self.load_data()

                for symbol in self.data:
                    logging.debug(f"Running symbol: {symbol}")
                    if self.data[symbol].empty:
                        continue

                    for _, row in self.data[symbol].iterrows():
                        self.evaluate_active_orders()
                        if self.capital <= 1:
                            break
                        
                        day = row.name
                        row = pd.DataFrame([row])
                        if self.__data__.get(symbol) is None:
                            self.__data__[symbol] = row
                            self.current_day = day
                        else:
                            self.__data__[symbol] = pd.concat([self.__data__[symbol], row])
                        self.update_results(symbol)

                        if day != self.current_day:
                            self.current_day = day
                            self.day_orders_count = 0
                            self.loss_count = 0
                        
                        for strategy in self.strategies:
                            data = self.__data__[symbol].iloc[-300:]
                            outcome = strategy.next(data)
                            if outcome in [0, 1]:
                                if self.day_orders_count == self.maximum_order_count:
                                    continue 

                                self.prepare_order(
                                    outcome = outcome,
                                    symbol = symbol,
                                    strategy = strategy.name,
                                    entry_price = data.iloc[-1]["Close"],
                                    entry_time = data.index[-1])

        
    def prepare_order(self, outcome: int, symbol: str, strategy: str, entry_price: float, entry_time: pd.Timestamp):
        if self.bileteral_orders == False:
            if len(self.active_orders):
                return
            
        if self.strike_time - self.__data__[symbol].iloc[-1].name.minute % self.strike_time != self.strike_time:
            return
        
        amount = self.capital * self.risk_ratio
        self.capital -= amount
                
        order = Order(
            symbol = symbol,
            type = "call" if outcome == 1 else "put",
            price = entry_price,
            place_at = entry_time,
            strategy = strategy,
            amount = amount,
            duration = self.strike_time,
            profit_ratio = self.profit_ratio
        )
        logging.debug(f"Placed an order: {order.__dict__}")
        self.active_orders.append(order)

        """Statistics collection"""
        self.day_orders_count += 1
        self.orders_count += 1
        if order.type == "call":
            self.call_orders_count += 1
        else:
            self.put_orders_count += 1


    def evaluate_active_orders(self):
        if len(self.active_orders) == 0:
            return

        for active_order in self.active_orders:
            current_time: pd.Timestamp = self.__data__[active_order.symbol].iloc[-1].name
            expire_at = active_order.expire_at
            
            if current_time >= expire_at:
                logging.debug(f"Order entry on {active_order.placed_at} has expired just now on {current_time}.")
                current_price = self.__data__[active_order.symbol].iloc[-1]["Close"]
                correct_order = False

                if active_order.type == "call":
                    if current_price > active_order.entry_price:
                        correct_order = True
                        self.correct_call_orders_count += 1
                elif active_order.type == "put":
                    if current_price < active_order.entry_price:
                        correct_order = True
                        self.correct_put_orders_count += 1

                if correct_order:
                    profit_earned = round(active_order.amount * (1 + active_order.profit_ratio), 2)
                    self.win_orders_count += 1
                    self.capital += profit_earned
                    self.risk_ratio += self.risk_ratio_factor
                    if profit_earned > self.best_order:
                        self.best_order = profit_earned
                else:
                    self.risk_ratio = self.risk_ratio_factor
                    if self.worst_order < active_order.amount:
                        self.worst_order = active_order.amount
                if self.highest_balance < self.capital:
                    self.highest_balance = self.capital
                self.balance_trend.append([current_time, self.capital])
                self.active_orders.remove(active_order)


    def update_results(self, symbol):
        headers = ["Metric", "Outcome"]
        total_seconds = self.end.timestamp() - self.start.timestamp()
        current_end = self.__data__[symbol].iloc[-1].name

        data_span = pd.Timedelta(seconds=total_seconds)
        peak_equity = round(self.highest_balance, 2)
        final_equity = round(self.capital, 2)
        roi = round(((self.capital - self.initial_capital)/self.initial_capital) * 100, 2)

        metrics = [
            ["Simulation Name", self.symbols],
            ["Data Span", data_span],
            ["Initial Equity", colors.color_print(f"${round(self.initial_capital, 2)}", colors.Colors.YELLOW)],
            ["Peak Equity", colors.color_print(f"${peak_equity}", colors.Colors.GREEN)],
            ["Final Equity", colors.color_print(f"${final_equity}", colors.Colors.GREEN if final_equity > self.initial_capital else colors.Colors.RED)],
            ["Best Profit", colors.color_print(f"${round(self.best_order, 2)}", colors.Colors.GREEN)],
            ["Worst Loss", colors.color_print(f"${round(self.worst_order, 2)}", colors.Colors.RED)],
            ["Return on Investment (ROI)", colors.color_print(f"{roi}%", colors.Colors.GREEN if roi > 0 else colors.Colors.RED)],
            ["Active Orders", colors.color_print(len(self.active_orders), colors.Colors.YELLOW)],
            ["Order Count", colors.color_print(self.orders_count, colors.Colors.YELLOW)]
        ]

        if self.call_orders_count:
            metrics.append(["Call Order Count", (colors.color_print(f"{self.call_orders_count} ({round((self.correct_call_orders_count/self.call_orders_count)*100, 2)}%)", colors.Colors.YELLOW))])
        else:
            metrics.append(["Call Order Count", "0 (0%)"])

        if self.put_orders_count:
            metrics.append(["Put Order Count", (colors.color_print(f"{self.put_orders_count} ({round((self.correct_put_orders_count/self.put_orders_count)*100, 2)}%)", colors.Colors.YELLOW))])
        else:
            metrics.append(["Put Order Count", "0 (0%)"])

        if self.orders_count:
            metrics.append(["Accuracy Score (%)", f"{round((self.win_orders_count/self.orders_count)*100, 2)}%"])
        else:
            metrics.append(["Accuracy Score (%)", "N/A"])

        metrics = [
            *metrics,
            ["Progress", f"{current_end}"]]

        tabular = tabulate(headers=headers, tabular_data=metrics, tablefmt="grid")
        self.clear_console()
        print(f"Results for {symbol}: {self.start}")
        print(tabular)
        return tabular


    def clear_console(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')