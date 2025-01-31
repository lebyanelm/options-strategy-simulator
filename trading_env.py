"""Dependencies"""
import ta
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
import ta.momentum
import ta.volatility
from actions import ACTIONS
from optimization import Optimization


class TradingEnv(gym.Env):
    """
    Policy:
    - Agent is rewarded x profit ratio when it makes a profit.
    - Agent is purnished with 1 when a loss is made.
    - Agent is purnished 1 + half (0.5) for making a streak losses.
    - Agent is also purnished when making orders above the maximum order count per day.

    Goal:
    An agent that predicts price trend in a given strike time, given x amount of backcandles,
    the agent should keep the daily order count to mitigate against risks and ensure longer growth.

    Possible improvements:
    Make sure the agent keeps a certain threshold of an accuracy to keep when making predictions.
    """
    def __init__(
            self, data: pd.DataFrame, optimization: Optimization):
        self.data = data
        self.capital = optimization.capital
        self.strike_time = optimization.strike_time
        self.interval = optimization.interval
        self.window_size = 1 or int(self.strike_time/self.interval)
        self.profit_ratio = optimization.profit_ratio
        self.target_accuracy = 1/(self.profit_ratio + 1)
        self.balance_at_risk = .1

        self.prices, self.signal_features = self._process_data()
        self.shape = (self.window_size, self.signal_features.shape[1])

        # Action and observation space
        self.action_space = spaces.Discrete(len(ACTIONS))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episde variables
        self.reset()

    def reset(self):
        self._set_new_values()
        self.end_step = self.signal_features.shape[0]
        return self._next_observation()
    
    def _set_new_values(self):
        self.current_step = self.interval
        self.daily_orders = 0
        self.total_reward = 0
        self.total_profit = 0
        self.total_reward = 0
        self.total_predictions = 0
        self.correct_predictions = 0
        self.current_accuracy = 0
        self.loss_streak = 0
        self.open_orders = []
        self.order_history = []
        self.has_ended = False
    
    def _process_data(self):
        open_ = self.data["Open"]
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]
        rsi = ta.momentum.rsi(close=close, window=13)
        atr = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)
        bollinger_low = ta.volatility.bollinger_lband(close=close, window=30)
        bollinger_up = ta.volatility.bollinger_hband(close=close, window=30)
        processed_data = pd.concat([close, rsi, atr, bollinger_low, bollinger_up], axis=1)
        processed_data.dropna(inplace=True)
        signal_features = np.column_stack((
            processed_data["Close"].to_numpy(),
            processed_data["rsi"].to_numpy(),
            processed_data["atr"].to_numpy(),
            processed_data["lband"].to_numpy(),
            processed_data["hband"].to_numpy()
        ))
        final_prices = processed_data["Close"].to_numpy()
        return final_prices.astype(np.float32), signal_features.astype(np.float32)

    def _next_observation(self):
        return self.signal_features[(self.current_step - self.window_size) : self.current_step]

    def _calculate_reward(self, action):
        step_reward = 0
        additional_reward = 0
        investment_amount = self.capital * self.balance_at_risk
        self.capital -= investment_amount
        profit = investment_amount * self.profit_ratio
        loss = investment_amount

        # Determine if the agent made a win/loss
        current_price = self.signal_features[self.current_step][0]
        expiry_price = self.signal_features[self.current_step + 1][0]

        """Reward the agent for a win with the profit and punish with loss, but reward half the profit on break-even."""
        if action == ACTIONS.CALL_ACTION:
            self.total_predictions += 1
            if expiry_price > current_price:
                step_reward = profit
                self.capital += (investment_amount + profit)
                self.correct_predictions += 1
            elif expiry_price < current_price:
                step_reward = -loss
            else:
                step_reward = -(loss * .5)
                
        elif action == ACTIONS.PUT_ACTION:
            self.total_predictions += 1
            if expiry_price < current_price:
                step_reward = profit
                self.capital += (investment_amount + profit)
                self.correct_predictions += 1
            elif expiry_price > current_price:
                step_reward -= loss
            else:
                step_reward = -(loss * .5)
        
        elif action == ACTIONS.NO_ACTION:
            pass

        """Additional rewarding for maintaining minimum level of accuracy"""
        self.current_accuracy = self.correct_predictions/max(1, self.total_predictions)
        if self.current_accuracy > self.target_accuracy:
            additional_reward += 0.5 + (self.current_accuracy - self.target_accuracy)
        else:
            additional_reward -= 1 + (self.target_accuracy - self.current_accuracy)
        return round(step_reward + additional_reward, 2)
        
    def step(self, action):
        self.current_step += 1
        self.has_ended = False
        step_reward = self._calculate_reward(action=action)
        info = dict()
        obs = self._next_observation()

        if self.current_step == self.end_step:
            self.has_ended = True

        self.total_reward += step_reward
        return obs, step_reward, False, self.has_ended, info

        

"""Reinforcement Learning stuff."""
data = pd.read_csv("https://raw.githubusercontent.com/nicknochnack/Reinforcement-Learning-for-Trading/main/data/gmedata.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data.sort_index(inplace=True)

optimization = Optimization()
env = TradingEnv(
    data=data,
    optimization=optimization
)

obs = env.reset()
EPISODES = 100
while env.has_ended == False:
    action = env.action_space.sample()
    obs, reward, _, done, info = env.step(action)
    print(f"Reward: {env.total_reward}, Done: {done}, Balance: {env.capital}")

