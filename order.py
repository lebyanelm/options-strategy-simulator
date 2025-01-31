import pandas as pd


class Order:
    def __init__(self,
                 symbol: str,
                 type: str,
                 price: float,
                 place_at: pd.Timestamp,
                 amount: float,
                 strategy: str,
                 duration: int = 15,
                 profit_ratio: float = 0.92):
        
        self.symbol = symbol
        self.type = type
        self.entry_price = price
        self.amount = round(amount, 2)
        self.profit_ratio = profit_ratio
        self.placed_at = place_at
        self.duration = duration
        self.strategy = strategy

        delta = pd.Timedelta(minutes=self.duration - (self.placed_at.minute % self.duration))
        self.expire_at = self.placed_at + delta