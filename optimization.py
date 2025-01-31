class Optimization:
    def __init__(self,
                interval: str = 60,
                capital: int = 100,
                maximum_win_streak: int = 3,
                maximum_order_count: int = 3,
                risk_ratio: float = 0.1,
                risk_ratio_factor: float = 0.05,
                strike_time: int = 9,
                bileteral_orders: bool = False,
                profit_ratio: float = .92,
                maximum_loss_streak: int = 2):
        self.interval = interval
        self.capital = capital
        self.maximum_win_streak = maximum_win_streak
        self.maximum_order_count = maximum_order_count
        self.risk_ratio = risk_ratio
        self.risk_ratio_factor = risk_ratio_factor
        self.strike_time = strike_time
        self.bileteral_orders = bileteral_orders
        self.profit_ratio = profit_ratio
        self.maximum_loss_streak = maximum_loss_streak