from order import Order


class TradingEngine:
    __trades__: list[Order] = []
    __trades_count__ = 0
    __trades_correct__ = 0
    __trades_incorrect__ = 0

    __allows_bileteral_orders__ = True
    
    def __init__(self, bileteral_orders = True) -> None:
        self.__allows_bileteral_orders__ = bileteral_orders
        pass


    def place_order(self, order: Order):
        for trade in self.__trades__:
            if trade.symbol == order.symbol:
                if self.__allows_bileteral_orders__ == False:
                    return False
        self.__trades__.append(order)
        return True