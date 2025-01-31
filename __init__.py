from strategies.svc_model import SVCAI
from strategies.bbands_rsi import BBRSI
from optimization import Optimization
from trading_env import TradingEnv
import backtesting
import signal
import time
import logging


logging.basicConfig(
    level=None, # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
)


"""
Program stop detector.
"""
def signal_handler(signal = None, frame = None) -> None:
    """
    Handle simulation cancellation (force-stop) to print-out partial results.
    """
    exit(signal if signal is not None else 0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# bt = backtesting.Backtesting(
#     symbols=["EURUSD=X"],
#     strategies=[SVCAI])
# bt.run(optimizations=[
#     Optimization(bileteral_orders=False),
# ])

# Include the data to be used for the RL training


