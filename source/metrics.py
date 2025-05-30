import pandas as pd
import numpy as np

def compute_returns(pnl: pd.Series) -> pd.DataFrame:
    """
    Compute cumulative and daily returns, plus risk metrics.
    """
    daily_ret = pnl / pnl.shift(1).fillna(pnl.mean())
    cum_ret = (1 + daily_ret).cumprod() - 1
    metrics = {
        "daily_ret": daily_ret,
        "cum_ret": cum_ret,
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_dd": max_drawdown(cum_ret),
        "VaR_95": value_at_risk(daily_ret, 0.95),
    }
    return pd.DataFrame(metrics)

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    return (returns.mean() - risk_free) / returns.std()

def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    neg = returns[returns < 0]
    return (returns.mean() - risk_free) / neg.std() if not neg.empty else np.nan

def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return dd.min()

def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    return -np.percentile(returns.dropna(), (1 - level) * 100)