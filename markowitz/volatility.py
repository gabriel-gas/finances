import numpy as np
import pandas as pd
import os
from scipy.stats import norm

def rolling_volatility(df_returns: pd.DataFrame, window = 21, annualize: bool = True) -> pd.DataFrame:

    scale = np.sqrt(252) if annualize else 1.0

    return df_returns.rolling(window=window).std()*scale

def ewma_volatility(df_returns: pd.DataFrame, lambda_param = 0.94, annualize: bool = True) -> pd.DataFrame:
    #$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$$
    variance = df_returns.ewm(alpha=(1-lambda_param)).var()
    scale = np.sqrt(252) if annualize else 1.0
    return np.sqrt(variance)*scale

def historical_VaR(df_returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    percentile = (1-confidence_level)*100
    VaR_series = np.percentile(df_returns.dropna(), percentile, axis=0)
    return pd.Series(VaR_series, index = df_returns.columns)

def CVaR(df_returns: pd.DataFrame, confidence_level: float = .95) -> pd.DataFrame:
    CVaR_dict = {}
    for col in df_returns.columns:
        series = df_returns[col].dropna()
        VaR_limit = np.percentile(series, (1 - confidence_level) * 100)
        tail_losses = series[series <= VaR_limit]
        CVaR_dict[col] = tail_losses.mean()

    return pd.Series(CVaR_dict)

def run_volatility_pipeline(returns_path: str = "data/log_returns.csv", output_dir: str = "data"):
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Arquivo '{returns_path}' não encontrado. Execute o data_etl.py primeiro!")

    df_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    roll_vol = rolling_volatility(df_returns, window=21)
    ewma_vol = ewma_volatility(df_returns)
    var_95 = historical_VaR(df_returns, confidence_level=0.95)
    CVaR_95 = CVaR(df_returns, confidence_level=0.95)

    risk_summary = pd.DataFrame(
        {
            "VaR_95_daily": var_95,
            "CVaR_95_daily": CVaR_95,
            "Vol_EWMA_annualized": ewma_vol.iloc[-1],
        }
    )

    roll_vol.to_csv(os.path.join(output_dir, "rolling_volatility.csv"))
    risk_summary.to_csv(os.path.join(output_dir, "risk_summary.csv"))

    print(f"Metrics saved in '{output_dir}/':")
    print(risk_summary.map(lambda x: f'{x*100:.2f}%'))
    
    return roll_vol, risk_summary


if __name__ == "__main__":
    run_volatility_pipeline()