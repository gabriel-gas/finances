import os 
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#----METRICS----
def metrics_annualized(df_returns: pd.DataFrame):
    mean_ann, mean_daily = df_returns.mean() * 252, df_returns.mean()
    cov_ann, cov_daily = df_returns.cov() * 252, df_returns.cov()
    return mean_ann, cov_ann, mean_daily, cov_daily

def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, 
                          cov_matrix: pd.DataFrame):
    ret = np.dot(weights, mean_returns)
    variance = weights.T @ cov_matrix @ weights
    vol = np.sqrt(variance)
    return ret, vol

#----OPTIMIZERS----
def negative_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, 
                          cov_matrix: pd.DataFrame, risk_free_rate: float):
    ret, vol = portfolio_performance( weights, mean_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol
    return -sharpe

def optimize_max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float):
    num_assets = len(mean_returns)

    constraints = { "type": "eq", "fun": lambda x: np.sum(x) - 1.0}

    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    initial_weights = num_assets * [1.0 / num_assets]

    result = minimize( negative_sharpe_ratio, initial_weights, 
        args=( mean_returns, cov_matrix, risk_free_rate), 
        method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x

def optimize_min_variance(mean_returns: pd.Series, cov_matrix: pd.DataFrame):
    num_assets = len(mean_returns)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]

    def portfolio_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x

#---- MONTE CARLO: Efficient frontier mapping ----
def simulate_portfolios(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                        risk_free_rate: float,
                        num_simulations: int = 10_000):
    num_assets = len(mean_returns)
    results = np.zeros((3+num_assets, num_simulations))
    np.random.seed(42)

    for i in range(num_simulations):    
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)

        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
        results[3:, i] = weights

    columns = ["Return", "Volatility", "Sharpe"] + list(mean_returns.index)

    return pd.DataFrame(results.T, columns=columns)

'''
Assumptions: log returns follow a Gaussian distribution with constant volatility.

This model is useful for simulation, but it does not fully capture real-market
features such as fat tails, volatility clustering, and time-varying volatility.

'''

#---- MONTE CARLO: Trajectory simulation via Brownian Motion ---- 
def simulate_portfolio_trajectories( weights: np.ndarray, mean_returns_daily: pd.Series,
    cov_matrix_daily: pd.DataFrame,
    initial_investment: float = 100000.0,
    time_horizon_days: int = 252,
    num_simulations: int = 10000,):

    np.random.seed(42)

    mu_p = np.dot(weights, mean_returns_daily)
    variance = weights.T @ cov_matrix_daily @ weights
    sigma_p = np.sqrt(variance)

    # Drift adjustment for log returns (Itô's Lemma)
    drift = mu_p - 0.5*(sigma_p**2)
    shocks = np.random.normal(0,1, (time_horizon_days, num_simulations))

    daily_log_returns = drift + sigma_p * shocks
    cum_log_returns = np.vstack([np.zeros(num_simulations), np.cumsum(daily_log_returns, axis = 0)])
    price_paths = initial_investment * np.exp(cum_log_returns)

    #tail risk evaluation at day 252 
    final_values = price_paths[-1, :]
    percentage_changes = (final_values - initial_investment) / initial_investment

    var_95_mc = np.percentile(percentage_changes, 5)
    cvar_95_mc = percentage_changes[percentage_changes <= var_95_mc].mean()

    return price_paths, var_95_mc, cvar_95_mc

#--- MAIN PIPELINE ---
def run_portfolio_optimization_pipeline(
    returns_path: str = "data/log_returns.csv", output_dir: str = "data"
):
    """Orchestrates Markowitz optimization and Monte Carlo simulation engines."""
    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            f"File '{returns_path}' not found. Please run previous ETL scripts first!"
        )

    df_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    mean_ann, cov_ann, mean_daily, cov_daily = metrics_annualized(df_returns)

    # Risk-free rate estimate from LFTS11 or default 10%
    risk_free_rate = mean_ann["LFTS11.SA"] if "LFTS11.SA" in mean_ann else 0.10

    # 1. Deterministic Markowitz Optimizations
    weights_sharpe = optimize_max_sharpe(mean_ann, cov_ann, risk_free_rate)
    weights_min_vol = optimize_min_variance(mean_ann, cov_ann)

    ret_sharpe, vol_sharpe = portfolio_performance(weights_sharpe, mean_ann, cov_ann)
    ret_min_vol, vol_min_vol = portfolio_performance(
        weights_min_vol, mean_ann, cov_ann
    )

    # 2. Monte Carlo Engine 1: Efficient Frontier (10,000 points)
    df_monte_carlo = simulate_portfolios(
        mean_ann, cov_ann, risk_free_rate, num_simulations=10000
    )

    # 3. Monte Carlo Engine 2: 1-Year Price Paths for Max Sharpe Portfolio
    paths, var_mc_1y, cvar_mc_1y = simulate_portfolio_trajectories(
        weights_sharpe,
        mean_daily,
        cov_daily,
        initial_investment=100000.0,
        time_horizon_days=252,
        num_simulations=10000,
    )

    df_allocations = pd.DataFrame(
        {
            "Max_Sharpe_Weights": weights_sharpe,
            "Min_Variance_Weights": weights_min_vol,
        },
        index=df_returns.columns,
    )

    df_summary = pd.DataFrame(
        {
            "Max_Sharpe_Portfolio": [
                ret_sharpe,
                vol_sharpe,
                (ret_sharpe - risk_free_rate) / vol_sharpe,
                var_mc_1y,
                cvar_mc_1y,
            ],
            "Min_Variance_Portfolio": [
                ret_min_vol,
                vol_min_vol,
                (ret_min_vol - risk_free_rate) / vol_min_vol,
                np.nan,
                np.nan,
            ],
        },
        index=[
            "Expected_Annual_Return",
            "Annual_Volatility",
            "Sharpe_Ratio",
            "VaR_95_MonteCarlo_1Year",
            "CVaR_95_MonteCarlo_1Year",
        ],
    )

    # Exporting results to CSV
    df_allocations.to_csv(os.path.join(output_dir, "portfolio_allocations.csv"))
    df_summary.to_csv(os.path.join(output_dir, "portfolio_summary.csv"))
    df_monte_carlo.to_csv(os.path.join(output_dir, "monte_carlo_portfolios.csv"))

    print("\nOptimized Weights Allocation:")
    print(df_allocations.map(lambda x: f"{x * 100:.2f}%"))
    print("\nPerformance & Risk Summary (1-Year Monte Carlo Projection):")
    print(df_summary)

    return df_allocations, df_summary, df_monte_carlo, paths


if __name__ == "__main__":
    run_portfolio_optimization_pipeline()
