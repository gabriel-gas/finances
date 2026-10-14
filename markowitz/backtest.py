import os
import numpy as np
import pandas as pd

def run_backtest():
    returns_path = 'data/log_returns.csv'
    alloc_path = 'data/portfolio_allocations.csv'
    regimes_path = 'data/market_regimes.csv'

    required_files = [returns_path, alloc_path, regimes_path]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Required file {file_path} not found. Please run preceding pipeline modules first')

    # Loading datasets
    df_returns = pd.read_csv(returns_path, index_col = 0, parse_dates= True)
    df_alloc = pd.read_csv(alloc_path, index_col = 0)
    df_regimes = pd.read_csv(regimes_path, index_col = 0, parse_dates = True)

    # Aligning dates
    common_index = df_returns.index.intersection(df_regimes.index)
    df_returns = df_returns.loc[common_index]
    regimes = df_regimes.loc[common_index, 'Market_Regime']

    # Extracting portfolio weights from Markowitz optimization
    weights_max_sharpe = df_alloc['Max_Sharpe_Weights'].values
    weights_min_var = df_alloc['Min_Variance_Weights'].values

    # Walk-Forward
    static_returns = []
    dynamic_returns = []

    for date, row in df_returns.iterrows():
        regime = regimes.loc[date]

        # Static Buy & Hold Strategy: Max Sharpe
        r_static = np.dot(row.values, weights_max_sharpe)
        static_returns.append(r_static)

        # Dynamic Regime-Switching Strategy :
        # State 0 (calm) -> Max Sharpe | State 1 (Stress) -> Min Variance (Defensive)
        if regime == 0:
            w_dynamic = weights_max_sharpe
        else:
            w_dynamic = weights_min_var

        r_dynamic = np.dot(row.values, w_dynamic)
        dynamic_returns.append(r_dynamic)

    static_series= pd.Series(static_returns, index = common_index)
    dynamic_series = pd.Series(dynamic_returns, index = common_index)

    # Now we evaluate:
    def compute_metrics(daily_returns: pd.Series, strategy_name: str) -> dict:
        cumu_return = (1 + daily_returns).prod() - 1
        ann_return = daily_returns.mean() * 252
        ann_vol = daily_returns.std()* np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_return > 0 else 0

        # Max Drawdown
        cumulative = (1+daily_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        return{
            "Strategy": strategy_name,
            "Cumulative Return (%)": round(cumu_return * 100, 2),
            "Annualized Return (%)": round(ann_return * 100, 2),
            "Annualized Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown (%)": round(max_dd * 100, 2),
        }

    stats_static = compute_metrics(static_series, "Static Max Sharpe (B&H)")
    stats_dynamic = compute_metrics(dynamic_series, "HMM Regime-Switching")

    perf_table = pd.DataFrame([stats_static,stats_dynamic])

    print("\n" + "="*50)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*50)
    print(perf_table.to_string(index=False))
    print("="*50 + "\n")

    # Saving
    os.makedirs('data', exist_ok = True)

    results_df = pd.DataFrame({
        "Static_MaxSharpe_Return": static_series,
        "HMM_Dynamic_Return": dynamic_series,
        "Market_Regime": regimes })

    output_csv = "data/backtest_results.csv"
    results_df.to_csv(output_csv)
    print(f"Detailed backtest results successfully exported to {output_csv}")


if __name__== '__main__':
    run_backtest()
