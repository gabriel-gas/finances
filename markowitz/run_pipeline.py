import subprocess

print("Starting pipeline...")

# 1. Download data from Yahoo Finance and compute returns
subprocess.run(["python", "data_etl.py"])

# 2. Compute volatility and risk metrics (VaR/CVaR)
subprocess.run(["python", "volatility.py"])

# 3. Run Markowitz optimization and Monte Carlo simulations
subprocess.run(["python", "portfolio_opt.py"])

# 4. Train Machine Learning model to detect market regimes (HMM)
subprocess.run(["python", "market_regimes.py"])

# 5. Simulate historical backtest crossing strategies
subprocess.run(["python", "backtest.py"])

print("Pipeline finished successfully!")