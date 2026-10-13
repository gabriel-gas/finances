# Markowitz Portfolio Optimization & Market Regime Detection Pipeline

Modular quantitative finance pipeline built in Python. This project implements Modern Portfolio Theory (MPT), advanced tail risk metrics, stochastic asset trajectory simulations, and machine learning-driven market regime classification using Hidden Markov Models (HMM).

--- 
## Core Modules Description

### 1. Data Ingestion & ETL (`data_etl.py`)
* Fetches historical adjusted close prices from Yahoo Finance (`yfinance`).
* Handles missing data points and synchronizes trading calendars.
* Computes continuous daily log returns:
  $$\ln\left(\frac{P_t}{P_{t-1}}\right)$$

### 2. Volatility & Tail Risk (`volatility.py`)
* **EWMA Volatility (RiskMetrics):** Implements exponential smoothing ($\lambda = 0.94$) to capture volatility clustering.
* **Tail Risk Metrics:** Computes parametric and historical **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR / Expected Shortfall)** at 95% and 99% confidence levels.

### 3. Portfolio Optimization & Monte Carlo (`portfolio_opt.py`)
* **Deterministic Optimization (SciPy SLSQP):**
  * *Max Sharpe Portfolio (Tangency Portfolio):* Maximizes risk-adjusted excess return.
  * *Global Minimum Variance (GMV):* Minimizes absolute portfolio volatility via covariance matrix shaping.
  * *Constraints:* Fully invested ($\sum w_i = 1$) and non-short-selling bounds ($0 \le w_i \le 1$).
* **Monte Carlo Engines:**
  * *Engine 1:* Maps the Efficient Frontier across 10,000 random weight combinations.
  * *Engine 2:* Simulates 1-year future asset trajectories via Geometric Brownian Motion (GBM) incorporating **Itô's Lemma drift correction** to project portfolio drawdown distributions.

### 4. Market Regime Detection (`market_regimes.py`)
* **Gaussian Hidden Markov Models (HMM):**
  * Unsupervised machine learning model that infers unobservable market states (Calm / Low Volatility vs. Stress / High Volatility) from observable market returns and rolling volatility.
  * Employs the **Baum-Welch (EM) algorithm** for parameter estimation and the **Viterbi algorithm** for state decoding.
  * **Label-Switching Prevention:** Automatically sorts state parameters by volatility variance, ensuring State 0 is strictly enforced as the low-volatility calm regime across pipeline executions.

### 5. Dynamic Backtesting Engine (`backtest.py`) *(Coming Soon / In Development)*
* **Walk-Forward Simulation:** Simulates historical out-of-sample portfolio performance over time.
* **Regime-Switching Allocation Strategy:** Dynamically adjusts asset exposure based on the HMM regime output. Shifts capital from aggressive equities to safe-haven fixed-income assets (`LFTS11`) during high-volatility stress states (State 1) and returns to the Max Sharpe portfolio during calm states (State 0).

### 6. Interactive Web Dashboard (`app.py`) *(Coming Soon / In Development)*
* **Streamlit Interface:** Provides a user-friendly web interface to visualize the Efficient Frontier, Monte Carlo simulation paths, historical asset correlations, and real-time market regime color-coded charts.


## USAGE
Run the core modules sequentially from your terminal:
### 1. Run Data ETL
python src/data_etl.py

### 2. Run Volatility Analysis
python src/volatility.py

### 3. Run Portfolio Optimization & Monte Carlo
python src/portfolio_opt.py

### 4. Run Market Regime Detection
python src/market_regimes.py

### 5. Run Interactive Dashboard (Optional)
streamlit run src/app.py
