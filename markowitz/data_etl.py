from datetime import datetime
import os
import numpy as np
import pandas as pd
import yfinance as yf

def download_market_data(tickers: list[str], start_date: str, end_date: str ) -> pd.DataFrame:

    clean_tickers = [t.strip().upper() + (".SA" if not t.endswith(".SA") else "") for t in tickers]

    raw_data = yf.download(clean_tickers, start = start_date, end = end_date, auto_adjust = False)

    prices = (raw_data["Adj Close"] if "Adj Close" in raw_data.columns else raw_data["Close"] )
    return prices.to_frame() if isinstance(prices, pd.Series) else prices

def clean_and_align(df_prices: pd.DataFrame) -> pd.DataFrame:
    return df_prices.ffill().bfill().dropna(how = 'all')

def compute_log(df_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(df_prices / df_prices.shift(1)).dropna()

def run_etl_pipeline(tickers: list[str], start_date: str, end_date: str = None, output_dir: str = "data"):

    if end_date == None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok = True)

    prices = download_market_data(tickers, start_date, end_date)
    clean_prices = clean_and_align(prices)
    returns = compute_log(clean_prices)

    
    clean_prices.to_csv(os.path.join(output_dir, "clean_prices.csv"))
    returns.to_csv(os.path.join(output_dir, "log_returns.csv"))
    
    return clean_prices, returns



if __name__ == "__main__":
    b3_universe = [
        "PETR4.SA",   # Petrobras (Energy)
        "VALE3.SA",   # Vale (Materials)
        "ITUB4.SA",   # Itaú Unibanco (Financials)
        "WEGE3.SA",   # WEG (Industrials)
        "BOVA11.SA",  # Ibovespa ETF (Equity Benchmark)
        "LFTS11.SA"   # Tesouro Selic ETF (Fixed Income / Cash Proxy)
    ]
    
    run_etl_pipeline(
        tickers=b3_universe,
        start_date="2020-01-01",
        end_date= None
    )


