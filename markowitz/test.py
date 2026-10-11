import os 
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def metrics(returns: pd.DataFrame, annualized: bool = True):
    cov_ann, cov_daily = returns.cov()*252 , returns.cov()
    mean_ann, mean_daily = returns.mean()*252 , returns.mean() 

    return cov_ann, cov_daily, mean_ann, mean_daily

