''' Now, as an attempt to solve the limitation of only considering static gaussian 
 distributions, which, naturally, fail during structural market shocks, I'll apply a hidden markov model '''

#HMM - Hidden Markov Model: Resolves the problem by assuming that the observed market features are governed by an 
# underlying, unobservable (hidden) discrete state. v

import os
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

'''Low variance state and high variance state will be 0 and 1, respectively. This will be mathematically
 forced (on the following code) '''

def prepare_features(df_returns: pd.DataFrame) -> pd.DataFrame:
    market_return = df_returns.mean(axis = 1)
    # 20-day rolling (annualized)
    market_vol = market_return.rolling(window = 20) * np.sqrt(252)

    features = pd.DataFrame({"Market_Return": market_return, "Market_Volatility": market_vol}).dropna()

    return features


def fit_ordered_hmm(X: np.ndarray, n_components: int = 2, random_state: int = 42):
    scaler =  StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter = 1000, random_state=random_state)
    model.fit(X_scaled) #Baum-welch algorithm (just implementing)

    variances = [model.covars_[i][1,1] for i in range(n_components)]
    state_order = np.argsort(variances)

# If State 0 is not the low-variance state, re-order model parameters
    if not np.array_equal(state_order, np.arange(n_components)):
        model.means_ = model.means_[state_order]
        model.covars_ = model.covars_[state_order]
        model.transmat_ = model.transmat_[state_order][:, state_order]
        model.startprob_ = model.startprob_[state_order]

    return model, scaler
