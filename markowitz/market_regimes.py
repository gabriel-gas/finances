''' Now, as an attempt to solve the limitation of only considering static gaussian 
 distributions, which, naturally, fail during structural market shocks, I'll apply a hidden markov model '''

#HMM - Hidden Markov Model: Resolves the problem by assuming that the observed market features are governed by an 
# underlying, unobservable (hidden) discrete state. v

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

'''Low variance state and high variance state will be 0 and 1, respectively. This will be mathematically
 forced (on the following code) '''

def prepare_features(df_returns: pd.DataFrame) -> pd.DataFrame:
    market_return = df_returns.mean(axis = 1)
    # 20-day rolling (annualized)
    market_vol = market_return.rolling(window = 20).std() * np.sqrt(252)

    features = pd.DataFrame({"Market_Return": market_return, "Market_Volatility": market_vol}).dropna()

    return features


def fit_ordered_hmm(X: np.ndarray, n_components: int = 2, random_state: int = 42):
    scaler =  StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter = 1000, random_state=random_state)
    model.fit(X_scaled) #Baum-welch algorithm 

    variances = [model.covars_[i][1,1] for i in range(n_components)]
    state_order = np.argsort(variances)
    

# If State 0 is not the low-variance state, re-order model parameters
    if not np.array_equal(state_order, np.arange(n_components)):
        model.means_ = model.means_[state_order]
        model.covars_ = model.covars_[state_order]
        model.transmat_ = model.transmat_[state_order][:, state_order]
        model.startprob_ = model.startprob_[state_order]

    return model, scaler

def predict_regimes(model, scaler, X: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X)
    hidden_states = model.predict(X_scaled)
    return hidden_states

def main():
    # Ensures directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Loading log returns from ETL
    returns_path = 'data/log_returns.csv'
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f'File {returns_path} not found. Run data_etl.py first')

    df_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Extracting features 
    features_df = prepare_features(df_returns)
    X = features_df.values

    # Fit ordered HMM
    model, scaler = fit_ordered_hmm(X)

    # Predict hidden states
    hidden_states = predict_regimes(model, scaler, X)

    #Saving results
    result_df = features_df.copy()
    result_df['Market_Regime']  = hidden_states
    result_df['Regime_label'] = np.where(result_df['Market_Regime'] == 0, 'Calm (Low Vol)', 'Stress (High Vol)')
    output_path = 'data/market_regimes.csv'
    result_df.to_csv(output_path)


if __name__ == '__main__':
    main()
