import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentiment_signals import groupby_date, normalize_senti_score, generate_signals

def load_price(path="^GSPC.csv"):
    return pd.read_csv(path)

def join_sentiment_price(df_senti, sp):
    sp = sp.rename(columns={'Date':'date'})
    return pd.merge(df_senti, sp, on='date', how='inner')

def compute_returns(df):
    df = df.copy()
    df['Ret'] = df['Adj Close'] / df['Adj Close'].shift(1) - 1
    df['ModelRet'] = df['signal'].shift(1) * df['Ret']
    return df

def performance(ret):
    mean = ret.mean()
    std = ret.std()
    sharpe = mean/std if std!=0 else np.nan
    return mean, std, sharpe

def backtest(tweet_df):
    df = groupby_date(tweet_df)
    df = normalize_senti_score(df)
    df = generate_signals(df)

    sp = load_price()
    merged = join_sentiment_price(df, sp)
    merged = compute_returns(merged)

    model_ret = merged['ModelRet'].dropna()
    mean, vol, sharpe = performance(model_ret)
    return mean, vol, sharpe
