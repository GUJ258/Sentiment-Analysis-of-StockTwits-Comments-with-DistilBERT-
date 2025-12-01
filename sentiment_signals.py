import numpy as np
import pandas as pd
from config import TRAILING_DAYS, BREAK_THRESHOLD, TRADE_THRESHOLD

def senti_score(label):
    return -1 if label == 'bullish' else 1

def groupby_date(df):
    df = df.copy()
    df['senti_score'] = df['senti_label'].apply(senti_score)
    return df.groupby('date').sum(numeric_only=True)[['senti_score']]

def normalize_senti_score(df_date):
    df = df_date.copy()
    std = df['senti_score'].std()
    df['Normal_senti_score'] = df['senti_score'] / std if std != 0 else 0
    return df

def generate_signals(df_date):
    df = df_date.copy()
    df['trailing_sum'] = df['Normal_senti_score'].rolling(TRAILING_DAYS).sum()
    df['break'] = np.where(df['trailing_sum'] > BREAK_THRESHOLD, 1,
                   np.where(df['trailing_sum'] < -2*BREAK_THRESHOLD, -1, 0))
    roll = df['break'].rolling(TRAILING_DAYS).sum()
    df['signal'] = np.where(roll >= TRADE_THRESHOLD, 1,
                    np.where(roll <= -TRADE_THRESHOLD, -1, 0))
    return df
