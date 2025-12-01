import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, DATASET_CSV

def load_raw_splits():
    train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
    val_df = pd.read_csv(VAL_CSV, encoding='utf-8')
    test_df = pd.read_csv(TEST_CSV, encoding='utf-8')
    return train_df, val_df, test_df

def build_full_dataset(save=True):
    train_df, val_df, test_df = load_raw_splits()
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if save:
        df.to_csv(DATASET_CSV, index=False)
    return df

def load_full_dataset():
    if os.path.exists(DATASET_CSV):
        return pd.read_csv(DATASET_CSV)
    return build_full_dataset(save=True)

def split_model_and_perf(df, ratio=0.8):
    split = int(len(df) * ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def prepare_model_df(model_df):
    model_df = model_df.rename(columns={'processed':'text','senti_label':'label'})
    encoder = LabelEncoder()
    model_df['label'] = encoder.fit_transform(model_df['label'])
    return model_df[['text','label']].reset_index(drop=True), encoder
