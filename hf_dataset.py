import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def build_hf_dataset(model_df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(model_df, test_size=test_size, random_state=seed)
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
