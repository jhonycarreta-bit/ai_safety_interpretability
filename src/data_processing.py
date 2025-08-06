import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['feature_sum'] = df.select_dtypes(include='number').sum(axis=1)
    return df
