import pandas as pd
from sklearn.model_selection import train_test_split


class TextPreprocessor:

    def __init__(self, data: pd.DataFrame):
        self.data = data


    def split(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        X: pd.Series = self.data["text"]
        y: pd.Series = self.data["label"]

        return train_test_split(X, y, test_size=.3, random_state=42)