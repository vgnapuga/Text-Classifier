import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from typing import Union


class TextClassifier:

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB)
        ])


    def train(
            self,
            X_train: Union[list[str], pd.Series],
            y_train: Union[list[str], pd.Series]
        ) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: Union[list[str], pd.Series]) -> list[str]:
        return self.model.predict(X_test).tolist()