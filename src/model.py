import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from config import MODEL_PATH

from typing import Union


class TextClassifier:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path: str = model_path
        self.model: Union[Pipeline, None] = None


    def train(self, X_train, y_train) -> None:
        print("[✓] Обучаем модель...")
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("clf", MultinomialNB())
        ])
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, self.model_path)
        print(f"[✓] Модель сохранена: {self.model_path}")

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[!] Модель не найдена: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"[✓] Модель загружена из: {self.model_path}")

    def predict(self, X_test: Union[list[str], pd.Series]) -> list[str]:
        if self.model is None:
            self.load()

        return self.model.predict(X_test).tolist()
    
    def evaluate(self, y_test: Union[list[str], pd.Series], y_pred: Union[list[str], pd.Series]) -> None:
        print(classification_report(y_test, y_pred))