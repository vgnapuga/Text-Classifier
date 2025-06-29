from src.loader import DatasetLoader
from src.preprocessor import TextPreprocessor
from src.model import TextClassifier


class TextClassificationPipeline:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.loader = DatasetLoader(data_path)
        self.classifier = TextClassifier()


    def run(self, evaluate: bool) -> None:
        print("[1] Загрузка данных...")
        data = self.loader.load_data()

        print("[2] Разделение на train/test...")
        preprocessor = TextPreprocessor(data)
        X_train, X_test, y_train, y_test = preprocessor.split()

        print("[3] Обучение модели...")
        self.classifier.train(X_train, y_train)

        if evaluate:
            print("[4] Оценка модели...")
            y_pred = self.classifier.predict(X_test)
            self.classifier.evaluate(y_test, y_pred)
        else:
            print("[4] Пропуск оценки модели.")