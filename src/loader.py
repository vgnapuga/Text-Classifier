import pandas as pd


class DatasetLoader:

    def __init__(self, filePath: str):
        self.filePath = filePath

    
    def load_data(self) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(self.filePath)

        if "text" not in data.columns or "label" not in data.columns:
            raise ValueError("CSV файл должен содержать столбцы 'text' и 'label'")
        
        return data