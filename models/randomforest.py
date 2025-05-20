from typing import List
import warnings
import numpy as np
from joblibspark import register_spark
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import parallel_backend
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
            n_jobs=-1  # Sử dụng tất cả CPU cores
        )

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 784)  # MNIST: 28x28 = 784
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X, y)

        predictions = self.model.predict(X)
        accuracy = self.model.score(X, y)
        precision = precision_score(y, predictions, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predictions, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)

        return predictions, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 784)
        y = np.array(df.select("label").collect()).reshape(-1)
        
        predictions = self.model.predict(X)
        accuracy = self.model.score(X, y)
        precision = precision_score(y, predictions, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predictions, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)
        cm = confusion_matrix(y, predictions)
        
        return predictions, accuracy, precision, recall, f1, cm