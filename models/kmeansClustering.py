from typing import List
import numpy as np
import matplotlib.pyplot as plt
import warnings
from joblibspark import register_spark
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import parallel_backend
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.decomposition import IncrementalPCA
from pyspark.sql.dataframe import DataFrame
from torchvision.utils import make_grid
from torch import tensor

warnings.filterwarnings('ignore')
register_spark()

class Kmeans:
    def __init__(self, n_clusters=10):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init_size=1024, reassignment_ratio=0.01, batch_size=512)
        self.pca = IncrementalPCA(n_components=10, whiten=False, batch_size=512)

    def configure_model(self, configs):
        model = self.model
        model.batch_size = configs.batch_size
        model.max_iter = configs.batch_size * 20
        return model

    def train(self, df: DataFrame, km: MiniBatchKMeans) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 784)  # MNIST: 28x28 = 784
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=8):
            km.partial_fit(X)

        predicted_cluster = km.predict(X)
        reference_labels = self.get_reference_dict(predicted_cluster, y)
        predicted_labels = self.get_labels(predicted_cluster, reference_labels)
        accuracy = accuracy_score(y, predicted_labels)
        loss = km.inertia_
        precision = precision_score(y, predicted_cluster, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predicted_cluster, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)

        return [km, predicted_cluster, accuracy, loss, precision, recall, f1]

    def predict(self, df: DataFrame, km: MiniBatchKMeans) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 784)
        y = np.array(df.select("label").collect()).reshape(-1)
        predicted_cluster = km.predict(X)
        reference_labels = self.get_reference_dict(predicted_cluster, y)
        predicted_labels = self.get_labels(predicted_cluster, reference_labels)
        cm = confusion_matrix(y, predicted_labels)
        accuracy = accuracy_score(y, predicted_labels)
        loss = km.inertia_
        precision = precision_score(y, predicted_cluster, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predicted_cluster, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)

        X = self.inverse_transform(X, mean=0.1307, std=0.3081)  # MNIST mean and std
        cluster_dict = {i: [] for i in np.unique(predicted_labels).astype(int)}
        for i in np.unique(predicted_labels):
            cluster_dict[i] = X[predicted_labels == i]

        self.visualize(cluster_dict, y)
        return [predicted_labels, accuracy, loss, precision, recall, f1, cm]

    def inverse_transform(self, X: np.ndarray, mean: float, std: float) -> np.ndarray:
        X = X * std + mean
        X = X * 255.0
        X = X.reshape(-1, 28, 28).astype(int)  # MNIST: 28x28
        return X

    def get_reference_dict(self, predictions, y):
        reference_label = {}
        for i in range(len(np.unique(predictions))):
            index = np.where(predictions == i, 1, 0)
            num = np.bincount(y[index == 1]).argmax()
            reference_label[i] = num
        return reference_label

    def get_labels(self, clusters, reference_labels):
        temp_labels = np.random.rand(len(clusters))
        for i in range(len(clusters)):
            temp_labels[i] = reference_labels[clusters[i]]
        return temp_labels

    def visualize(self, cluster_dict, y):
        classes = [str(i) for i in range(10)]
        for i in cluster_dict:
            grid = make_grid(tensor(cluster_dict[i].reshape(-1, 1, 28, 28)), nrow=10)
            fig = plt.figure(figsize=(10, 10))
            plt.title(f"Class: {classes[i]}")
            plt.imshow(grid.permute(1, 2, 0))
            plt.savefig(f"images/class_{classes[i]}.png")
            plt.close()