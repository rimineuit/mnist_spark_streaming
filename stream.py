import time
import json
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
import gzip

parser = argparse.ArgumentParser(description='Streams MNIST data to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Data folder', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', required=False, type=bool, default=False)
parser.add_argument('--split', '-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep', '-t', help="streaming interval", required=False, type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 6100

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []

    def load_mnist(self, image_file: str, label_file: str):
        # Load MNIST images
        with gzip.open(image_file, 'rb') as f:
            f.read(16)  # Skip header
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
        # Load MNIST labels
        with gzip.open(label_file, 'rb') as f:
            f.read(8)  # Skip header
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels

    def data_generator(self, image_file: str, label_file: str, batch_size: int):
        images, labels = self.load_mnist(image_file, label_file)
        self.data = list(map(np.ndarray.tolist, images))
        self.labels = labels.tolist()
        size_per_batch = (len(self.data) // batch_size) * batch_size
        batch = []
        for ix in range(0, size_per_batch, batch_size):
            image_batch = self.data[ix:ix + batch_size]
            label_batch = self.labels[ix:ix + batch_size]
            batch.append([image_batch, label_batch])
        self.data = self.data[size_per_batch:]
        self.labels = self.labels[size_per_batch:]
        return batch

    def sendMNISTBatchToSpark(self, tcp_connection, image_file, label_file, batch_size, split="train"):
        total_batch = (60000 if split == "train" else 10000) // batch_size + 1
        pbar = tqdm(total=total_batch)
        data_received = 0
        batches = self.data_generator(image_file, label_file, batch_size)
        for batch in batches:
            images, labels = batch
            images = np.array(images)
            batch_size, feature_size = images.shape
            images = images.tolist()

            payload = dict()
            for batch_idx in range(batch_size):
                payload[batch_idx] = dict()
                for feature_idx in range(feature_size):
                    payload[batch_idx][f'feature-{feature_idx}'] = images[batch_idx][feature_idx]
                payload[batch_idx]['label'] = labels[batch_idx]

            payload = (json.dumps(payload) + "\n").encode()
            try:
                tcp_connection.send(payload)
            except BrokenPipeError:
                print("Either batch size is too big or the connection was closed")
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")

            data_received += 1
            pbar.update(n=1)
            pbar.set_description(f"it: {data_received} | received: {batch_size} images")
            time.sleep(sleep_time)

    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamMNISTDataset(self, tcp_connection, folder, batch_size, split):
        if split == "train":
            image_file = os.path.join(folder, 'train-images-idx3-ubyte.gz')
            label_file = os.path.join(folder, 'train-labels-idx1-ubyte.gz')
        else:
            image_file = os.path.join(folder, 't10k-images-idx3-ubyte.gz')
            label_file = os.path.join(folder, 't10k-labels-idx1-ubyte.gz')
        self.sendMNISTBatchToSpark(tcp_connection, image_file, label_file, batch_size, split)

if __name__ == '__main__':
    args = parser.parse_args()
    data_folder = args.folder
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    if endless:
        while True:
            dataset.streamMNISTDataset(tcp_connection, data_folder, batch_size, train_test_split)
    else:
        dataset.streamMNISTDataset(tcp_connection, data_folder, batch_size, train_test_split)

    tcp_connection.close()