import os
import gzip
import numpy as np
from torchvision import datasets
import argparse

def save_mnist_to_ubyte(data_dir: str):
    """
    Tải dataset MNIST và lưu vào thư mục data/ dưới dạng các file .gz
    tương tự định dạng từ http://yann.lecun.com/exdb/mnist/
    """
    # Tạo thư mục data/ nếu chưa tồn tại
    os.makedirs(data_dir, exist_ok=True)

    # Tải MNIST dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)

    # Lưu train images
    train_images = train_dataset.data.numpy().astype(np.uint8)
    train_labels = train_dataset.targets.numpy().astype(np.uint8)

    # Lưu test images
    test_images = test_dataset.data.numpy().astype(np.uint8)
    test_labels = test_dataset.targets.numpy().astype(np.uint8)

    # Định dạng file theo chuẩn MNIST
    def write_ubyte_images(images, filename):
        with gzip.open(filename, 'wb') as f:
            # Header: magic number (2051), number of images, rows (28), cols (28)
            header = np.array([2051, len(images), 28, 28], dtype='>i4').tobytes()
            f.write(header)
            f.write(images.tobytes())

    def write_ubyte_labels(labels, filename):
        with gzip.open(filename, 'wb') as f:
            # Header: magic number (2049), number of labels
            header = np.array([2049, len(labels)], dtype='>i4').tobytes()
            f.write(header)
            f.write(labels.tobytes())

    # Lưu các file
    write_ubyte_images(train_images, os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    write_ubyte_labels(train_labels, os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    write_ubyte_images(test_images, os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    write_ubyte_labels(test_labels, os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    print(f"MNIST dataset đã được tải và lưu vào {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải và lưu MNIST dataset vào thư mục data/")
    parser.add_argument('--folder', '-f', help='Thư mục lưu dataset', default='data', type=str)
    args = parser.parse_args()
    
    save_mnist_to_ubyte(args.folder)