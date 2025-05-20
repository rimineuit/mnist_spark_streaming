# MNIST Spark Streaming Project

This project implements a streaming data pipeline using PySpark to train machine learning models (SVM, RandomForest, or KMeans) on the MNIST dataset. The system consists of a server that streams MNIST data and a client that receives and processes the data for training.

## Prerequisites
- Python 3.8+
- Spark 3.5.0 (included via `pyspark` package)
- Required Python packages (listed in `requirements.txt`)

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
