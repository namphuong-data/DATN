FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir tensorflow==2.13.0

RUN pip install --no-cache-dir "deltalake[pyarrow]" s3fs boto3

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib matplotlib

WORKDIR /app
