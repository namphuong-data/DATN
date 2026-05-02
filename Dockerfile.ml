# Huấn luyện: train_lstm_demand.py (PyTorch), train_lgbm_low_hour_0_4.py, train_lgbm_evening_17_23.py — không dùng TensorFlow
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/usr/local/bin:${PATH}"

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    git \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    "deltalake[pyarrow]" \
    s3fs \
    boto3 \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    matplotlib \
    holidays \
    lightgbm==4.3.0 \
    confluent-kafka

# Base image đã có torch + CUDA 11.8; không cài thêm tensorflow
