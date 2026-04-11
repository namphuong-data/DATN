# Pipeline Lakehouse — Docker Setup

Hệ thống Docker thực hiện **Batch Pipeline (Page 2)** với **MinIO** làm object storage và **Apache Spark** xử lý dữ liệu theo kiến trúc **Medallion (Bronze → Silver → Gold)**.

---

## Kiến trúc dịch vụ

```
┌─────────────────────────────────────────────────────────┐
│  docker-compose.yml                                     │
│                                                         │
│  minio          :9000 (S3 API)   :9001 (Console UI)    │
│  minio-init     tạo bucket khi khởi động               │
│  spark-master   :7077 (RPC)      :8080 (UI)             │
│  spark-worker-1 :8081 (UI)                              │
│  spark-worker-2 :8082 (UI)                              │
└─────────────────────────────────────────────────────────┘
```

---

## Luồng xử lý (Batch only)

```
dataset/batch/2022/*.parquet   (flat — tất cả file thẳng trong 2022/)
          │
          ▼
   ┌─────────────┐
   │   BRONZE    │  s3a://lakehouse/bronze/2022/
   │  Raw ingest │  Parquet thô + metadata audit columns
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   SILVER    │  s3a://lakehouse/silver/2022/
   │   Cleaned   │  Dedup · Trim · Parse timestamp
   │    + Typed  │  total_value · is_completed · event_month
   └──────┬──────┘
          │
          ▼
   ┌──────────────────────────────────────────┐
   │                   GOLD                   │
   │  s3a://lakehouse/gold/2022/              │
   │  ├── customer_features/  (hành vi KH)   │
   │  ├── product_features/   (doanh thu SP) │
   │  └── monthly_summary/    (tổng theo T)  │
   └──────────────────────────────────────────┘
```

---

## Cấu trúc MinIO

```
lakehouse/
├── bronze/
│   └── 2022/              ← parquet thô từ toàn bộ file trong 2022/
├── silver/
│   └── 2022/              ← cleaned, typed, derived columns
└── gold/
    └── 2022/
        ├── customer_features/    ← per-customer behaviour stats
        ├── product_features/     ← per-product sales stats
        └── monthly_summary/      ← monthly aggregation
```

---

## Cấu trúc thư mục dự án

```
pipeline/
├── docker-compose.yml
├── Dockerfile.spark            ← Bitnami Spark 3.5 + Hadoop-AWS JARs
├── dataset/
│   └── batch/
│       └── 2022/
│           ├── file_a.parquet  ← đặt file Parquet của bạn vào đây (flat)
│           ├── file_b.parquet
│           └── ...
├── spark/
│   ├── conf/
│   │   └── spark-defaults.conf ← cấu hình S3A → MinIO
│   └── jobs/
│       └── pipeline_batch.py   ← job chính (Bronze → Silver → Gold)
└── scripts/
    ├── run_batch.sh             ← chạy pipeline 1 lệnh
    └── verify_minio.py          ← kiểm tra bucket sau khi chạy
```

---

## Hướng dẫn sử dụng

### 1. Đặt file Parquet vào đúng thư mục

```
dataset/batch/2022/your_file_1.parquet
dataset/batch/2022/your_file_2.parquet
...
```

File Parquet cần có ít nhất các cột (pipeline tự adapt nếu schema khác):
```
transaction_id, customer_id, product_id, amount, quantity,
category, timestamp, status
```

### 2. Build & khởi động stack

```bash
docker compose build
docker compose up -d
```

Chờ 30–60 giây để các service healthy.

### 3. Chạy Batch Pipeline

```bash
bash scripts/run_batch.sh
```

Hoặc trực tiếp trong container:

```bash
docker exec spark-master \
  /opt/bitnami/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /opt/spark/jobs/pipeline_batch.py
```

### 4. Kiểm tra kết quả

**MinIO Console** → http://localhost:9001
Login: `minioadmin` / `minioadmin123`

**Script verify** (cần `pip install minio` trên máy host):
```bash
python scripts/verify_minio.py
```

---

## Theo dõi Spark jobs

| URL | Mục đích |
|-----|----------|
| http://localhost:8080 | Spark Master UI |
| http://localhost:8081 | Worker 1 UI |
| http://localhost:8082 | Worker 2 UI |

```bash
docker logs -f spark-master    # live log của driver
```

---

## Dừng hệ thống

```bash
docker compose down        # giữ volume MinIO (data không mất)
docker compose down -v     # xóa toàn bộ volume (reset sạch)
```

---

## Port summary

| Service | Port | Giao thức |
|---------|------|-----------|
| MinIO S3 API | 9000 | HTTP/S3A |
| MinIO Console | 9001 | HTTP |
| Spark Master RPC | 7077 | TCP |
| Spark Master UI | 8080 | HTTP |
| Spark Worker 1 | 8081 | HTTP |
| Spark Worker 2 | 8082 | HTTP |
