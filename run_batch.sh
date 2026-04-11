#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_batch.sh
# Load toàn bộ dataset/batch/*.parquet → MinIO (Delta Lake)
# (Bronze layer)
#
# Cách dùng từ máy host:
#   bash scripts/run_batch.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SPARK_MASTER="spark://spark-master:7077"
JOB_FILE="/opt/spark/jobs/pipeline_batch.py"

# Delta Lake 2.4.0 tương thích với Spark 3.4.x
DELTA_PACKAGE="io.delta:delta-core_2.12:2.4.0"

echo "======================================================"
echo " Batch Pipeline — dataset/batch/*.parquet → MinIO"
echo " Layer: bronze/all/  (Delta Lake format)"
echo "======================================================"

docker exec spark-master \
  /opt/spark/bin/spark-submit \
    --master "${SPARK_MASTER}" \
    --deploy-mode client \
    --packages "${DELTA_PACKAGE}" \
    --conf spark.jars.ivy=/root/.ivy2 \
    --conf spark.executor.memory=1g \
    --conf spark.driver.memory=1g \
    --conf spark.executor.cores=2 \
    --conf spark.sql.shuffle.partitions=4 \
    --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
    --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
    --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
    --conf spark.hadoop.fs.s3a.access.key=minioadmin \
    --conf spark.hadoop.fs.s3a.secret.key=minioadmin123 \
    --conf spark.hadoop.fs.s3a.path.style.access=true \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
    "${JOB_FILE}"

echo ""
echo "======================================================"
echo " Done!  MinIO console → http://localhost:9001"
echo "   minioadmin / minioadmin123"
echo "   Bucket: lakehouse"
echo "     bronze/all/          ← Delta Lake (_delta_log/)"
echo "======================================================"
