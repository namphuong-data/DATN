#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_gold.sh
# Chạy Gold Pipeline: Silver → Gold (demand by zone)
#
# Cách dùng:
#   bash scripts/run_gold.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SPARK_MASTER="spark://spark-master:7077"
DELTA_PACKAGE="io.delta:delta-core_2.12:2.4.0"
GOLD_JOB="/opt/spark/jobs/pipeline_gold.py"

echo "======================================================"
echo " Gold Pipeline — Silver → Gold (demand by zone)"
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
    --conf spark.sql.session.timeZone=UTC \
    --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
    --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
    --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
    --conf spark.hadoop.fs.s3a.access.key=minioadmin \
    --conf spark.hadoop.fs.s3a.secret.key=minioadmin123 \
    --conf spark.hadoop.fs.s3a.path.style.access=true \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
    "${GOLD_JOB}"

echo ""
echo "======================================================"
echo " ✓ Gold done!"
echo " MinIO console → http://localhost:9001"
echo "   Bucket: lakehouse"
echo "   gold/demand_by_zone/   ← Delta (features for model)"
echo "======================================================"
