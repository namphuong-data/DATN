#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_silver_gold.sh
# Chạy Silver → Gold pipeline
#
# Cách dùng:
#   bash scripts/run_silver_gold.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SPARK_MASTER="spark://spark-master:7077"
DELTA_PACKAGE="io.delta:delta-core_2.12:2.4.0"

SILVER_JOB="/opt/spark/jobs/pipeline_silver.py"
# GOLD_JOB="/opt/spark/jobs/pipeline_gold.py"

COMMON_CONF=(
  --master "${SPARK_MASTER}"
  --deploy-mode client
  --packages "${DELTA_PACKAGE}"
  --conf spark.jars.ivy=/root/.ivy2
  --conf spark.executor.memory=1g
  --conf spark.driver.memory=1g
  --conf spark.executor.cores=2
  --conf spark.sql.shuffle.partitions=4
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension"
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"
  --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000
  --conf spark.hadoop.fs.s3a.access.key=minioadmin
  --conf spark.hadoop.fs.s3a.secret.key=minioadmin123
  --conf spark.hadoop.fs.s3a.path.style.access=true
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false
)

# ── SILVER ───────────────────────────────────────────────────
echo "======================================================"
echo " [1/2] Silver Pipeline — Bronze → Silver (clean data)"
echo "======================================================"
docker exec spark-master \
  /opt/spark/bin/spark-submit "${COMMON_CONF[@]}" "${SILVER_JOB}"

echo ""
echo "✓ Silver done!"
echo ""

# ── GOLD ─────────────────────────────────────────────────────
# echo "======================================================"
# echo " [2/2] Gold Pipeline — Silver → Gold (demand by zone)"
# echo "======================================================"
# docker exec spark-master \
#   /opt/spark/bin/spark-submit "${COMMON_CONF[@]}" "${GOLD_JOB}"

echo ""
echo "======================================================"
echo " ✓ All done!"
echo " MinIO console → http://localhost:9001"
echo "   Bucket: lakehouse"
echo "   silver/all/              ← Delta (cleaned)"
echo "   gold/demand_by_zone/     ← Delta (features for model)"
echo "======================================================"
