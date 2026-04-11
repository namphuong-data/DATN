"""
pipeline_gold.py
─────────────────────────────────────────────────────────────────
Gold layer: MinIO Silver (Delta) → MinIO Gold (Delta)
- Aggregate theo PULocationID + pickup_hour (window 60 phút)
- Tạo feature table cho model dự đoán demand
─────────────────────────────────────────────────────────────────

Schema output (gold/demand_by_zone):
  pickup_date        date
  pickup_hour        int       -- 0–23
  PULocationID       int       -- NYC zone ID
  trip_count         long      -- số chuyến trong 1 giờ (TARGET)
  avg_trip_distance  double
  avg_fare_amount    double
  avg_tip_amount     double
  avg_duration_min   double
  avg_passenger      double
  total_passengers   long
─────────────────────────────────────────────────────────────────
"""

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline_gold")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET         = "lakehouse"
SILVER_SRC     = f"s3a://{BUCKET}/silver/all"
GOLD_DST       = f"s3a://{BUCKET}/gold/demand_by_zone"


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("Pipeline_Gold")
        .config("spark.hadoop.fs.s3a.endpoint",                 MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",               MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key",               MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access",        "true")
        .config("spark.hadoop.fs.s3a.impl",                     "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled",   "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.shuffle.partitions",                 "4")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def process_gold(spark: SparkSession):
    log.info("[GOLD] Reading Silver Delta: %s", SILVER_SRC)
    df = spark.read.format("delta").load(SILVER_SRC)
    log.info("[GOLD] Silver rows: %d", df.count())

    # ── Aggregate: mỗi zone × mỗi giờ ────────────────────────
    # GROUP BY: pickup_date + pickup_hour + PULocationID
    # → 1 row = demand tại zone X trong giờ H ngày D
    gold = (
        df.groupBy("pickup_date", "pickup_hour", "PULocationID")
        .agg(
            # TARGET: số chuyến đặt xe trong 1 giờ tại zone
            F.count("*").alias("trip_count"),

            # Features mô tả đặc trưng zone trong giờ đó
            F.round(F.avg("trip_distance"),    4).alias("avg_trip_distance"),
            F.round(F.avg("fare_amount"),      4).alias("avg_fare_amount"),
            F.round(F.avg("tip_amount"),       4).alias("avg_tip_amount"),
            F.round(F.avg("trip_duration_min"),4).alias("avg_duration_min"),
            F.round(F.avg("passenger_count"),  4).alias("avg_passenger"),
            F.sum("passenger_count").cast("long").alias("total_passengers"),
        )
        .withColumn("pickup_dayofweek",
                    F.dayofweek(F.col("pickup_date")))   # 1=Sun … 7=Sat
        .withColumn("is_weekend",
                    F.when(F.col("pickup_dayofweek").isin(1, 7), 1).otherwise(0))
        .withColumn("_layer",        F.lit("gold"))
        .withColumn("_processed_at", F.current_timestamp())
        .orderBy("pickup_date", "pickup_hour", "PULocationID")
    )

    count = gold.count()
    log.info("[GOLD] Rows after aggregate: %d", count)
    log.info("[GOLD] Sample:")
    gold.show(5, truncate=False)

    log.info("[GOLD] Writing Delta → %s", GOLD_DST)
    (
        gold.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("pickup_date")          # partition theo ngày → query nhanh
        .save(GOLD_DST)
    )
    log.info("[GOLD] ✓ Done — %d rows written", count)
    log.info("[GOLD] Schema:")
    gold.printSchema()


def main():
    log.info("=" * 60)
    log.info("Gold Pipeline START  (demand_by_zone)")
    log.info("  Source : %s", SILVER_SRC)
    log.info("  Output : %s", GOLD_DST)
    log.info("=" * 60)
    spark = build_spark()
    try:
        process_gold(spark)
        log.info("Gold Pipeline COMPLETE")
    except Exception as exc:
        log.error("Gold Pipeline FAILED: %s", exc, exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
