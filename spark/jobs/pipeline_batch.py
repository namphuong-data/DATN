"""
pipeline_batch.py
─────────────────────────────────────────────────────────────────
Batch pipeline: dataset/batch/*.parquet → MinIO Bronze (Delta Lake)
─────────────────────────────────────────────────────────────────
"""

import os
import glob
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta import configure_spark_with_delta_pip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline_batch")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET         = "lakehouse"
DATASET_SRC    = "/opt/spark/dataset/batch"
BRONZE_DST     = f"s3a://{BUCKET}/bronze/all"

TIMESTAMP_TYPES = (T.TimestampType, T.TimestampNTZType, T.DateType)
NUMERIC_TYPES   = (
    T.ByteType, T.ShortType, T.IntegerType, T.LongType,
    T.FloatType, T.DoubleType, T.DecimalType,
)


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("BatchPipeline_Bronze")
        .config("spark.hadoop.fs.s3a.endpoint",                 MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",               MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key",               MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access",        "true")
        .config("spark.hadoop.fs.s3a.impl",                     "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled",   "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.shuffle.partitions",                 "4")
        .config("spark.sql.parquet.mergeSchema",                "false")
        .config("spark.sql.parquet.enableVectorizedReader",     "false")
        
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def normalize_df(df: DataFrame) -> DataFrame:
    """Chuẩn hóa toàn bộ cột về 3 kiểu: Timestamp / Double / String."""
    new_cols = []
    for field in df.schema.fields:
        name  = field.name
        dtype = field.dataType
        if isinstance(dtype, TIMESTAMP_TYPES):
            new_cols.append(F.col(name).cast(T.TimestampType()).alias(name))
        elif isinstance(dtype, NUMERIC_TYPES):
            new_cols.append(F.col(name).cast(T.DoubleType()).alias(name))
        else:
            new_cols.append(F.col(name).cast(T.StringType()).alias(name))
    return df.select(new_cols)


def ingest_bronze(spark: SparkSession) -> DataFrame:
    log.info("[BRONZE] Scanning: %s", DATASET_SRC)

    parquet_files = sorted(glob.glob(os.path.join(DATASET_SRC, "*.parquet")))
    if not parquet_files:
        raise RuntimeError(f"Không tìm thấy file .parquet nào trong {DATASET_SRC}")

    log.info("[BRONZE] Found %d parquet files", len(parquet_files))

    df_all  = None
    skipped = []

    for i, fpath in enumerate(parquet_files):
        fname = os.path.basename(fpath)
        try:
            df_norm = normalize_df(spark.read.parquet(fpath))
            count   = df_norm.count()
            log.info("[BRONZE] [%d/%d] %-45s → %d rows",
                     i + 1, len(parquet_files), fname, count)
            if df_all is None:
                df_all = df_norm
            else:
                df_all = df_all.unionByName(df_norm, allowMissingColumns=True)
        except Exception as e:
            log.warning("[BRONZE] SKIP %s → %s", fname, str(e)[:120])
            skipped.append(fname)
            continue

    if df_all is None:
        raise RuntimeError("Không đọc được file nào!")

    if skipped:
        log.warning("[BRONZE] Skipped %d file(s): %s", len(skipped), skipped)

    # Gắn metadata
    df_all = (
        df_all
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_layer",       F.lit("bronze"))
    )

    total = df_all.count()
    log.info("[BRONZE] Total rows : %d", total)
    log.info("[BRONZE] Columns    : %s", df_all.columns)
    log.info("[BRONZE] Writing Delta → %s", BRONZE_DST)

    # Ghi Delta Lake (overwrite toàn bộ)
    (
        df_all.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_DST)
    )

    log.info("[BRONZE] ✓ Done — %d rows written (Delta format)", total)
    return df_all


def main():
    log.info("=" * 60)
    log.info("Batch Pipeline START  (Bronze — Delta Lake)")
    log.info("  Source : %s", DATASET_SRC)
    log.info("  Output : %s  (Delta)", BRONZE_DST)
    log.info("=" * 60)

    spark = build_spark()
    try:
        ingest_bronze(spark)
        log.info("Pipeline COMPLETE → s3a://%s/bronze/all/ (Delta)", BUCKET)
    except Exception as exc:
        log.error("Pipeline FAILED: %s", exc, exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
