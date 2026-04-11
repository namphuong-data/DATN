import logging
from pyspark.sql import functions as F
from spark.jobs.config.spark_builder import get_spark_session

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("pipeline_silver")

# Constants
BRONZE_SRC = "s3a://lakehouse/bronze/all"
SILVER_DST = "s3a://lakehouse/silver/all"

# Toàn bộ cột giữ lại trong Silver (Bronze + tính toán thêm)
SILVER_COLUMNS = [
    # ── Cột gốc từ Bronze ──────────────────────────────────
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "_source_file",
    "_ingested_at",
    # ── Cột mới tính toán ──────────────────────────────────
    "trip_duration_min",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_date",
    "_layer",
    "_processed_at",
]


def process_silver():
    spark = get_spark_session("NYC_Taxi_Silver_Transformation")

    try:
        # ── 1. Đọc Bronze ───────────────────────────────────
        log.info("[SILVER] Đang đọc dữ liệu từ Bronze Delta...")
        df = spark.read.format("delta").load(BRONZE_SRC)
        raw_count = df.count()
        log.info("[SILVER] Tổng số dòng Bronze : %d", raw_count)
        log.info("[SILVER] Số cột Bronze       : %d  %s", len(df.columns), df.columns)

        # ── 2. Lọc dữ liệu ─────────────────────────────────
        # Pickup/Dropoff không null, cùng tháng/năm
        df_cleaned = df.filter(
            F.col("tpep_pickup_datetime").isNotNull()
            & F.col("tpep_dropoff_datetime").isNotNull()
            & (
                F.year("tpep_pickup_datetime")
                == F.year("tpep_dropoff_datetime")
            )
            & (
                F.month("tpep_pickup_datetime")
                == F.month("tpep_dropoff_datetime")
            )
        )

        # Lọc outlier cơ bản
        df_cleaned = df_cleaned.filter(
            (F.col("trip_distance") > 0)
            & (F.col("fare_amount") >= 0)
            & (F.col("passenger_count") > 0)
            & (F.col("PULocationID").between(1, 263))
            & (F.col("DOLocationID").between(1, 263))
        )

        filtered_count = df_cleaned.count()
        log.info(
            "[SILVER] Sau lọc: %d dòng (bỏ %d dòng, %.1f%%)",
            filtered_count,
            raw_count - filtered_count,
            (raw_count - filtered_count) / raw_count * 100,
        )

        # ── 3. Thêm cột tính toán ───────────────────────────
        df_cleaned = (
            df_cleaned
            .withColumn(
                "trip_duration_min",
                (
                    F.unix_timestamp("tpep_dropoff_datetime")
                    - F.unix_timestamp("tpep_pickup_datetime")
                ) / 60.0,
            )
            .withColumn("pickup_year",      F.year("tpep_pickup_datetime"))
            .withColumn("pickup_month",     F.month("tpep_pickup_datetime"))
            .withColumn("pickup_day",       F.dayofmonth("tpep_pickup_datetime"))
            .withColumn("pickup_hour",      F.hour("tpep_pickup_datetime"))
            .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
            .withColumn("pickup_date",      F.to_date("tpep_pickup_datetime"))
            .withColumn("_layer",           F.lit("silver"))
            .withColumn("_processed_at",    F.current_timestamp())
        )

        # ── 4. Chọn đúng thứ tự cột ────────────────────────
        df_cleaned = df_cleaned.select(SILVER_COLUMNS)
        log.info(
            "[SILVER] Số cột Silver: %d  %s",
            len(df_cleaned.columns),
            df_cleaned.columns,
        )

        # ── 5. Ghi Delta ────────────────────────────────────
        log.info("[SILVER] Đang ghi dữ liệu vào %s ...", SILVER_DST)
        (
            df_cleaned.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("pickup_year", "pickup_month")
            .save(SILVER_DST)
        )
        log.info("[SILVER] ✓ Ghi Silver thành công!")

    except Exception as e:
        log.error("[SILVER] Lỗi trong quá trình xử lý: %s", e, exc_info=True)
        raise

    finally:
        # ── 6. Kiểm tra kết quả ────────────────────────────
        try:
            log.info("[SILVER] ── Kiểm tra kết quả ──")
            df_check = spark.read.format("delta").load(SILVER_DST)
            log.info("[SILVER] Tổng cột  : %d", len(df_check.columns))
            log.info("[SILVER] Tổng dòng : %d", df_check.count())
            log.info(
                "[SILVER] Danh sách cột:\n  %s",
                "\n  ".join(df_check.columns),
            )
            df_check.printSchema()
            df_check.show(3, truncate=False, vertical=True)
        except Exception as e:
            log.warning(
                "[SILVER] Không thể đọc lại Silver để kiểm tra: %s", e
            )
        finally:
            spark.stop()


if __name__ == "__main__":
    process_silver()
