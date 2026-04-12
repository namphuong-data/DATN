"""
pipeline_gold.py
─────────────────────────────────────────────────────────────────
Gold Pipeline: Silver → Gold (demand_by_zone)
Mục tiêu: Tạo feature table phục vụ ML dự báo trip_count
          tại từng khu vực (PULocationID) cho 60 phút tiếp theo.

Timezone strategy:
  - tpep_pickup_datetime trong Silver là giờ NYC (America/New_York)
    nhưng được lưu không có timezone info (naive timestamp).
  - Dùng to_utc_timestamp() để convert sang UTC thật trước khi
    tạo window → window_start/end sẽ là UTC chuẩn.
  - Dùng from_utc_timestamp(..., "America/New_York") để extract
    hour/dow/month phản ánh đúng thực tế hành vi đặt xe NYC.

Features được tạo ra:
  - window_start / window_end     : khung giờ 60 phút (UTC)
  - trip_count                    : số chuyến trong khung giờ
  - target_demand                 : trip_count của khung giờ t+1 (nhãn ML)
  - hour, day_of_week, is_weekend : đặc trưng thời gian (giờ NYC)
  - lag_1h, lag_2h                : nhu cầu 1h và 2h trước
  - rolling_avg_3h                : trung bình trượt 3 giờ gần nhất
─────────────────────────────────────────────────────────────────
"""

import logging
from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType
from config.spark_builder import get_spark_session

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("pipeline_gold")

# ── Constants ─────────────────────────────────────────────────
SILVER_SRC      = "s3a://lakehouse/silver/all"
GOLD_DST        = "s3a://lakehouse/gold/demand_by_zone"

WINDOW_DURATION = "1 hour"   # kích thước cửa sổ thời gian
SLIDE_DURATION  = "1 hour"   # bước trượt (tumbling window)
LOCATION_COUNT  = 263        # số LocationID hợp lệ NYC (1–263)


# ─────────────────────────────────────────────────────────────
# 1. Đọc Silver
# ─────────────────────────────────────────────────────────────
def read_silver(spark):
    log.info("[GOLD] Đọc dữ liệu Silver từ %s ...", SILVER_SRC)
    df = spark.read.format("delta").load(SILVER_SRC)
    log.info("[GOLD] Silver rows: %d", df.count())
    return df


# ─────────────────────────────────────────────────────────────
# 2. Tính trip_count theo (PULocationID × 1-hour window)
#
# tpep_pickup_datetime là giờ NYC thật, lưu dạng naive (không có
# timezone info). Spark window() dùng thẳng giá trị này để gom
# nhóm → window_start/end cũng sẽ là giờ NYC. Không cần convert.
# ─────────────────────────────────────────────────────────────
def aggregate_demand(df):
    log.info("[GOLD] Aggregating demand per zone per hour window ...")

    df_agg = (
        df
        .groupBy(
            F.col("PULocationID"),
            F.window(F.col("tpep_pickup_datetime"), WINDOW_DURATION, SLIDE_DURATION)
        )
        .agg(F.count("*").alias("trip_count"))
        .select(
            F.col("PULocationID"),
            F.col("window.start").alias("window_start"),   # giờ NYC
            F.col("window.end").alias("window_end"),       # giờ NYC
            F.col("trip_count"),
        )
    )

    log.info("[GOLD] Aggregated rows (before filling zeros): %d", df_agg.count())
    return df_agg


# ─────────────────────────────────────────────────────────────
# 3. Fill zeros – đảm bảo mọi (LocationID × time_slot) đều có dòng
# ─────────────────────────────────────────────────────────────
def fill_zero_demand(spark, df_agg):
    log.info("[GOLD] Filling zero-demand slots via cross join ...")

    # ── 3a. Tập tất cả time slots có trong dữ liệu ───────────
    df_slots = df_agg.select("window_start", "window_end").distinct()

    # ── 3b. Tập tất cả LocationIDs hợp lệ (1–263) ───────────
    df_locations = spark.range(1, LOCATION_COUNT + 1).toDF("PULocationID")

    # ── 3c. Cross join → khung đầy đủ ───────────────────────
    df_full = df_locations.crossJoin(df_slots)

    # ── 3d. Left join với aggregated data ───────────────────
    df_filled = (
        df_full
        .join(
            df_agg,
            on=["PULocationID", "window_start", "window_end"],
            how="left"
        )
        .withColumn("trip_count", F.coalesce(F.col("trip_count"), F.lit(0)))
    )

    log.info("[GOLD] After zero-fill rows: %d", df_filled.count())
    return df_filled


# ─────────────────────────────────────────────────────────────
# 4. Feature Engineering
#
# window_start đã là giờ NYC (naive, tách thẳng từ
# tpep_pickup_datetime) → extract hour/dow/year/month trực tiếp,
# không cần convert timezone.
# ─────────────────────────────────────────────────────────────
def build_features(df_filled):
    log.info("[GOLD] Building time & lag features ...")

    # ── Window spec phân vùng theo zone, sắp xếp theo thời gian ──
    w_zone = (
        Window
        .partitionBy("PULocationID")
        .orderBy("window_start")
    )

    # ── Window spec cho rolling average (3 hàng trước, không tính hiện tại) ──
    w_roll3 = (
        Window
        .partitionBy("PULocationID")
        .orderBy("window_start")
        .rowsBetween(-3, -1)
    )

    df_features = (
        df_filled

        # ── Target: trip_count của khung giờ t+1 ─────────────
        .withColumn(
            "target_demand",
            F.lead("trip_count", 1).over(w_zone)
        )

        # ── Đặc trưng thời gian — tách thẳng từ window_start (giờ NYC) ──
        .withColumn("hour",        F.hour("window_start"))
        .withColumn("day_of_week", F.dayofweek("window_start"))  # 1=CN, 2=T2, ..., 7=T7
        .withColumn(
            "is_weekend",
            (F.dayofweek("window_start").isin(1, 7)).cast(IntegerType())
        )
        .withColumn("year",  F.year("window_start"))
        .withColumn("month", F.month("window_start"))

        # ── Lag features ──────────────────────────────────────
        .withColumn("lag_1h", F.lag("trip_count", 1).over(w_zone))
        .withColumn("lag_2h", F.lag("trip_count", 2).over(w_zone))

        # ── Rolling average (3 giờ gần nhất, không tính hiện tại) ──
        .withColumn("rolling_avg_3h", F.avg("trip_count").over(w_roll3))

        # ── Metadata ──────────────────────────────────────────
        .withColumn("_layer",        F.lit("gold"))
        .withColumn("_processed_at", F.current_timestamp())
    )

    # Loại bỏ các dòng cuối mỗi zone (không có target_demand)
    df_features = df_features.filter(F.col("target_demand").isNotNull())

    log.info("[GOLD] Feature rows (after dropping null targets): %d", df_features.count())
    return df_features


# ─────────────────────────────────────────────────────────────
# 5. Ghi Gold Delta Lake
# ─────────────────────────────────────────────────────────────
def write_gold(df_features):
    log.info("[GOLD] Ghi dữ liệu Gold vào %s ...", GOLD_DST)

    (
        df_features
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("year", "month")
        .save(GOLD_DST)
    )

    log.info("[GOLD] ✓ Ghi Gold thành công!")


# ─────────────────────────────────────────────────────────────
# 6. Kiểm tra kết quả
# ─────────────────────────────────────────────────────────────
def verify_gold(spark):
    log.info("[GOLD] ── Kiểm tra kết quả ──")
    try:
        df_check = spark.read.format("delta").load(GOLD_DST)
        log.info("[GOLD] Tổng cột  : %d", len(df_check.columns))
        log.info("[GOLD] Tổng dòng : %d", df_check.count())
        log.info(
            "[GOLD] Danh sách cột:\n  %s",
            "\n  ".join(df_check.columns),
        )
        df_check.printSchema()

        log.info("[GOLD] Sample — top 5 zones by avg demand:")
        (
            df_check
            .groupBy("PULocationID")
            .agg(F.avg("trip_count").alias("avg_demand"))
            .orderBy(F.desc("avg_demand"))
            .show(5, truncate=False)
        )

        # Kiểm tra phân phối hour — giờ cao điểm NYC nên rõ ở 7-9am, 5-7pm
        log.info("[GOLD] Phân phối hour theo giờ NYC:")
        df_check.groupBy("hour").agg(
            F.avg("trip_count").alias("avg_trips")
        ).orderBy("hour").show(24, truncate=False)

        log.info("[GOLD] Sample rows (vertical):")
        df_check.show(3, truncate=False, vertical=True)

    except Exception as e:
        log.warning("[GOLD] Không thể đọc lại Gold để kiểm tra: %s", e)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def process_gold():
    spark = get_spark_session("NYC_Taxi_Gold_DemandByZone")

    try:
        df_silver   = read_silver(spark)
        df_agg      = aggregate_demand(df_silver)
        df_filled   = fill_zero_demand(spark, df_agg)
        df_features = build_features(df_filled)
        write_gold(df_features)

    except Exception as e:
        log.error("[GOLD] Lỗi trong quá trình xử lý: %s", e, exc_info=True)
        raise

    finally:
        verify_gold(spark)
        spark.stop()


if __name__ == "__main__":
    process_gold()
