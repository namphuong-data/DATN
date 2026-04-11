"""
check_layers.py
Kiểm tra dữ liệu trong từng layer Bronze/Silver/Gold từ MinIO
"""

import io
import pandas as pd
from minio import Minio

# ── Kết nối MinIO ──────────────────────────────────────────
client = Minio(
    "localhost:9000", access_key="minioadmin", secret_key="minioadmin123", secure=False
)

BUCKET = "lakehouse"
YEAR = "2022"


def read_layer(prefix: str, max_files: int = 3) -> pd.DataFrame:
    """Đọc tối đa max_files parquet từ một layer."""
    objects = list(client.list_objects(BUCKET, prefix=prefix, recursive=True))
    parquet_files = [o for o in objects if o.object_name.endswith(".parquet")]

    if not parquet_files:
        print(f"  [!] Không tìm thấy file parquet tại: {prefix}")
        return pd.DataFrame()

    dfs = []
    for obj in parquet_files[:max_files]:
        data = client.get_object(BUCKET, obj.object_name)
        dfs.append(pd.read_parquet(io.BytesIO(data.read())))

    return pd.concat(dfs, ignore_index=True)


def check_bronze():
    print("\n" + "=" * 55)
    print("  LAYER: BRONZE")
    print("=" * 55)
    df = read_layer(f"bronze/{YEAR}/")
    if df.empty:
        return

    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Schema    :\n{df.dtypes}")
    print(f"\n  Sample (3 rows):")
    print(df.head(3).to_string())
    print(f"\n  Null counts:\n{df.isnull().sum()}")


def check_silver():
    print("\n" + "=" * 55)
    print("  LAYER: SILVER")
    print("=" * 55)
    df = read_layer(f"silver/{YEAR}/")
    if df.empty:
        return

    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Schema    :\n{df.dtypes}")

    # Phân bố theo tháng
    if "event_month" in df.columns:
        print(f"\n  Phân bố theo tháng:")
        print(df["event_month"].value_counts().sort_index().to_string())

    # Thống kê số
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  Thống kê numeric:")
        print(df[num_cols].describe().to_string())


def check_gold():
    gold_tables = ["customer_features", "product_features", "monthly_summary"]

    for table in gold_tables:
        print("\n" + "=" * 55)
        print(f"  LAYER: GOLD / {table}")
        print("=" * 55)
        df = read_layer(f"gold/{YEAR}/{table}/")
        if df.empty:
            continue

        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {df.columns.tolist()}")
        print(f"\n  Sample (5 rows):")
        print(df.head(5).to_string())

        # Thống kê riêng cho từng bảng
        if table == "customer_features":
            if "total_spent" in df.columns:
                print(f"\n  Top 5 khách hàng chi nhiều nhất:")
                print(
                    df.nlargest(5, "total_spent")[
                        ["customer_id", "txn_count", "total_spent", "completion_rate"]
                    ].to_string()
                )

        elif table == "product_features":
            if "total_revenue" in df.columns:
                print(f"\n  Top 5 sản phẩm doanh thu cao nhất:")
                print(df.nlargest(5, "total_revenue").to_string())

        elif table == "monthly_summary":
            print(f"\n  Doanh thu theo tháng:")
            cols = [
                c
                for c in [
                    "event_month",
                    "txn_count",
                    "monthly_revenue",
                    "avg_order_value",
                    "active_customers",
                ]
                if c in df.columns
            ]
            print(df.sort_values("event_month")[cols].to_string())


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MinIO Lakehouse Inspector")
    print(f"  Bucket: {BUCKET}  |  Year: {YEAR}")
    print("=" * 55)

    check_bronze()
    check_silver()
    check_gold()

    print("\n✓ Done.")
