import pandas as pd
import pyarrow.parquet as pq


def read_preview(file_path, num_rows=10):
    """
    Đọc nhanh n dòng đầu tiên của file Parquet
    """
    try:
        # Sử dụng ParquetFile để chỉ mở header và đọc số dòng chỉ định
        # Cách này nhanh hơn pd.read_parquet vì không nạp toàn bộ file vào RAM
        parquet_file = pq.ParquetFile(file_path)

        # Đọc batch đầu tiên với kích thước num_rows
        head_data = next(parquet_file.iter_batches(batch_size=num_rows))

        # Chuyển sang DataFrame để hiển thị đẹp
        df = head_data.to_pandas()

        print(f"✅ Đã đọc {num_rows} dòng đầu tiên từ: {file_path}")
        return df

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


# --- SỬ DỤNG ---
file_to_check = "/home/namphuong/DATN/part-00000-58aefbb7-6bc5-4afe-abf4-977cb812e7f7-c000.snappy.parquet"  # Thay tên file của bạn vào đây
df_preview = read_preview(file_to_check, 10)

if df_preview is not None:
    # Hiển thị toàn bộ các cột (không bị dấu ...)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n--- DỮ LIỆU PREVIEW ---")
    print(df_preview)
    df_preview.to_csv("/home/namphuong/DATN/eda.csv", index=False, encoding="utf-8-sig")
