import pandas as pd
import glob
import os

# Đường dẫn gốc
base_path = "dataset/batch"
# Danh sách các folder năm cần xử lý
years = ["2022", "2023", "2024"]


def summarize_parquet_with_quality(root_path, folders):
    for folder in folders:
        folder_path = os.path.join(root_path, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder {folder_path} không tồn tại.")
            continue

        print(f"📂 Đang phân tích folder: {folder}...")
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))

        all_stats = []

        for file_path in parquet_files:
            file_name = os.path.basename(file_path)
            try:
                # Đọc file
                df = pd.read_parquet(file_path)

                total_rows = len(df)
                total_cols = len(df.columns)

                # 1. Thống kê cơ bản
                stats = {
                    "file_name": file_name,
                    "total_rows": total_rows,
                    "total_columns": total_cols,
                    "unique_ids": df["id"].nunique() if "id" in df.columns else "N/A",
                }

                # 2. Thống kê chi tiết từng cột: % Non-null
                # df.count() trả về số lượng giá trị không phải null trong mỗi cột
                if total_rows > 0:
                    fill_rates = (df.count() / total_rows) * 100
                    for col_name, rate in fill_rates.items():
                        stats[f"col_{col_name}_fill_%"] = round(rate, 2)
                else:
                    for col_name in df.columns:
                        stats[f"col_{col_name}_fill_%"] = 0.0

                all_stats.append(stats)
                print(f"   ✅ Đã quét xong: {file_name} ({total_cols} cột)")

            except Exception as e:
                print(f"   ❌ Lỗi tại file {file_name}: {e}")

        # Xuất ra CSV
        if all_stats:
            summary_df = pd.DataFrame(all_stats)
            # Sắp xếp các cột: đưa thông tin chung lên đầu, sau đó đến các cột chi tiết
            cols = ["file_name", "total_rows", "total_columns", "unique_ids"]
            detail_cols = [c for c in summary_df.columns if c not in cols]
            summary_df = summary_df[cols + detail_cols]

            output_filename = f"quality_report_{folder}.csv"
            summary_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
            print(f"💾 Đã lưu báo cáo chất lượng: {output_filename}\n")


if __name__ == "__main__":
    summarize_parquet_with_quality(base_path, years)
