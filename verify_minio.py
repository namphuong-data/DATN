"""
verify_minio.py
─────────────────────────────────────────────────────────────
Kiểm tra cấu trúc MinIO sau khi chạy pipeline.
Chạy trên máy host (cần: pip install minio)

  python scripts/verify_minio.py
─────────────────────────────────────────────────────────────
"""

from minio import Minio
from minio.error import S3Error
import sys

ENDPOINT   = "localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin123"
BUCKET     = "lakehouse"

client = Minio(ENDPOINT, access_key=ACCESS_KEY,
               secret_key=SECRET_KEY, secure=False)

print("=" * 55)
print(f"  MinIO bucket: {BUCKET}")
print("=" * 55)

try:
    if not client.bucket_exists(BUCKET):
        print(f"[ERROR] Bucket '{BUCKET}' does not exist.")
        sys.exit(1)

    layers = {"bronze": [], "silver": [], "gold": []}
    total  = 0

    for obj in client.list_objects(BUCKET, recursive=True):
        total += 1
        name = obj.object_name
        for layer in layers:
            if name.startswith(f"{layer}/"):
                layers[layer].append(name)
                break

    for layer, objects in layers.items():
        print(f"\n  [{layer.upper()}]  ({len(objects)} objects)")
        # Group by year/month
        paths = sorted(set("/".join(o.split("/")[:3]) for o in objects))
        for p in paths:
            count = sum(1 for o in objects if o.startswith(p))
            print(f"    {p}/  →  {count} file(s)")

    print(f"\n  Total objects: {total}")
    print("=" * 55)

except S3Error as e:
    print(f"[S3Error] {e}")
    sys.exit(1)
