"""After a power cut: delete work/*.parquet files that can't be read (truncated mid-write), so reruns rebuild them."""
import glob
import os

import polars as pl

for f in sorted(glob.glob("work/*.parquet") + glob.glob("work/*/*.parquet")):
    try:
        pl.read_parquet_schema(f)
    except Exception as e:
        print(f"bad, removing: {f} ({e.__class__.__name__})")
        os.remove(f)
print("parquet check done")
