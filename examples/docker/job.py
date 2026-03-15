from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Spark job for Docker-based end-to-end testing.")
    parser.add_argument("--rows", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.appName("docker-local-spark-job").getOrCreate()
    try:
        df = spark.range(args.rows).withColumn("bucket", F.col("id") % 17)
        grouped = df.groupBy("bucket").agg(
            F.count("*").alias("row_count"),
            F.sum("id").alias("id_sum"),
        )
        rows = grouped.orderBy("bucket").collect()
        print(f"computed_rows={len(rows)}")
        print(f"first_bucket={rows[0].bucket if rows else 'none'}")
        print(f"shuffle_partitions={spark.conf.get('spark.sql.shuffle.partitions')}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
