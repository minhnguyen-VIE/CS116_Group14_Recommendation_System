from __future__ import annotations

import polars as pl


def get_collaborative(item_id: str, transactions_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute item co-occurrence by day:
    - use updated_date truncated to date
    - match sessions by (customer_id, date)
    - count how often other items appear with target item
    """
    tx_with_date = transactions_df.with_columns(
        pl.col("updated_date").dt.date().alias("tx_date")
    )

    target_sessions = (
        tx_with_date.filter(pl.col("item_id") == item_id)
        .select(["customer_id", "tx_date"])
        .unique()
    )

    if target_sessions.height == 0:
        return pl.DataFrame(schema={"item_id": pl.String, "co_occurrence": pl.UInt32})

    co_occurrence = (
        tx_with_date.join(target_sessions, on=["customer_id", "tx_date"], how="inner")
        .filter(pl.col("item_id") != item_id)
        .group_by("item_id")
        .agg(pl.len().alias("co_occurrence"))
        .sort(["co_occurrence", "item_id"], descending=[True, False])
    )

    return co_occurrence
