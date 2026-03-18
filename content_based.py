from __future__ import annotations

import polars as pl


def get_content_based(item_id: str, items_df: pl.DataFrame) -> pl.DataFrame:
    """
    Score candidates by category hierarchy overlap:
    +1 category_l3 match, +1 category_l2 match, +1 category_l1 match.
    Deduplicate by category, remove the target category, then sort by score.
    """
    target = items_df.filter(pl.col("item_id") == item_id).head(1)
    if target.height == 0:
        return pl.DataFrame(schema={"item_id": pl.String, "score": pl.Int32})

    target_l1 = target.item(0, "category_l1")
    target_l2 = target.item(0, "category_l2")
    target_l3 = target.item(0, "category_l3")
    target_category = target.item(0, "category")

    scored = (
        items_df.with_columns(
            (
                pl.when(pl.col("category_l3") == target_l3).then(1).otherwise(0)
                + pl.when(pl.col("category_l2") == target_l2).then(1).otherwise(0)
                + pl.when(pl.col("category_l1") == target_l1).then(1).otherwise(0)
            ).alias("score")
        )
        .filter(pl.col("item_id") != item_id)
        .filter(pl.col("category") != target_category)
        .sort("score", descending=True)
        .unique(subset=["category"], keep="first")
        .sort(["score", "item_id"], descending=[True, False])
    )

    return scored
