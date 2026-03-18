from __future__ import annotations

from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ITEMS_PATH = DATA_DIR / "items.parquet"
TRANSACTIONS_PATH = DATA_DIR / "transactions-2025-12.parquet"


def load_items() -> pl.DataFrame:
    """Load items and keep only products that are still on sale."""
    items = pl.read_parquet(ITEMS_PATH)
    return items.filter(pl.col("sale_status") == 1)


def load_transactions(items_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    Load transactions and keep rows whose item_id is currently active.

    Mapping is done through an inner join on item_id.
    """
    active_items = items_df if items_df is not None else load_items()
    active_item_ids = active_items.select("item_id").unique()

    transactions = pl.read_parquet(TRANSACTIONS_PATH)
    return transactions.join(active_item_ids, on="item_id", how="inner")


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience loader for items + mapped transactions."""
    items = load_items()
    transactions = load_transactions(items)
    return items, transactions
