from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

st.set_page_config(page_title="CS116 Recommender", layout="wide")

# Custom CSS to reduce box sizes
st.markdown("""
<style>
.stContainer {
    padding: 0.5rem !important;
}
.stButton > button {
    height: 2rem !important;
    font-size: 0.8rem !important;
    padding: 0.25rem 0.5rem !important;
}
.stImage img {
    max-height: 120px !important;
    width: auto !important;
}
.stTextInput input {
    font-size: 0.9rem !important;
    height: 2.5rem !important;
}
.stMarkdown p, .stMarkdown span {
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

from collaborative_based import get_collaborative
from content_based import get_content_based
from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent
LOCAL_PLACEHOLDER = BASE_DIR / "images" / "placeholder.jpg"
FALLBACK_PLACEHOLDER = "https://via.placeholder.com/320x220.png?text=Product+Image"
HOME_SAMPLE_PER_CATEGORY = 6
DETAIL_RECOMMENDATION_LIMIT = 12


@st.cache_data(show_spinner="Loading data...")
def load_cached_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    return load_data()


def format_price(value: Any) -> str:
    if isinstance(value, Decimal):
        return f"{value:,.0f}"
    if value is None:
        return "-"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def get_placeholder_image() -> str:
    if LOCAL_PLACEHOLDER.exists():
        return str(LOCAL_PLACEHOLDER)
    return FALLBACK_PLACEHOLDER


def get_item_row(items_df: pl.DataFrame, item_id: str) -> dict[str, Any] | None:
    row = items_df.filter(pl.col("item_id") == item_id).head(1)
    if row.height == 0:
        return None
    return row.to_dicts()[0]


# ── Product Detail Dialog ────────────────────────────────────────────


@st.dialog("Product Detail", width="large")
def show_product_detail(item_id: str) -> None:
    items_df, transactions_df = load_cached_data()
    item = get_item_row(items_df, item_id)
    if item is None:
        st.error("Product not found.")
        return

    st.caption(
        f"{item['category_l1']}  >  {item['category_l2']}  >  {item['category_l3']}"
    )

    img_col, info_col = st.columns([1, 2])
    with img_col:
        st.image(get_placeholder_image(), use_container_width=True)
    with info_col:
        st.subheader(str(item["category"]))
        st.markdown(f"**Brand:** {item.get('brand', '-')}")
        st.markdown(f"**Manufacturer:** {item.get('manufacturer', '-')}")
        st.markdown(
            f'**Price:** <span style="color:red;font-weight:bold">{format_price(item.get("price"))}</span>',
            unsafe_allow_html=True,
        )
        st.button("Buy", type="primary", use_container_width=True)

    st.divider()

    st.subheader("Frequently Bought Together")
    collab = get_collaborative(item_id, transactions_df)
    if collab.height > 0:
        collab_with_info = (
            collab.join(
                items_df.select(["item_id", "category", "brand", "price"]),
                on="item_id",
                how="left",
            )
            .sort("co_occurrence", descending=True)
            .unique(subset=["category"], keep="first")
            .sort("co_occurrence", descending=True)
            .head(DETAIL_RECOMMENDATION_LIMIT)
        )
        _mini_grid(collab_with_info, "collab", show_co_occurrence=True)
    else:
        st.info("No data available.")

    st.divider()

    st.subheader("Related Products")
    related = get_content_based(item_id, items_df).head(DETAIL_RECOMMENDATION_LIMIT)
    if related.height > 0:
        _mini_grid(
            related.select(["item_id", "category", "brand", "price"]),
            "related",
        )
    else:
        st.info("No data available.")


def _mini_grid(
    df: pl.DataFrame, prefix: str, cols: int = 10, show_co_occurrence: bool = False,
) -> None:
    rows = df.to_dicts()
    for i in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, item in enumerate(rows[i : i + cols]):
            with columns[j]:
                st.image(get_placeholder_image(), use_container_width=True)
                st.markdown(f"**{item.get('category', '-')}**")
                st.caption(item.get("brand", "-"))
                st.markdown(
                    f'<span style="color:red">{format_price(item.get("price"))}</span>',
                    unsafe_allow_html=True,
                )
                if show_co_occurrence and "co_occurrence" in item:
                    st.caption(f"Bought together: {item['co_occurrence']}x")


# ── Home Page ────────────────────────────────────────────────────────


def render_item_card(item: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        st.image(get_placeholder_image(), use_container_width=True)
        st.markdown(f"**{item.get('category', '-')}**")
        st.caption(f"{item.get('brand', '-')}")
        st.markdown(
            f'<span style="color:red">{format_price(item.get("price"))}</span>',
            unsafe_allow_html=True,
        )
        if st.button("View", key=key, use_container_width=True):
            st.session_state["popup_item_id"] = str(item["item_id"])


def render_item_grid(df: pl.DataFrame, prefix: str, cols: int = 3) -> None:
    if df.height == 0:
        st.info("No products.")
        return
    rows = df.to_dicts()
    for i in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, item in enumerate(rows[i : i + cols]):
            with columns[j]:
                render_item_card(item, key=f"{prefix}_{item.get('item_id')}")


def render_home(items_df: pl.DataFrame) -> None:
    st.title("CS116 Recommender")

    search_id = st.text_input("Search by item_id", placeholder="Enter item_id ...")
    if search_id:
        matched = items_df.filter(pl.col("item_id") == search_id.strip())
        if matched.height == 0:
            st.warning("item_id not found.")
        else:
            st.subheader("Search Result")
            render_item_grid(matched, prefix="search", cols=1)
        return

    if "sample_seed" not in st.session_state:
        st.session_state["sample_seed"] = random.randint(0, 999_999)
    seed = st.session_state["sample_seed"]

    categories = (
        items_df.select("category_l1")
        .drop_nulls()
        .unique()
        .sort("category_l1")
        .get_column("category_l1")
        .to_list()
    )

    for cat in categories:
        st.header(str(cat))
        subset = items_df.filter(pl.col("category_l1") == cat)
        sampled = subset.sample(
            n=min(HOME_SAMPLE_PER_CATEGORY, subset.height),
            seed=seed,
        ).select(
            ["item_id", "category", "brand", "price",
             "category_l1", "category_l2", "category_l3", "manufacturer"]
        )
        render_item_grid(sampled, prefix=f"home_{cat}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    items_df, _ = load_cached_data()

    render_home(items_df)

    popup_id = st.session_state.pop("popup_item_id", None)
    if popup_id:
        show_product_detail(popup_id)


if __name__ == "__main__":
    main()
