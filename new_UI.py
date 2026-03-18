import streamlit as st
import polars as pl
import pandas as pd # Streamlit hiển thị dataframe tốt hơn với pandas, nhưng ta xử lý bằng polars

# --- TÍCH HỢP HÀM DATA LOADER ---
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ITEMS_PATH = DATA_DIR / "items.parquet"
TRANSACTIONS_PATH = DATA_DIR / "transactions-2025-12.parquet"

def load_items() -> pl.DataFrame:
    items = pl.read_parquet(ITEMS_PATH)
    return items.filter(pl.col("sale_status") == 1)

def load_transactions(items_df: pl.DataFrame | None = None) -> pl.DataFrame:
    active_items = items_df if items_df is not None else load_items()
    active_item_ids = active_items.select("item_id").unique()
    transactions = pl.read_parquet(TRANSACTIONS_PATH)
    return transactions.join(active_item_ids, on="item_id", how="inner")

def load_data_full() -> tuple[pl.DataFrame, pl.DataFrame]:
    items = load_items()
    transactions = load_transactions(items)
    return items, transactions

# --- TÍCH HỢP HÀM CONTENT BASED ---
def get_content_based(item_id: str, items_df: pl.DataFrame) -> pl.DataFrame:
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

# --- TÍCH HỢP HÀM COLLABORATIVE ---
def get_collaborative(item_id: str, transactions_df: pl.DataFrame) -> pl.DataFrame:
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

# --- HÀM UPSALE CHO TÃ ---
SIZE_ORDER = ["NB", "S", "M", "L", "XL", "XXL", "XXXL"]
def upsale_score(target_size, candidate_size):
    try:
        t_idx = SIZE_ORDER.index(target_size)
        c_idx = SIZE_ORDER.index(candidate_size)
        return max(0, c_idx - t_idx)
    except ValueError:
        return 0

# Cấu hình trang
st.set_page_config(layout="wide", page_title="CS116 Store")

# --- 1. CSS CUSTOM: MESH GRADIENT & GLASSMORPHISM ---
st.markdown(f"""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;700&display=swap');
    
    /* Global Font */
    * {{
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }}

    /* Mesh Gradient Background */
    .main {{
        background-color: #0f172a; /* Slate 900 */
        background-image: 
            radial-gradient(at 0% 0%, hsla(243, 75%, 59%, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, hsla(243, 75%, 59%, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 100%, hsla(243, 75%, 59%, 0.15) 0px, transparent 50%),
            radial-gradient(at 0% 100%, hsla(243, 75%, 59%, 0.1) 0px, transparent 50%);
        animation: mesh 20s ease infinite;
    }}

    @keyframes mesh {{
        0% {{ background-position: 0% 0%; }}
        50% {{ background-position: 100% 100%; }}
        100% {{ background-position: 0% 0%; }}
    }}

    /* Glassmorphism Card */
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
        color: white;
        min-height: 250px; /* Ensure consistent height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .glass-card:hover {{
        transform: translateY(-5px);
        border: 1px solid rgba(99, 102, 241, 0.5); /* Indigo 500 */
    }}

    /* Nút Primary Indigo 500 */
    .stButton>button {{
        background-color: #6366f1 !important; /* Indigo 500 */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100%;
        font-weight: 600;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Light Mode Adjustments */
    @media (prefers-color-scheme: light) {{
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            color: black;
        }}
    }}

    /* Category List */
    .category-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .category-item {{
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        background-color: transparent;
    }}
    .category-item:hover {{
        background-color: #e5e7eb; /* Gray rectangle on hover */
    }}
    .category-item.selected {{
        background-color: #6366f1;
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# --- 2. XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pl.read_parquet("data\items.parquet")
    # Giả lập xử lý cột size như bước trước bạn yêu cầu
    size_pattern = r"(?i)(NB|S|M|L|XL|XXL|XXXL|[23]XL)"
    df = df.with_columns(
        pl.col("description").str.extract(size_pattern, 1).str.to_uppercase().fill_null("Không xác định").alias("size")
    )
    return df

df = load_data()

# Khởi tạo session state để quản lý việc xem chi tiết
if 'viewing_item' not in st.session_state:
    st.session_state.viewing_item = None

# --- 3. SIDEBAR: DANH MỤC ---
categories = df.select("category_l1").unique().sort("category_l1").to_series().to_list()
with st.sidebar:
    st.title("🛒 CS116 Store")
    
    # Custom category list
    selected_cat = None
    for cat in categories:
        if st.button(cat, key=f"cat_{cat}", help=f"Chọn {cat}"):
            selected_cat = cat
    
    # If no button pressed, default to first category
    if selected_cat is None:
        selected_cat = categories[0] if categories else None
    
    if st.button("Về trang chủ"):
        st.session_state.viewing_item = None

# --- 4. GIAO DIỆN CHÍNH ---
if st.session_state.viewing_item is None:
    # HIỂN THỊ DANH SÁCH SẢN PHẨM
    st.subheader(f"Danh mục: {selected_cat}")
    
    # Lọc sản phẩm theo danh mục
    filtered_df = df.filter(pl.col("category_l1") == selected_cat)
    
    # Tạo grid 3 cột
    cols = st.columns(3)
    for idx, row in enumerate(filtered_df.to_dicts()):
        with cols[idx % 3]:
            # HTML cho Box sản phẩm Glassmorphism
            size_html = f"<p style='font-size: 0.8rem; color: #cbd5e1;'>Size: {row['size']}</p>" if row['size'] != "Không xác định" else ""
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="width:100%; height:150px; background:#1e293b; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-bottom:15px;">
                    <span style="color:#64748b;">Image Placeholder</span>
                </div>
                <p style="font-weight:bold; font-size:1.1rem; margin:5px 0;">{row['category']}</p>
                <h4 style="margin:0; color:#818cf8;">{row['brand'] if row['brand'] else 'No Brand'}</h4>
                <p style="color:#10b981; font-size:1.2rem; margin:0;">{row['price']:,.0f} VNĐ</p>
                {size_html}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Xem sản phẩm", key=f"btn_{row['item_id']}"):
                st.session_state.viewing_item = row['item_id']
                st.rerun()

else:
    # HIỂN THỊ CHI TIẾT SẢN PHẨM
    item_id = st.session_state.viewing_item
    item = df.filter(pl.col("item_id") == item_id).to_dicts()[0]
    
    if st.button("← Quay lại"):
        st.session_state.viewing_item = None
        st.rerun()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown(f"""<div class="glass-card" style="height:400px; display:flex; align-items:center; justify-content:center;">
            <h2 style="color:#64748b;">Hình ảnh {item['category']}</h2>
        </div>""", unsafe_allow_html=True)
        
    with col2:
        st.title(item['category'])
        st.subheader(f"Thương hiệu: {item['brand']}")
        st.header(f"Giá: {item['price']:,.0f} VNĐ")
        if item['size'] != "Không xác định":
            st.write(f"**Kích cỡ:** {item['size']}")
        st.write(f"**Mô tả:** {item['description']}")

    st.divider()
    
    # PHẦN GỢI Ý (Mẫu)
    tab1, tab2 = st.tabs(["🔥 Gợi ý mua cùng", "✨ Gợi ý tương tự/upsale"])

    # Load dữ liệu đầy đủ
    items_df, transactions_df = load_data_full()

    # Lọc size hợp lệ
    valid_items_df = items_df.filter(pl.col("size").is_in(SIZE_ORDER))

    # Gợi ý co-buy
    with tab1:
        cobuy_df = get_collaborative(item_id, transactions_df)
        if cobuy_df.height == 0:
            st.info("Không có dữ liệu mua cùng.")
        else:
            top_cobuy = cobuy_df.join(items_df, on="item_id", how="inner").head(4).to_dicts()
            cols = st.columns(4)
            for idx, row in enumerate(top_cobuy):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px; font-size:0.8rem;">
                        <p style="margin:0; font-weight:bold;">{row['category'][:30]}...</p>
                        <p style="color:#10b981;">{row['price']:,.0f}đ</p>
                        <p style="color:#6366f1;">Co-buy: {row['co_occurrence']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Gợi ý tương tự/upsale
    with tab2:
        # Nếu là tã, thêm logic upsale
        if item['category_l1'] == "Tã" and item['size'] in SIZE_ORDER:
            # Lọc các sản phẩm tã có size lớn hơn
            upsale_candidates = valid_items_df.filter(
                (pl.col("category_l1") == "Tã") & (pl.col("item_id") != item_id)
            )
            upsale_candidates = upsale_candidates.with_columns(
                pl.struct([pl.col("size")]).apply(lambda x: upsale_score(item['size'], x['size'])).alias("score_upsale")
            ).filter(pl.col("score_upsale") > 0)
            # Kết hợp với co-buy
            upsale_join = upsale_candidates.join(get_collaborative(item_id, transactions_df), on="item_id", how="left")
            upsale_join = upsale_join.with_columns(
                (pl.col("co_occurrence") * pl.col("score_upsale")).fill_null(0).alias("upsale_rank")
            ).sort("upsale_rank", descending=True).head(4)
            upsale_list = upsale_join.to_dicts()
            cols = st.columns(4)
            for idx, row in enumerate(upsale_list):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px; font-size:0.8rem;">
                        <p style="margin:0; font-weight:bold;">{row['category'][:30]}...</p>
                        <p style="color:#10b981;">{row['price']:,.0f}đ</p>
                        <p style="color:#6366f1;">Up-sale: {row['size']}</p>
                        <p style="color:#6366f1;">Rank: {row['upsale_rank']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Gợi ý tương tự theo category_l3/l2 và co-buy
            content_df = get_content_based(item_id, items_df)
            similar_join = content_df.join(get_collaborative(item_id, transactions_df), on="item_id", how="left")
            similar_join = similar_join.join(items_df, on="item_id", how="inner")
            similar_join = similar_join.with_columns(
                (pl.col("score") + pl.col("co_occurrence").fill_null(0)).alias("sim_rank")
            ).sort("sim_rank", descending=True).head(4)
            sim_list = similar_join.to_dicts()
            cols = st.columns(4)
            for idx, row in enumerate(sim_list):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px; font-size:0.8rem;">
                        <p style="margin:0; font-weight:bold;">{row['category'][:30]}...</p>
                        <p style="color:#10b981;">{row['price']:,.0f}đ</p>
                        <p style="color:#6366f1;">Rank: {row['sim_rank']}</p>
                    </div>
                    """, unsafe_allow_html=True)