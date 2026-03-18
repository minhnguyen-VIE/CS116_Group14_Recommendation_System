# %%
import polars as pl

# %%
# Đọc file Parquet
df_items = pl.read_parquet("items.parquet")
df_trans = pl.read_parquet("transactions-2025-12.parquet")

# %%
df_items = df_items.filter(pl.col("category_l1").str.contains("(?i)Tã"))
# Loại bỏ các sản phẩm không còn bán
df_items = df_items.filter(pl.col("sale_status") == 1)

# %%
size_summary = df_items["size"].value_counts().sort("count", descending=True)

# 2. In kết quả với cấu hình hiển thị đầy đủ các dòng
with pl.Config(tbl_rows=100): 
    print("Thống kê các giá trị trong cột size:")
    print(size_summary)

# %%
raw_pattern = r"\s(XXXL|XXL|XL|XS|NB|S|M|L|newborn|Newborn|NEWBORN)\s"

df_items = df_items.with_columns(
    pl.when(pl.col("size") == "Không xác định")
    .then(
        pl.col("description")
        .str.extract(raw_pattern, 1) # Bắt thằng đầu tiên nằm giữa 2 khoảng trắng
        .str.replace(r"(?i)newborn", "NB")
        .str.to_uppercase()
    )
    .otherwise(pl.col("size"))
    .alias("size")
)

# %%
pattern_with_number = r"(?:^|[^a-zA-Z])(XXXL|XXL|XL|XS|NB|S|M|L)(\d)"

df_items = df_items.with_columns(
    pl.when((pl.col("size") == "Không xác định") | (pl.col("size").is_null()))
    .then(
        pl.col("description")
        # Trích xuất nhóm 1 (là phần chữ cái L, XL...) đứng ngay trước số
        .str.extract(pattern_with_number, 1)
        .str.to_uppercase()
    )
    .otherwise(pl.col("size"))
    .alias("size")
)

# %%
pattern_in_parens = r"\((XXXL|XXL|XL|XS|NB|S|M|L|newborn)\s*,\s*"

df_items = df_items.with_columns(
    pl.when((pl.col("size") == "Không xác định") | (pl.col("size").is_null()))
    .then(
        pl.col("description")
        # Trích xuất nhóm số 1 (chữ cái size) nằm sau dấu ngoặc và trước dấu phẩy
        .str.extract(pattern_in_parens, 1)
        .str.replace(r"(?i)newborn", "NB")
        .str.to_uppercase()
    )
    .otherwise(pl.col("size"))
    .alias("size")
)

# %%
# 1. Regex cho trường hợp dính ngoặc hoặc ký tự không phải chữ (ví dụ: "XXL(", "(M)", "L-")
# Ta dùng [^a-zA-Z] để đại diện cho bất kỳ ký tự nào KHÔNG PHẢI chữ cái
pattern_non_alpha = r"([^a-zA-Z]|^)(XXXL|XXL|XL|XS|NB|S|M|L)([^a-zA-Z]|$)"

# 2. Regex cho trường hợp đứng đơn lẻ (dùng ranh giới từ \b - cái này Polars hỗ trợ)
pattern_standalone = r"(?i)\b(XXXL|XXL|XL|XS|NB|S|M|L)\b"

df_items = df_items.with_columns(
    pl.coalesce([
        # Tầng 1: Trích xuất từ size dính ký tự lạ
        # Lưu ý: lấy index 2 vì nhóm 1 là ký tự đứng trước, nhóm 2 là size
        pl.col("size").str.extract(pattern_non_alpha, 2),
        
        # Tầng 2: Trích xuất size đứng đơn lẻ
        pl.col("size").str.extract(pattern_standalone, 1),
        
        # Tầng 3: Nếu không khớp mẫu nào, giữ nguyên giá trị cũ
        pl.col("size")
    ])
    .str.to_uppercase()
    .alias("size")
)

# %%
stats = (
    df_items.group_by("size")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

# Tính thêm tỷ lệ phần trăm (%)
total = df_items.height
stats = stats.with_columns(
    (pl.col("count") / total * 100).round(2).alias("percentage")
)

print(stats)
null_either = df_items.filter(
    pl.col("description").is_null() | pl.col("size").is_null()
)

print("Các dòng thiếu description hoặc size:")
print(null_either)

# %%
with pl.Config(fmt_str_lengths=2000):
    df_items = df_items.drop_nulls(subset=["size"])
    print(df_items.select("item_id", "size"))

# %%
from collaborative_based import get_collaborative
from content_based import get_content_based

#=============================
# Cập nhập input tại đây
#=============================
# Lấy các item liên quan đến item_id "2287000000004" 

related = get_content_based("2286000000002", df_items)
related2 = get_collaborative("2286000000002", df_trans)

print(related2)

# %%
result = (
    related2
    .join(df_items.select("item_id", "size"), on="item_id", how="left")
    
    
)
print(result.select("item_id", "size","score"))
print(result.select("item_id", "size","co_occurrence"))

# %%
import polars as pl

target_item_id = "2286000000002"

size_order = {
    "NB": 0,
    "S": 1,
    "M": 2,
    "L": 3,
    "XL": 4,
    "XXL": 5,
    "XXXL": 6
}

def size_rank_expr(col_name: str) -> pl.Expr:
    return (
        pl.when(pl.col(col_name) == "NB").then(0)
        .when(pl.col(col_name) == "S").then(1)
        .when(pl.col(col_name) == "M").then(2)
        .when(pl.col(col_name) == "L").then(3)
        .when(pl.col(col_name) == "XL").then(4)
        .when(pl.col(col_name) == "XXL").then(5)
        .when(pl.col(col_name) == "XXXL").then(6)
        .otherwise(None)
    )

# Lấy size của item gốc
target_size_row = (
    df_items
    .filter(pl.col("item_id") == target_item_id)
    .select("size")
    .head(1)
)

if target_size_row.height == 0:
    raise ValueError(f"Không tìm thấy item_id = {target_item_id} trong df_items")

target_size = target_size_row.item(0, "size")
target_rank = size_order.get(target_size)

if target_rank is None:
    raise ValueError(f"Size của target item không hợp lệ: {target_size}")

# Ghép size của candidate vào bảng related2
result = (
    related2
    .join(
        df_items.select("item_id", "size").unique(subset=["item_id"]),
        on="item_id",
        how="left"
    )
    .drop_nulls(subset=["size"])
)

# Tính score
result_scored = (
    result
    .with_columns([
        pl.lit(target_size).alias("target_size"),
        pl.lit(target_rank).alias("target_rank"),
        size_rank_expr("size").alias("candidate_rank"),
    ])
    .with_columns([
        (pl.col("candidate_rank") - pl.col("target_rank")).alias("size_delta")
    ])
    .with_columns([
        pl.col("size_delta").abs().alias("distance")
    ])
    .with_columns([
        pl.when(pl.col("size_delta") == 0).then(1.0)

        # lệch 1 size
        .when(pl.col("size_delta") == 1).then(0.9)     # candidate lớn hơn target 1 nấc
        .when(pl.col("size_delta") == -1).then(0.7)    # candidate nhỏ hơn target 1 nấc

        # lệch 2 size
        .when(pl.col("size_delta") == 2).then(0.75)
        .when(pl.col("size_delta") == -2).then(0.5)

        # lệch từ 3 size trở lên
        .when(pl.col("size_delta") >= 3).then(0.4)
        .when(pl.col("size_delta") <= -3).then(0.2)

        .otherwise(0.2)
        .alias("size_coefficient")
    ])
    .with_columns([
        (pl.col("co_occurrence") * pl.col("size_coefficient")).alias("final_score")
    ])
    .sort(["final_score", "co_occurrence"], descending=[True, True])
)

print(
    result_scored.select(
        "item_id",
        "size",
        "candidate_rank",
        "target_size",
        "target_rank",
        "size_delta",
        "distance",
        "co_occurrence",
        "size_coefficient",
        "final_score"
    )
)


from __future__ import annotations
import polars as pl

def get_score_upsale(result_scored: pl.DataFrame) -> pl.DataFrame:
    scored = result_scored.select(
        "item_id",
        "size",
        "candidate_rank",
        "target_size",
        "target_rank",
        "size_delta",
        "distance",
        "co_occurrence",
        "size_coefficient",
        "final_score"
    )
    return scored