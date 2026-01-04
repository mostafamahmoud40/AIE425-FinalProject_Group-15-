import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL SETTINGS
# ============================================================

pd.options.display.float_format = "{:.2f}".format

def save2(df, path):
    df2 = df.copy()
    float_cols = df2.select_dtypes(include="float").columns
    df2[float_cols] = df2[float_cols].round(2)
    df2.to_csv(path, index=False)

print("Global 2-decimal formatting enabled.")

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "SECTION1_DimensionalityReduction/data"
TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/Statistical Analysis"
PLOT_DIR = "SECTION1_DimensionalityReduction/results/plots/Statistical Analysis"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Step 1: Load Dataset
# ============================================================

ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

print("Users:", ratings.userId.nunique())
print("Movies:", ratings.movieId.nunique())
print("Ratings:", len(ratings))

# ============================================================
# Step 2: Preprocess ratings (1–5 scale)
# ============================================================

ratings['rating'] = ratings['rating'].clip(1, 5).astype("float32")

# ============================================================
# Step 3: Number of ratings per user (n_u)
# ============================================================

user_rating_count = ratings.groupby('userId').size().reset_index(name='n_u')
save2(user_rating_count, f"{TABLE_DIR}/user_rating_count.csv")

# ============================================================
# Step 4: Number of ratings per item (n_i)
# ============================================================

item_rating_count = ratings.groupby('movieId').size().reset_index(name='n_i')
save2(item_rating_count, f"{TABLE_DIR}/item_rating_count.csv")

# ============================================================
# Step 5: Average rating per user (r̄_u)
# ============================================================

user_avg_rating = ratings.groupby('userId')['rating'].mean().reset_index(name='r_bar_u')
save2(user_avg_rating, f"{TABLE_DIR}/user_avg_rating.csv")

# ============================================================
# Step 6: Average rating per item (r̄_i)
# ============================================================

item_avg_rating = ratings.groupby('movieId')['rating'].mean().reset_index(name='r_bar_i')
save2(item_avg_rating, f"{TABLE_DIR}/item_avg_rating.csv")

# ============================================================
# Step 7: Sort n_i and plot distribution
# ============================================================

item_sorted = item_rating_count.sort_values("n_i")

plt.figure(figsize=(10,5))
plt.plot(item_sorted['n_i'])
plt.xlabel("Items (sorted ascending)")
plt.ylabel("Number of Ratings")
plt.title("Distribution of Number of Ratings per Item")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/ratings_per_item_distribution.png")
plt.close()

# ============================================================
# Popularity groups: Low / Medium / High
# ============================================================

item_rating_count['popularity_pct'] = item_rating_count['n_i'].rank(pct=True) * 100

low_popularity = item_rating_count[item_rating_count['popularity_pct'] < 33]
medium_popularity = item_rating_count[
    (item_rating_count['popularity_pct'] >= 33) &
    (item_rating_count['popularity_pct'] < 66)
]
high_popularity = item_rating_count[item_rating_count['popularity_pct'] >= 66]

save2(low_popularity, f"{TABLE_DIR}/low_popularity_items.csv")
save2(medium_popularity, f"{TABLE_DIR}/medium_popularity_items.csv")
save2(high_popularity, f"{TABLE_DIR}/high_popularity_items.csv")

# ============================================================
# Step 8: Group products by percentile of avg rating
# ============================================================

item_avg_rating['percentile'] = item_avg_rating['r_bar_i'].rank(pct=True) * 100

bins = [0,1,5,10,20,30,40,50,60,70,100]
labels = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10']

item_avg_rating['group'] = pd.cut(
    item_avg_rating['percentile'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

group_counts = item_avg_rating['group'].value_counts().sort_index()
save2(item_avg_rating, f"{TABLE_DIR}/item_avg_rating_with_groups.csv")
group_counts.to_csv(f"{TABLE_DIR}/product_groups_count.csv")

# ============================================================
# Step 9: Total ratings per group (sorted)
# ============================================================

merged = item_avg_rating.merge(item_rating_count, on="movieId")

group_totals = (
    merged.groupby("group")['n_i']
    .sum()
    .reset_index()
    .sort_values("n_i")
)

save2(group_totals, f"{TABLE_DIR}/group_total_ratings_sorted.csv")

# ============================================================
# Step 10: Plot group distributions
# ============================================================

plt.figure(figsize=(10,4))
plt.bar(group_totals['group'], group_totals['n_i'])
plt.xlabel("Group")
plt.ylabel("Total Ratings")
plt.title("Ratings per Group (Sorted)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/ratings_per_group.png")
plt.close()

# ============================================================
# Step 11: Select and save target users
# ============================================================

user_rating_count['percentile'] = user_rating_count['n_u'].rank(pct=True) * 100

U1 = user_rating_count[user_rating_count.percentile < 2].sample(1).userId.iloc[0]
U2 = user_rating_count[(user_rating_count.percentile>=2)&(user_rating_count.percentile<5)].sample(1).userId.iloc[0]
U3 = user_rating_count[(user_rating_count.percentile>=5)&(user_rating_count.percentile<10)].sample(1).userId.iloc[0]

target_users_df = pd.DataFrame({
    "User": ["U1", "U2", "U3"],
    "userId": [U1, U2, U3]
})
target_users_df.to_csv(f"{TABLE_DIR}/target_users.csv", index=False)

# ============================================================
# Step 12 : Select target items 
# ============================================================

# Merge popularity and rating info
item_stats = item_rating_count.merge(
    item_avg_rating,
    on="movieId"
)


pop_low = item_stats["n_i"].quantile(0.30)
pop_high = item_stats["n_i"].quantile(0.70)


candidate_items = item_stats[
    (item_stats["n_i"] >= pop_low) &
    (item_stats["n_i"] <= pop_high) &
    (item_stats["r_bar_i"] >= 2.5) &
    (item_stats["r_bar_i"] <= 4.0)
]

# Safety check
if len(candidate_items) < 2:
    raise ValueError("Not enough items found.")

# Reproducible selection
np.random.seed(42)
selected_items = candidate_items.sample(2)

I1, I2 = selected_items["movieId"].tolist()

# Save target items
target_items_df = pd.DataFrame({
    "Item": ["I1", "I2"],
    "movieId": [I1, I2]
})

target_items_df.to_csv(
    f"{TABLE_DIR}/target_items.csv",
    index=False
)

print("Selected target items :")
print(target_items_df)


# ============================================================
# Step 13: Co-rating users & items
# ============================================================

user_items = ratings.groupby('userId')['movieId'].apply(set)
item_users = ratings.groupby('movieId')['userId'].apply(set)

def count_common_users(u):
    target = user_items[u]
    return user_items.apply(lambda s: len(s & target))

def count_common_items(i):
    target = item_users[i]
    return item_users.apply(lambda s: len(s & target))

CU1 = count_common_users(U1)
CU2 = count_common_users(U2)
CU3 = count_common_users(U3)

CI1 = count_common_items(I1)
CI2 = count_common_items(I2)

CU1.to_csv(f"{TABLE_DIR}/No_common_users_U1.csv")
CU2.to_csv(f"{TABLE_DIR}/No_common_users_U2.csv")
CU3.to_csv(f"{TABLE_DIR}/No_common_users_U3.csv")

CI1.to_csv(f"{TABLE_DIR}/No_coRated_items_I1.csv")
CI2.to_csv(f"{TABLE_DIR}/No_coRated_items_I2.csv")

# ============================================================
# Step 14: Thresholds (30% co-rated items)
# ============================================================

def compute_threshold(CU, user):
    t = 0.3 * len(user_items[user])
    eligible = CU[CU >= t]
    return eligible.max() if not eligible.empty else CU.max()

thresholds = pd.DataFrame({
    "User": ["U1", "U2", "U3"],
    "Threshold": [
        compute_threshold(CU1, U1),
        compute_threshold(CU2, U2),
        compute_threshold(CU3, U3)
    ]
})

thresholds.to_csv(f"{TABLE_DIR}/thresholds_30pct.csv", index=False)

print("Section ONE – Statistical Analysis completed successfully.")
