# ============================================================
# Part 1: PCA Method with Mean-Filling (Steps 1–13)
# MovieLens 20M 
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "SECTION1_DimensionalityReduction/data"
STAT_TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/Statistical Analysis"
PCA_TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/PCA Method with Mean-Filling"
PCA_PLOT_DIR = "SECTION1_DimensionalityReduction/results/plots/PCA Method with Mean-Filling"

os.makedirs(PCA_TABLE_DIR, exist_ok=True)
os.makedirs(PCA_PLOT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# ============================================================
# LOAD COMMON DATA 
# ============================================================

ratings = pd.read_csv(
    f"{DATA_DIR}/ratings.csv",
    usecols=["userId", "movieId", "rating"],
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)

target_items = pd.read_csv(
    f"{STAT_TABLE_DIR}/target_items.csv"
)

I1, I2 = target_items["movieId"].tolist()

# ============================================================
# STEP 1: Average rating for target items
# ============================================================

avg_target_items = (
    ratings[ratings["movieId"].isin([I1, I2])]
    .groupby("movieId")["rating"]
    .mean()
    .reset_index(name="avg_rating")
)

avg_target_items.to_csv(
    f"{PCA_TABLE_DIR}/avg_rating_target_items.csv",
    index=False
)

# ============================================================
# STEP 2: Mean-filling target items
# ============================================================

all_users = ratings["userId"].unique().astype("int32")
item_means = dict(zip(avg_target_items.movieId, avg_target_items.avg_rating))

mean_filled_items = []

for item_id in [I1, I2]:
    item_ratings = ratings.loc[
        ratings["movieId"] == item_id,
        ["userId", "movieId", "rating"]
    ]

    rated_users = set(item_ratings["userId"])
    missing_users = np.array(
        list(set(all_users) - rated_users),
        dtype="int32"
    )

    filled_rows = pd.DataFrame({
        "userId": missing_users,
        "movieId": np.full(len(missing_users), item_id, dtype="int32"),
        "rating": np.full(len(missing_users), item_means[item_id], dtype="float32")
    })

    mean_filled_items.append(
        pd.concat([item_ratings, filled_rows], ignore_index=True)
    )

mean_filled_df = pd.concat(mean_filled_items, ignore_index=True)

mean_filled_df.to_csv(
    f"{PCA_TABLE_DIR}/mean_filled_target_items.csv",
    index=False
)

# ============================================================
# STEP 3: Average rating per item
# ============================================================

item_avg_rating = (
    ratings
    .groupby("movieId")["rating"]
    .mean()
    .reset_index(name="avg_rating")
)

item_avg_rating.to_csv(
    f"{PCA_TABLE_DIR}/avg_rating_all_items.csv",
    index=False
)

# ============================================================
# STEP 4: Mean-centering
# ============================================================

ratings_centered = ratings.merge(
    item_avg_rating,
    on="movieId",
    how="left"
)

ratings_centered["rating_diff"] = (
    ratings_centered["rating"] - ratings_centered["avg_rating"]
).astype("float32")

ratings_centered = ratings_centered[
    ["userId", "movieId", "rating_diff"]
]

ratings_centered.to_csv(
    f"{PCA_TABLE_DIR}/item_mean_centered_ratings.csv",
    index=False
)

# ============================================================
# STEP 5: Covariance computation
# ============================================================

def compute_covariance(target_item, df):
    target_df = df[df["movieId"] == target_item][
        ["userId", "rating_diff"]
    ].rename(columns={"rating_diff": "r_target"})

    joined = df.merge(target_df, on="userId", how="inner")
    joined = joined[joined["movieId"] != target_item]

    joined["prod"] = joined["rating_diff"] * joined["r_target"]

    cov = (
        joined
        .groupby("movieId", as_index=False)["prod"]
        .mean()
        .rename(columns={"prod": "covariance"})
    )

    cov["target_item"] = target_item
    return cov

cov_all = pd.concat(
    [compute_covariance(I1, ratings_centered),
     compute_covariance(I2, ratings_centered)],
    ignore_index=True
)

cov_all.to_csv(
    f"{PCA_TABLE_DIR}/item_covariance_target_items.csv",
    index=False
)

# ============================================================
# STEP 6: Covariance matrix
# ============================================================

cov_matrix = (
    cov_all
    .pivot(index="movieId", columns="target_item", values="covariance")
    .fillna(0.0)
)

cov_matrix.columns = [f"cov_with_{c}" for c in cov_matrix.columns]

cov_matrix.to_csv(
    f"{PCA_TABLE_DIR}/covariance_matrix_target_items.csv"
)

# ============================================================
# STEP 7: Top-5 / Top-10 peers
# ============================================================

item_rating_count = pd.read_csv(
    f"{STAT_TABLE_DIR}/item_rating_count.csv"
)

cov_long = (
    cov_matrix.reset_index()
    .melt(id_vars="movieId", var_name="target_item", value_name="covariance")
)

cov_long["target_item"] = (
    cov_long["target_item"]
    .str.replace("cov_with_", "")
    .astype("int32")
)

cov_long = cov_long.merge(item_rating_count, on="movieId", how="left")

peer_rows = []

for target in cov_long["target_item"].unique():
    subset = cov_long[cov_long["target_item"] == target]
    subset = subset.sort_values(
        by=["covariance", "n_i"],
        ascending=[False, False]
    )

    peer_rows.append(subset.head(5).assign(peer_group="Top-5"))
    peer_rows.append(subset.head(10).assign(peer_group="Top-10"))

peers_df = pd.concat(peer_rows, ignore_index=True)

peers_df.to_csv(
    f"{PCA_TABLE_DIR}/top5_top10_peers.csv",
    index=False
)

# ============================================================
# STEP 8 & 10 : Reduced space using TRUE PCA
# ============================================================

def reduced_space(peers_df, label, max_components):
    spaces = []

    for target in peers_df["target_item"].unique():

        # Select peer items
        peer_items = peers_df[
            (peers_df["target_item"] == target) &
            (peers_df["peer_group"] == label)
        ]["movieId"].tolist()

        # User × Item matrix (mean-centered, mean-filled)
        subset = ratings_centered[
            ratings_centered["movieId"].isin(peer_items)
        ]

        X = subset.pivot_table(
            index="userId",
            columns="movieId",
            values="rating_diff",
            fill_value=0.0
        )

        # ----------------------------------------------------
        # TRUE PCA (covariance + eigen-decomposition)
        # ----------------------------------------------------

        # Covariance matrix (items × items)
        C = np.cov(X.values, rowvar=False)

        # Eigen-decomposition (symmetric matrix)
        eigvals, eigvecs = np.linalg.eigh(C)

        # Sort eigenvalues descending
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Select top principal components
        k = min(max_components, eigvecs.shape[1])
        W = eigvecs[:, :k]

        # Project users into PCA space
        Z = X.values @ W

        latent_df = pd.DataFrame(
            Z,
            index=X.index,
            columns=[f"PC{i+1}" for i in range(k)]
        ).reset_index()

        latent_df["target_item"] = target
        spaces.append(latent_df)

    return pd.concat(spaces, ignore_index=True)


# ============================================================
# STEP 8: Reduced space (Top-5)
# ============================================================

reduced_top5 = reduced_space(peers_df, "Top-5", 2)
reduced_top5.to_csv(
    f"{PCA_TABLE_DIR}/reduced_space_users_top5.csv",
    index=False
)

# ============================================================
# STEP 9: Predictions (Top-5)
# ============================================================

ratings_original = ratings[["userId", "movieId"]]

def predict(latent_df, label):
    preds = []

    for target in latent_df["target_item"].unique():
        latent_sub = latent_df[latent_df["target_item"] == target]
        pc_cols = [c for c in latent_sub.columns if c.startswith("PC")]

        rated_users = set(
            ratings_original[ratings_original["movieId"] == target]["userId"]
        )

        item_mean = mean_filled_df[
            mean_filled_df["movieId"] == target
        ]["rating"].mean()

        for _, row in latent_sub.iterrows():
            if row["userId"] in rated_users:
                continue

            value = item_mean + row[pc_cols].sum()
            preds.append({
                "userId": row["userId"],
                "target_item": target,
                f"predicted_rating_{label}": float(value)
            })

    df = pd.DataFrame(preds)
    df[f"predicted_rating_{label}"] = df[f"predicted_rating_{label}"].clip(1, 5)
    return df

pred_top5 = predict(reduced_top5, "top5")
pred_top5.to_csv(
    f"{PCA_TABLE_DIR}/predicted_ratings_top5.csv",
    index=False
)

# ============================================================
# STEP 10: Reduced space (Top-10)
# ============================================================

reduced_top10 = reduced_space(peers_df, "Top-10", 3)
reduced_top10.to_csv(
    f"{PCA_TABLE_DIR}/reduced_space_users_top10.csv",
    index=False
)

# ============================================================
# STEP 11: Predictions (Top-10)
# ============================================================

pred_top10 = predict(reduced_top10, "top10")
pred_top10.to_csv(
    f"{PCA_TABLE_DIR}/predicted_ratings_top10.csv",
    index=False
)

# ============================================================
# STEP 12: Comparison
# ============================================================

comparison = pred_top5.merge(
    pred_top10,
    on=["userId", "target_item"],
    how="inner"
)

comparison["absolute_difference"] = (
    comparison["predicted_rating_top5"]
    - comparison["predicted_rating_top10"]
).abs()

comparison.to_csv(
    f"{PCA_TABLE_DIR}/comparison_top5_vs_top10.csv",
    index=False
)

# ============================================================
# STEP 13: VISUALIZATION
# ============================================================

plt.figure(figsize=(8,4))
sns.histplot(cov_all["covariance"], bins=50, kde=True)
plt.title("Distribution of Covariance Values")
plt.savefig(f"{PCA_PLOT_DIR}/covariance_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(cov_matrix, cmap="coolwarm", center=0)
plt.title("Covariance Matrix Heatmap (Target Items)")
plt.savefig(f"{PCA_PLOT_DIR}/covariance_heatmap.png")
plt.close()

plt.figure(figsize=(6,5))
sns.scatterplot(
    data=reduced_top5,
    x="PC1",
    y="PC2",
    hue="target_item",
    alpha=0.6
)
plt.title("Reduced User Space (Top-5 Peers)")
plt.savefig(f"{PCA_PLOT_DIR}/pca_space_top5.png")
plt.close()

if "PC3" in reduced_top10.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        data=reduced_top10,
        x="PC1",
        y="PC2",
        hue="target_item",
        alpha=0.6
    )
    plt.title("Reduced User Space (Top-10 Peers)")
    plt.savefig(f"{PCA_PLOT_DIR}/pca_space_top10.png")
    plt.close()

plt.figure(figsize=(7,4))
sns.kdeplot(pred_top5["predicted_rating_top5"], label="Top-5", fill=True)
sns.kdeplot(pred_top10["predicted_rating_top10"], label="Top-10", fill=True)
plt.legend()
plt.title("Prediction Distribution Comparison")
plt.savefig(f"{PCA_PLOT_DIR}/prediction_distribution_comparison.png")
plt.close()

plt.figure(figsize=(5,5))
sns.scatterplot(
    x=comparison["predicted_rating_top5"],
    y=comparison["predicted_rating_top10"],
    alpha=0.4
)
plt.plot([1,5], [1,5], linestyle="--", color="red")
plt.xlabel("Top-5 Prediction")
plt.ylabel("Top-10 Prediction")
plt.title("Top-5 vs Top-10 Predictions")
plt.savefig(f"{PCA_PLOT_DIR}/prediction_scatter_top5_vs_top10.png")
plt.close()

plt.figure(figsize=(7,4))
sns.histplot(comparison["absolute_difference"], bins=40)
plt.title("Absolute Difference |Top-5 − Top-10|")
plt.savefig(f"{PCA_PLOT_DIR}/absolute_prediction_difference.png")
plt.close()

plt.figure(figsize=(7,4))
sns.boxplot(
    data=comparison,
    x="target_item",
    y="absolute_difference"
)
plt.title("Prediction Difference per Target Item")
plt.savefig(f"{PCA_PLOT_DIR}/difference_per_target_item.png")
plt.close()

print("Part 1 (Steps 1–13) completed successfully.")
