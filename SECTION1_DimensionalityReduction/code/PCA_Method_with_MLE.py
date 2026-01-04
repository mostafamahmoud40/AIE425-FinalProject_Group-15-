# ============================================================
# Part 2: PCA Method with Maximum Likelihood Estimation (MLE)
# FULL & COMPLETE: Steps 1–10 + ALL visualizations
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
MEAN_TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/PCA Method with Mean-Filling"
MLE_TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/PCA Method with MLE"
MLE_PLOT_DIR = "SECTION1_DimensionalityReduction/results/plots/PCA Method with MLE"

os.makedirs(MLE_TABLE_DIR, exist_ok=True)
os.makedirs(MLE_PLOT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# ============================================================
# LOAD DATA 
# ============================================================

ratings = pd.read_csv(
    f"{DATA_DIR}/ratings.csv",
    usecols=["userId", "movieId", "rating"],
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)

target_items = pd.read_csv(f"{STAT_TABLE_DIR}/target_items.csv")
I1, I2 = target_items["movieId"].tolist()

# ============================================================
# STEP 1: MLE COVARIANCE 
# ============================================================

item_means = ratings.groupby("movieId")["rating"].mean()

def mle_covariance(target_item):
    target = ratings[ratings["movieId"] == target_item][
        ["userId", "rating"]
    ].rename(columns={"rating": "r_target"})

    joined = ratings.merge(target, on="userId", how="inner")
    joined = joined[joined["movieId"] != target_item]

    joined["diff_i"] = joined["rating"] - joined["movieId"].map(item_means)
    joined["diff_j"] = joined["r_target"] - item_means[target_item]
    joined["prod"] = joined["diff_i"] * joined["diff_j"]

    cov = joined.groupby("movieId", as_index=False)["prod"].mean()
    cov.rename(columns={"prod": "covariance"}, inplace=True)
    cov["target_item"] = target_item
    return cov

cov_all = pd.concat(
    [mle_covariance(I1), mle_covariance(I2)],
    ignore_index=True
)

cov_matrix = (
    cov_all
    .pivot(index="movieId", columns="target_item", values="covariance")
    .fillna(0.0)
)

cov_matrix.columns = [f"cov_with_{c}" for c in cov_matrix.columns]

cov_all.to_csv(f"{MLE_TABLE_DIR}/item_covariance_target_items_mle.csv", index=False)
cov_matrix.to_csv(f"{MLE_TABLE_DIR}/covariance_matrix_target_items_mle.csv")

# ============================================================
# STEP 2: TOP-5 & TOP-10 PEERS
# ============================================================

cov_all["abs_cov"] = cov_all["covariance"].abs()

peer_rows = []
for t in cov_all["target_item"].unique():
    s = cov_all[cov_all["target_item"] == t].sort_values("abs_cov", ascending=False)
    peer_rows.append(s.head(5).assign(peer_group="Top-5"))
    peer_rows.append(s.head(10).assign(peer_group="Top-10"))

peers_mle = pd.concat(peer_rows, ignore_index=True)
peers_mle.to_csv(f"{MLE_TABLE_DIR}/top5_top10_peers_mle.csv", index=False)

# ============================================================
# REDUCED SPACE FUNCTION (TRUE PCA)
# ============================================================

def reduced_space(peers, label, max_components):
    blocks = []

    for t in peers["target_item"].unique():

        # Select peer items
        items = peers[
            (peers["target_item"] == t) &
            (peers["peer_group"] == label)
        ]["movieId"].tolist()

        # Build user–item matrix (MLE: specified entries only)
        sub = ratings[ratings["movieId"].isin(items)].copy()
        sub["rating_diff"] = sub["rating"] - sub["movieId"].map(item_means)

        X = sub.pivot_table(
            index="userId",
            columns="movieId",
            values="rating_diff",
            fill_value=0.0
        )

        # ----------------------------------------------------
        # TRUE PCA (covariance + eigen-decomposition)
        # ----------------------------------------------------

        # Item–item covariance matrix
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

        latent_df["target_item"] = t
        blocks.append(latent_df)

    return pd.concat(blocks, ignore_index=True)


# ============================================================
# STEP 3 & 4: TOP-5 SPACE + PREDICTION
# ============================================================

latent_top5 = reduced_space(peers_mle, "Top-5", 2)
latent_top5.to_csv(f"{MLE_TABLE_DIR}/reduced_space_users_top5_mle.csv", index=False)

def predict(latent, label):
    preds = []
    rated_pairs = ratings[["userId", "movieId"]]

    for t in latent["target_item"].unique():
        L = latent[latent["target_item"] == t]
        pcs = [c for c in L.columns if c.startswith("PC")]
        rated_users = set(rated_pairs[rated_pairs["movieId"] == t]["userId"])

        for _, r in L.iterrows():
            if r["userId"] in rated_users:
                continue
            value = item_means[t] + r[pcs].sum()
            preds.append({
                "userId": r["userId"],
                "target_item": t,
                f"predicted_rating_{label}": float(value)
            })

    df = pd.DataFrame(preds)
    df[f"predicted_rating_{label}"] = df[f"predicted_rating_{label}"].clip(1, 5)
    return df

pred_top5 = predict(latent_top5, "top5_mle")
pred_top5.to_csv(f"{MLE_TABLE_DIR}/predicted_ratings_top5_mle.csv", index=False)

# ============================================================
# STEP 5 & 6: TOP-10 SPACE + PREDICTION
# ============================================================

latent_top10 = reduced_space(peers_mle, "Top-10", 3)
latent_top10.to_csv(f"{MLE_TABLE_DIR}/reduced_space_users_top10_mle.csv", index=False)

pred_top10 = predict(latent_top10, "top10_mle")
pred_top10.to_csv(f"{MLE_TABLE_DIR}/predicted_ratings_top10_mle.csv", index=False)

# ============================================================
# STEP 7–9: COMPARISONS
# ============================================================

mean_top5 = pd.read_csv(f"{MEAN_TABLE_DIR}/predicted_ratings_top5.csv")
mean_top10 = pd.read_csv(f"{MEAN_TABLE_DIR}/predicted_ratings_top10.csv")

comp_5_10 = pred_top5.merge(pred_top10, on=["userId", "target_item"])
comp_5_10["absolute_difference"] = (
    comp_5_10["predicted_rating_top5_mle"] -
    comp_5_10["predicted_rating_top10_mle"]
).abs()
comp_5_10.to_csv(f"{MLE_TABLE_DIR}/comparison_top5_vs_top10_mle.csv", index=False)

comp_mean_mle_5 = mean_top5.merge(pred_top5, on=["userId", "target_item"])
comp_mean_mle_10 = mean_top10.merge(pred_top10, on=["userId", "target_item"])

comp_mean_mle_5.to_csv(f"{MLE_TABLE_DIR}/comparison_mean_filling_vs_mle_top5.csv", index=False)
comp_mean_mle_10.to_csv(f"{MLE_TABLE_DIR}/comparison_mean_filling_vs_mle_top10.csv", index=False)

# ============================================================
# STEP 10: ALL VISUALIZATIONS (EXPLICITLY SAVED)
# ============================================================

plt.figure(figsize=(8,4))
sns.histplot(cov_all["covariance"], bins=50, kde=True)
plt.title("MLE Covariance Distribution")
plt.savefig(f"{MLE_PLOT_DIR}/covariance_distribution_mle.png")
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(cov_matrix, cmap="coolwarm", center=0)
plt.title("MLE Covariance Heatmap")
plt.savefig(f"{MLE_PLOT_DIR}/covariance_heatmap_mle.png")
plt.close()

plt.figure(figsize=(6,5))
sns.scatterplot(data=latent_top5, x="PC1", y="PC2", hue="target_item", alpha=0.6)
plt.title("PCA Space – Top-5 (MLE)")
plt.savefig(f"{MLE_PLOT_DIR}/pca_space_top5_mle.png")
plt.close()

plt.figure(figsize=(6,5))
sns.scatterplot(data=latent_top10, x="PC1", y="PC2", hue="target_item", alpha=0.6)
plt.title("PCA Space – Top-10 (MLE)")
plt.savefig(f"{MLE_PLOT_DIR}/pca_space_top10_mle.png")
plt.close()

plt.figure(figsize=(7,4))
sns.kdeplot(pred_top5["predicted_rating_top5_mle"], label="Top-5", fill=True)
sns.kdeplot(pred_top10["predicted_rating_top10_mle"], label="Top-10", fill=True)
plt.legend()
plt.title("Prediction Distribution (MLE)")
plt.savefig(f"{MLE_PLOT_DIR}/prediction_distribution_mle.png")
plt.close()

plt.figure(figsize=(5,5))
sns.scatterplot(
    x=comp_5_10["predicted_rating_top5_mle"],
    y=comp_5_10["predicted_rating_top10_mle"],
    alpha=0.4
)
plt.plot([1,5], [1,5], "--", color="red")
plt.title("Top-5 vs Top-10 (MLE)")
plt.savefig(f"{MLE_PLOT_DIR}/prediction_scatter_top5_vs_top10_mle.png")
plt.close()

plt.figure(figsize=(7,4))
sns.histplot(comp_5_10["absolute_difference"], bins=40)
plt.title("Absolute Difference |Top-5 − Top-10| (MLE)")
plt.savefig(f"{MLE_PLOT_DIR}/absolute_difference_top5_vs_top10_mle.png")
plt.close()

plt.figure(figsize=(5,5))
sns.scatterplot(
    x=comp_mean_mle_5["predicted_rating_top5"],
    y=comp_mean_mle_5["predicted_rating_top5_mle"],
    alpha=0.4
)
plt.plot([1,5], [1,5], "--", color="red")
plt.title("Mean-Filling vs MLE (Top-5)")
plt.savefig(f"{MLE_PLOT_DIR}/mean_vs_mle_top5.png")
plt.close()

plt.figure(figsize=(5,5))
sns.scatterplot(
    x=comp_mean_mle_10["predicted_rating_top10"],
    y=comp_mean_mle_10["predicted_rating_top10_mle"],
    alpha=0.4
)
plt.plot([1,5], [1,5], "--", color="red")
plt.title("Mean-Filling vs MLE (Top-10)")
plt.savefig(f"{MLE_PLOT_DIR}/mean_vs_mle_top10.png")
plt.close()

print("Part 2 COMPLETE: ALL tables and ALL plots saved successfully.")
