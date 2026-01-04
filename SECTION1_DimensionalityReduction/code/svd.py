# =========================================================
# Part 3: SVD for Collaborative Filtering
# Clean submission version
# =========================================================

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

# =========================================================
# Paths
# =========================================================
DATA_DIR = "SECTION1_DimensionalityReduction/data"
TABLE_DIR = "SECTION1_DimensionalityReduction/results/tables/svd"
PLOT_DIR  = "SECTION1_DimensionalityReduction/results/plots/svd"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# 1. Data Preparation
# =========================================================

ratings_df = pd.read_csv(f"{DATA_DIR}/ratings.csv")
movies_df  = pd.read_csv(f"{DATA_DIR}/movies.csv")

TOP_USERS = 8000
TOP_ITEMS = 5000

top_users = (
    ratings_df.groupby("userId")
    .size()
    .sort_values(ascending=False)
    .head(TOP_USERS)
    .index
)

top_items = (
    ratings_df.groupby("movieId")
    .size()
    .sort_values(ascending=False)
    .head(TOP_ITEMS)
    .index
)

ratings_df = ratings_df[
    ratings_df["userId"].isin(top_users) &
    ratings_df["movieId"].isin(top_items)
].copy()

user_ids = ratings_df["userId"].unique()
item_ids = ratings_df["movieId"].unique()

user_to_index = {u: i for i, u in enumerate(user_ids)}
item_to_index = {m: i for i, m in enumerate(item_ids)}
index_to_user = {i: u for u, i in user_to_index.items()}
index_to_item = {i: m for m, i in item_to_index.items()}

n_users = len(user_ids)
n_items = len(item_ids)

R = np.full((n_users, n_items), np.nan, dtype=np.float32)

for _, r in ratings_df.iterrows():
    R[user_to_index[r.userId], item_to_index[r.movieId]] = r.rating

item_means = np.nanmean(R, axis=0)

R_filled = R.copy()
mask = np.isnan(R_filled)
R_filled[mask] = np.take(item_means, np.where(mask)[1])

pd.DataFrame({
    "missing_after_fill": [int(np.isnan(R_filled).sum())]
}).to_csv(f"{TABLE_DIR}/matrix_completeness.csv", index=False)

R_full = R_filled

# =========================================================
# 2. Full SVD Decomposition
# =========================================================

t0 = time.time()
U, s, VT = np.linalg.svd(R_full, full_matrices=False)
svd_time = time.time() - t0

V = VT.T

pd.DataFrame({
    "sigma": s,
    "lambda": s**2
}).to_csv(f"{TABLE_DIR}/eigenpairs.csv", index=False)

pd.DataFrame({
    "UTU_error": [np.linalg.norm(U.T @ U - np.eye(U.shape[1]))],
    "VTV_error": [np.linalg.norm(V.T @ V - np.eye(V.shape[1]))],
    "svd_time_sec": [svd_time]
}).to_csv(f"{TABLE_DIR}/orthogonality_check.csv", index=False)

explained_var = (s**2) / np.sum(s**2)
cum_var = np.cumsum(explained_var)

plt.figure()
plt.plot(s)
plt.title("Singular Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid()
plt.savefig(f"{PLOT_DIR}/singular_values.png")
plt.close()

plt.figure()
plt.plot(cum_var)
plt.title("Scree Plot")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.savefig(f"{PLOT_DIR}/scree_plot.png")
plt.close()

# =========================================================
# 3. Truncated SVD
# =========================================================

def mae_rmse(A, B):
    return (
        float(np.mean(np.abs(A - B))),
        float(np.sqrt(np.mean((A - B) ** 2)))
    )

k_values = [5, 20, 50, 100]
k_values = [k for k in k_values if k <= min(n_users, n_items)]

rows = []

for k in k_values:
    Uk = U[:, :k]
    Sk = np.diag(s[:k])
    Vk = V[:, :k]
    R_hat = Uk @ Sk @ Vk.T
    mae, rmse = mae_rmse(R_full, R_hat)

    rows.append({"k": k, "MAE": mae, "RMSE": rmse})

err_df = pd.DataFrame(rows)
err_df.to_csv(f"{TABLE_DIR}/truncated_svd_errors.csv", index=False)

plt.figure()
plt.plot(err_df["k"], err_df["RMSE"], marker="o")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.title("Elbow Curve")
plt.grid()
plt.savefig(f"{PLOT_DIR}/rmse_vs_k.png")
plt.close()

k_opt = 20 if 20 in k_values else k_values[0]

# =========================================================
# 4. Rating Prediction
# =========================================================

Uk = U[:, :k_opt]
Sk = np.diag(s[:k_opt])
Vk = V[:, :k_opt]

target_users = (
    ratings_df.groupby("userId")
    .size()
    .sort_values(ascending=False)
    .head(3)
    .index
)

target_items = (
    ratings_df.groupby("movieId")
    .size()
    .sort_values(ascending=False)
    .head(2)
    .index
)

predictions = []

for u in target_users:
    for i in target_items:
        r_hat = (
            Uk[user_to_index[u]]
            @ Sk
            @ Vk[item_to_index[i]]
        )
        predictions.append({
            "userId": u,
            "movieId": i,
            "predicted_rating": float(r_hat)
        })

pd.DataFrame(predictions).to_csv(
    f"{TABLE_DIR}/svd_predictions.csv", index=False
)

# =========================================================
# 5. Comparative Analysis (SVD vs PCA)
# =========================================================

pd.DataFrame([{
    "Method": "SVD",
    "k": k_opt,
    "RMSE": err_df.loc[err_df.k == k_opt, "RMSE"].values[0],
    "Runtime_sec": svd_time,
    "Memory_MB": R_full.nbytes / (1024**2)
}]).to_csv(
    f"{TABLE_DIR}/method_comparison.csv", index=False
)

# =========================================================
# 6. Latent Factor Interpretation
# =========================================================

latent_rows = []

for f in range(3):
    for idx in np.argsort(np.abs(U[:, f]))[-10:]:
        latent_rows.append({
            "factor": f + 1,
            "type": "user",
            "id": index_to_user[idx],
            "loading": float(U[idx, f])
        })

    for idx in np.argsort(np.abs(V[:, f]))[-10:]:
        latent_rows.append({
            "factor": f + 1,
            "type": "item",
            "id": index_to_item[idx],
            "loading": float(V[idx, f])
        })

pd.DataFrame(latent_rows).to_csv(
    f"{TABLE_DIR}/latent_factors.csv", index=False
)

# =========================================================
# 7. Sensitivity Analysis
# =========================================================

def item_mean_fill(X):
    Y = X.copy()
    means = np.nanmean(Y, axis=0)
    mask = np.isnan(Y)
    Y[mask] = np.take(means, np.where(mask)[1])
    return Y

sens_rows = []

for pct in [0.1, 0.3, 0.5, 0.7]:
    X = R.copy()
    obs = np.argwhere(~np.isnan(X))
    drop = obs[np.random.choice(len(obs), int(len(obs)*pct), False)]
    for i, j in drop:
        X[i, j] = np.nan

    Xf = item_mean_fill(X)
    Ux, sx, VTx = np.linalg.svd(Xf, full_matrices=False)
    Xhat = Ux[:, :k_opt] @ np.diag(sx[:k_opt]) @ VTx[:k_opt]
    _, rmse = mae_rmse(R_full, Xhat)

    sens_rows.append({
        "missing_pct": int(pct * 100),
        "RMSE": rmse
    })

pd.DataFrame(sens_rows).to_csv(
    f"{TABLE_DIR}/sensitivity_missingness.csv", index=False
)

# =========================================================
# 8. Cold-Start Analysis
# =========================================================

eligible = ratings_df.groupby("userId").size()
eligible = eligible[eligible > 20].index.tolist()
cold_users = random.sample(eligible, min(50, len(eligible)))

rows = []

for u in cold_users:
    ur = ratings_df[ratings_df.userId == u]
    hidden = ur.sample(frac=0.8, random_state=42)
    visible = ur.drop(hidden.index)

    if len(visible) < 2:
        continue

    A, b = [], []
    for _, r in visible.iterrows():
        A.append(Sk @ Vk[item_to_index[r.movieId]])
        b.append(r.rating)

    u_hat, *_ = np.linalg.lstsq(np.vstack(A), np.array(b), rcond=None)

    for _, r in hidden.iterrows():
        pred = u_hat @ Sk @ Vk[item_to_index[r.movieId]]
        rows.append({
            "true": r.rating,
            "pred": float(pred)
        })

cold_df = pd.DataFrame(rows)

pd.DataFrame({
    "MAE": [np.mean(np.abs(cold_df.true - cold_df.pred))],
    "RMSE": [np.sqrt(np.mean((cold_df.true - cold_df.pred)**2))]
}).to_csv(
    f"{TABLE_DIR}/cold_start_results.csv", index=False
)

print("Pipeline completed successfully.")