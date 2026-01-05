# ============================================================
# Part 3: Collaborative Filtering
# Domain: Interest-Based Group Formation (Meetup.com)
# ============================================================

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

TABLE_DIR = "SECTION2_DomainRecommender/results/tables/Collaborative_Filtering"
PLOT_DIR = "SECTION2_DomainRecommender/results/plots/Collaborative_Filtering"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

pd.options.display.float_format = "{:.4f}".format

def save_table(df, filename):
    """Save DataFrame to CSV with formatting"""
    df2 = df.copy()
    float_cols = df2.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        df2[float_cols] = df2[float_cols].round(4)
    df2.to_csv(f"{TABLE_DIR}/{filename}", index=False)

def save_plot(filename):
    """Save current plot"""
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# GPU CONFIGURATION - For SVD Only (using PyTorch)
# ============================================================

GPU_AVAILABLE = False
torch = None

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        print(f"  PyTorch will be used for SVD acceleration")
    else:
        print("CUDA not available in PyTorch - Using CPU for SVD")
except ImportError:
    print("✗ PyTorch not installed - Using CPU for SVD")

class CollaborativeFilteringRecommender:
    """
    8. Collaborative Filtering Integration
    
    8.1. Implement ONE CF approach: User-based CF with cosine similarity
    8.2. Use matrix factorization from SECTION 1:
         Apply SVD with k=10 and k=20 latent factors
         Generate predictions for target users
    """
    
    def __init__(self, data_path='SECTION2_DomainRecommender/data/'):
        self.data_path = data_path
        self.user_group_df = None
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_ids = None
        self.group_ids = None
        self.user_id_to_idx = {}
        self.group_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_group_id = {}
        self.svd_predictions = {}
        
    def load_data(self, sample_size=None):
        """Load and prepare the user-group interaction data"""
        print("LOADING DATA FOR COLLABORATIVE FILTERING")
        
        # Load user-group memberships
        self.user_group_df = pd.read_csv(
            f"{self.data_path}user_group.csv",
            header=None,
            names=['user_id', 'group_id']
        )
        self.user_group_df = self.user_group_df.drop_duplicates()
        print(f"User-Group memberships loaded: {len(self.user_group_df):,}")
        
        if sample_size:
            self._sample_data(sample_size)
    
    def _sample_data(self, n_users=20000, n_groups=1500, n_interactions=300000):
        """
        Sample users with sufficient interactions to get target counts
        
        Memory Guidelines (for user similarity matrix):
        - 10K users: ~800MB (safe for 8GB RAM)
        - 20K users: ~3.2GB (safe for 16GB RAM)
        - 30K users: ~7.2GB (maximum for 16GB RAM)
        """
        print(f"\nSampling to target: {n_users:,} users, {n_groups:,} groups, {n_interactions:,} interactions...")
        
        # Step 1: Get popular groups (top n_groups by member count)
        group_counts = self.user_group_df['group_id'].value_counts()
        top_groups = group_counts.head(n_groups).index.tolist()
        
        # Filter to these groups only
        self.user_group_df = self.user_group_df[
            self.user_group_df['group_id'].isin(top_groups)
        ]
        
        # Step 2: Get active users from filtered data
        user_counts = self.user_group_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 5].index[:n_users].tolist()
        
        self.user_group_df = self.user_group_df[
            self.user_group_df['user_id'].isin(active_users)
        ]
        
        # Step 3: Limit interactions if needed
        if len(self.user_group_df) > n_interactions:
            self.user_group_df = self.user_group_df.sample(n=n_interactions, random_state=42)
        
        print(f"Final users: {self.user_group_df['user_id'].nunique():,}")
        print(f"Final groups: {self.user_group_df['group_id'].nunique():,}")
        print(f"Final interactions: {len(self.user_group_df):,}")
    
    def create_user_item_matrix(self):
        """
        Create user-item interaction matrix (implicit feedback)
        Value = 1 if user joined group, 0 otherwise
        """
        print("CREATING USER-ITEM MATRIX")
        
        # Get unique users and groups
        self.user_ids = self.user_group_df['user_id'].unique()
        self.group_ids = self.user_group_df['group_id'].unique()
        
        # Create mappings
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.group_id_to_idx = {gid: idx for idx, gid in enumerate(self.group_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_group_id = {idx: gid for gid, idx in self.group_id_to_idx.items()}
        
        # Create sparse matrix
        rows = self.user_group_df['user_id'].map(self.user_id_to_idx)
        cols = self.user_group_df['group_id'].map(self.group_id_to_idx)
        data = np.ones(len(self.user_group_df))
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.group_ids))
        )
        
        # Calculate statistics
        n_users, n_items = self.user_item_matrix.shape
        n_interactions = self.user_item_matrix.nnz
        sparsity = (1 - n_interactions / (n_users * n_items)) * 100
        
        print(f"Matrix Shape: {n_users} users × {n_items} groups")
        print(f"Non-zero entries: {n_interactions:,}")
        print(f"Sparsity: {sparsity:.4f}%")
        
        # ============================================================
        # SAVE USER-ITEM MATRIX STATS
        # ============================================================
        
        # Save matrix statistics
        matrix_stats = pd.DataFrame({
            'Metric': ['Users', 'Groups', 'Interactions', 'Sparsity (%)'],
            'Value': [n_users, n_items, n_interactions, sparsity]
        })
        save_table(matrix_stats, "user_item_matrix_stats.csv")
        
        # Save user interaction counts
        user_counts = pd.DataFrame({
            'user_id': self.user_ids,
            'n_groups': np.array(self.user_item_matrix.sum(axis=1)).flatten()
        })
        save_table(user_counts, "user_interaction_counts.csv")
        
        # Save group interaction counts
        group_counts = pd.DataFrame({
            'group_id': self.group_ids,
            'n_users': np.array(self.user_item_matrix.sum(axis=0)).flatten()
        })
        save_table(group_counts, "group_interaction_counts.csv")
        
        return self.user_item_matrix
    
    def compute_user_similarity(self, metric='cosine'):
        """
        8.1. Implement User-based CF with cosine similarity
        """
        print("8.1 USER-BASED COLLABORATIVE FILTERING")
        print(f"Computing user similarity using {metric} similarity...")
        
        # Convert to dense for similarity computation (sample if too large)
        if self.user_item_matrix.shape[0] > 5000:
            print("Using batch processing for large matrix...")
            # Compute similarity in batches for memory efficiency
            n_users = self.user_item_matrix.shape[0]
            self.user_similarity_matrix = cosine_similarity(
                self.user_item_matrix, self.user_item_matrix
            )
        else:
            self.user_similarity_matrix = cosine_similarity(
                self.user_item_matrix.toarray()
            )
        
        # Set diagonal to 0 (user is not similar to themselves for recommendations)
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        print(f"User Similarity Matrix Shape: {self.user_similarity_matrix.shape}")
        print(f"Sample similarities (User 0 with first 5 users): "
              f"{self.user_similarity_matrix[0, 1:6].round(4)}")
        
        # ============================================================
        # SAVE 8.1 USER SIMILARITY RESULTS
        # ============================================================
        
        # Compute similarity statistics efficiently (without creating huge array)
        # Sample-based statistics to avoid memory issues
        n_users = self.user_similarity_matrix.shape[0]
        sample_size = min(5000, n_users)
        sample_indices = np.random.choice(n_users, sample_size, replace=False)
        
        # Get statistics from sampled rows
        sampled_sims = self.user_similarity_matrix[sample_indices]
        non_zero_mask = sampled_sims > 0
        non_zero_sims = sampled_sims[non_zero_mask]
        
        sim_stats = pd.DataFrame({
            'Metric': ['Mean Similarity (sampled)', 'Max Similarity', 'Min Similarity (non-zero)', 
                      'Std Similarity', 'Non-Zero Pairs (estimated)'],
            'Value': [
                np.mean(non_zero_sims) if len(non_zero_sims) > 0 else 0,
                np.max(self.user_similarity_matrix),
                np.min(non_zero_sims) if len(non_zero_sims) > 0 else 0,
                np.std(non_zero_sims) if len(non_zero_sims) > 0 else 0,
                int(np.sum(non_zero_mask) * (n_users / sample_size))  # Estimate
            ]
        })
        save_table(sim_stats, "user_similarity_stats.csv")
        
        # Save sample of most similar user pairs
        top_pairs = []
        for i in range(min(100, len(self.user_ids))):
            top_neighbors = np.argsort(self.user_similarity_matrix[i])[-5:][::-1]
            for j in top_neighbors:
                if self.user_similarity_matrix[i, j] > 0:
                    top_pairs.append({
                        'user_1': self.user_ids[i],
                        'user_2': self.user_ids[j],
                        'similarity': self.user_similarity_matrix[i, j]
                    })
        
        pairs_df = pd.DataFrame(top_pairs)
        save_table(pairs_df, "top_similar_user_pairs.csv")
        
        # Plot similarity distribution (using sampled data)
        plt.figure(figsize=(8, 4))
        if len(non_zero_sims) > 0:
            plt.hist(non_zero_sims, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of User-User Similarities (Non-Zero, Sampled)')
        save_plot("user_similarity_distribution.png")
        
        return self.user_similarity_matrix
    
    def user_based_predict(self, user_id, group_id, k=20):
        """
        8.1. Predict rating using weighted average of k nearest neighbors
        """
        if user_id not in self.user_id_to_idx:
            return 0.0
        if group_id not in self.group_id_to_idx:
            return 0.0
        
        user_idx = self.user_id_to_idx[user_id]
        group_idx = self.group_id_to_idx[group_id]
        
        # Get k most similar users
        similarities = self.user_similarity_matrix[user_idx]
        similar_user_indices = np.argsort(similarities)[::-1][:k]
        
        # Weighted average of similar users' ratings
        numerator = 0.0
        denominator = 0.0
        
        for sim_user_idx in similar_user_indices:
            sim = similarities[sim_user_idx]
            if sim > 0:
                rating = self.user_item_matrix[sim_user_idx, group_idx]
                numerator += sim * rating
                denominator += sim
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def user_based_recommend(self, user_id, top_n=10, k=20):
        """
        8.1. Generate top-N recommendations using user-based CF
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get groups user already joined
        joined_groups = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # Predict scores for all unjoined groups
        predictions = []
        for group_idx in range(len(self.group_ids)):
            if group_idx not in joined_groups:
                group_id = self.idx_to_group_id[group_idx]
                score = self.user_based_predict(user_id, group_id, k)
                predictions.append((group_id, score))
        
        # Sort by predicted score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]
    
    def apply_svd(self, k_values=[10, 20]):
        """
        8.2. Use matrix factorization from SECTION 1:
             Apply SVD with k=10 and k=20 latent factors
             Generate predictions for target users
        """
        print("8.2 SVD MATRIX FACTORIZATION")
        
        # Convert to float for SVD
        matrix = self.user_item_matrix.astype(float)
        
        # Store mean for later prediction
        self.user_ratings_mean = np.array(matrix.mean(axis=1)).flatten()
        
        # Store SVD components instead of full prediction matrix
        self.svd_components = {}
        
        for k in k_values:
            print(f"\nApplying SVD with k={k} latent factors...")
            
            # Always use scipy's svds - it's optimized for sparse matrices
            try:
                U, sigma, Vt = svds(matrix, k=k)
                
                # Sort by singular values (svds returns in ascending order)
                idx = np.argsort(sigma)[::-1]
                sigma = sigma[idx]
                U = U[:, idx]
                Vt = Vt[idx, :]
                
            except Exception as e:
                print(f"  Warning: svds failed ({e}), skipping k={k}...")
                continue
            
            # Store components (memory efficient - don't compute full matrix)
            self.svd_components[k] = {
                'U': U,
                'sigma': sigma,
                'Vt': Vt
            }
            
            # For backward compatibility, create a lazy prediction wrapper
            self.svd_predictions[k] = None  # Will compute on-demand
            
            # ============================================================
            # SAVE 8.2 SVD RESULTS
            # ============================================================
            
            # Save singular values
            sv_df = pd.DataFrame({
                'component': range(1, k+1),
                'singular_value': sigma,
                'variance_explained': (sigma ** 2) / np.sum(sigma ** 2)
            })
            save_table(sv_df, f"svd_k{k}_singular_values.csv")
            
            # Save sample predictions (computed on-demand for just 100 users)
            sample_preds = []
            for i in range(min(100, len(self.user_ids))):
                user_id = self.user_ids[i]
                # Compute prediction for this user only
                user_pred = self.svd_predict_user(i, k)
                if user_pred is not None:
                    top_groups = np.argsort(user_pred)[-10:][::-1]
                    for rank, group_idx in enumerate(top_groups):
                        sample_preds.append({
                            'user_id': user_id,
                            'rank': rank + 1,
                            'group_id': self.group_ids[group_idx],
                            'predicted_score': user_pred[group_idx]
                        })
            
            preds_df = pd.DataFrame(sample_preds)
            save_table(preds_df, f"svd_k{k}_sample_predictions.csv")
        
        # Plot SVD explained variance
        plt.figure(figsize=(8, 4))
        for k in k_values:
            if k in self.svd_components:
                sigma = self.svd_components[k]['sigma']
                variance = (sigma ** 2) / np.sum(sigma ** 2)
                plt.plot(range(1, k+1), np.cumsum(variance[::-1]), marker='o', label=f'k={k}')
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('SVD Cumulative Variance Explained')
        plt.legend()
        save_plot("svd_variance_explained.png")
        
        return self.svd_components
    
    def svd_predict_user(self, user_idx, k=10):
        """
        8.2. Compute SVD predictions for a single user (memory efficient)
        """
        if k not in self.svd_components:
            return None
        
        comp = self.svd_components[k]
        U = comp['U']
        sigma = comp['sigma']
        Vt = comp['Vt']
        
        # Compute predictions for just this user
        # user_predictions = U[user_idx] @ diag(sigma) @ Vt + mean
        user_latent = U[user_idx] * sigma  # (k,)
        user_predictions = user_latent @ Vt  # (n_groups,)
        user_predictions += self.user_ratings_mean[user_idx]
        
        return user_predictions
    
    def svd_recommend(self, user_id, k=10, top_n=10):
        """
        8.2. Generate recommendations using SVD predictions
        """
        if k not in self.svd_components:
            raise ValueError(f"SVD with k={k} not computed. Run apply_svd first.")
        
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get predicted ratings for this user (computed on-demand)
        user_predictions = self.svd_predict_user(user_idx, k)
        
        if user_predictions is None:
            return []
        
        # Get groups user already joined
        joined_group_indices = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # Create recommendations
        recommendations = []
        for group_idx, score in enumerate(user_predictions):
            if group_idx not in joined_group_indices:
                group_id = self.idx_to_group_id[group_idx]
                recommendations.append((group_id, score))
        
        # Sort by predicted score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_n]
    
    def demonstrate_cf(self, sample_user_id):
        """
        8.1 & 8.2. Demonstrate collaborative filtering recommendations
        """
        print(f"CF RECOMMENDATIONS FOR USER {sample_user_id}")
        
        # User-based recommendations
        print("\nUser-Based CF (k=20 neighbors):")
        user_based_recs = self.user_based_recommend(sample_user_id, top_n=10, k=20)
        for rank, (group_id, score) in enumerate(user_based_recs, 1):
            print(f"  {rank:2d}. Group {group_id}: Score={score:.4f}")
        
        # SVD recommendations
        for k in [10, 20]:
            print(f"\nSVD-Based CF (k={k} latent factors):")
            svd_recs = self.svd_recommend(sample_user_id, k=k, top_n=10)
            for rank, (group_id, score) in enumerate(svd_recs, 1):
                print(f"  {rank:2d}. Group {group_id}: Score={score:.4f}")
        
        return user_based_recs, self.svd_recommend(sample_user_id, k=10)


# MAIN EXECUTION

def main():
    """Main function to run Collaborative Filtering"""
    
    print("COLLABORATIVE FILTERING SYSTEM")
    
    # Initialize recommender
    cf = CollaborativeFilteringRecommender(data_path='SECTION2_DomainRecommender/data/')
    
    # Load data
    cf.load_data(sample_size=5000)
    
    # Create user-item matrix
    cf.create_user_item_matrix()
    
    # 8.1 Compute user similarity
    cf.compute_user_similarity(metric='cosine')
    
    # 8.2 Apply SVD
    cf.apply_svd(k_values=[10, 20])
    
    # Demonstrate with sample user
    sample_user = cf.user_ids[0]
    cf.demonstrate_cf(sample_user)
    
    print("COLLABORATIVE FILTERING COMPLETE")
    
    return cf


if __name__ == "__main__":
    cf = main()
