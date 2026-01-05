# ============================================================
# Part 2: Content-Based Recommendation
# Domain: Interest-Based Group Formation (Meetup.com)
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

TABLE_DIR = "SECTION2_DomainRecommender/results/tables/Content_Based"
PLOT_DIR = "SECTION2_DomainRecommender/results/plots/Content_Based"

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
# 3. FEATURE EXTRACTION AND VECTOR SPACE MODEL
# ============================================================

class ContentBasedRecommender:
    """
    Content-Based Filtering Recommender System
    
    Approach:
    - Uses TF-IDF vectors for tag-based features
    - Builds user profiles from weighted tag preferences
    - Computes cosine similarity for recommendations
    """
    
    def __init__(self, data_path='SECTION2_DomainRecommender/data/'):
        self.data_path = data_path
        self.group_tag_df = None
        self.user_tag_df = None
        self.user_group_df = None
        self.tag_text_df = None
        self.item_feature_matrix = None
        self.tfidf_vectorizer = None
        self.group_ids = None
        self.user_profiles = {}
        
    def load_data(self, sample_size=None):
        """Load and optionally sample the datasets"""
        print("LOADING DATA")
        
        # Load tag text (tag_id, tag_text)
        self.tag_text_df = pd.read_csv(
            f"{self.data_path}tag_text.csv", 
            header=None, 
            names=['tag_id', 'tag_text']
        )
        print(f"Tags loaded: {len(self.tag_text_df):,}")
        
        # Load group-tag mappings (group_id, tag_id)
        self.group_tag_df = pd.read_csv(
            f"{self.data_path}group_tag.csv", 
            header=None, 
            names=['group_id', 'tag_id']
        )
        print(f"Group-Tag mappings loaded: {len(self.group_tag_df):,}")
        
        # Load user-tag preferences (user_id, tag_id)
        self.user_tag_df = pd.read_csv(
            f"{self.data_path}user_tag.csv", 
            header=None, 
            names=['user_id', 'tag_id']
        )
        print(f"User-Tag preferences loaded: {len(self.user_tag_df):,}")
        
        # Load user-group memberships (user_id, group_id)
        self.user_group_df = pd.read_csv(
            f"{self.data_path}user_group.csv", 
            header=None, 
            names=['user_id', 'group_id']
        )
        # Remove duplicates
        self.user_group_df = self.user_group_df.drop_duplicates()
        print(f"User-Group memberships loaded: {len(self.user_group_df):,}")
        
        # Sample data if specified (for faster processing)
        if sample_size:
            self._sample_data(sample_size)
            
    def _sample_data(self, n_users=10000, n_groups=977, n_interactions=150000):
        """Sample a subset of data for target counts"""
        print(f"\nSampling to target: {n_users:,} users, {n_groups:,} groups, {n_interactions:,} interactions...")
        
        # Step 1: Get popular groups (top n_groups by member count)
        group_counts = self.user_group_df['group_id'].value_counts()
        top_groups = group_counts.head(n_groups).index.tolist()
        
        # Filter user_group to these groups only
        self.user_group_df = self.user_group_df[
            self.user_group_df['group_id'].isin(top_groups)
        ]
        
        # Step 2: Get active users from filtered data
        user_counts = self.user_tag_df['user_id'].value_counts()
        users_in_groups = set(self.user_group_df['user_id'].unique())
        active_users = [u for u in user_counts.index if u in users_in_groups][:n_users]
        
        # Filter all dataframes
        self.user_tag_df = self.user_tag_df[
            self.user_tag_df['user_id'].isin(active_users)
        ]
        self.user_group_df = self.user_group_df[
            self.user_group_df['user_id'].isin(active_users)
        ]
        
        # Limit interactions if needed
        if len(self.user_group_df) > n_interactions:
            self.user_group_df = self.user_group_df.sample(n=n_interactions, random_state=42)
        
        # Keep only groups that remain
        active_groups = self.user_group_df['group_id'].unique()
        self.group_tag_df = self.group_tag_df[
            self.group_tag_df['group_id'].isin(active_groups)
        ]
        
        print(f"Final users: {len(active_users):,}")
        print(f"Final groups: {len(active_groups):,}")
        print(f"Final interactions: {len(self.user_group_df):,}")
        
    def create_group_text_features(self):
        """
        3.1. Text feature extraction: TF-IDF vectors with basic preprocessing
             (tokenization, stop-word removal)
        """
        print("3.1 TF-IDF FEATURE EXTRACTION")
        
        # Merge group_tag with tag_text to get tag names
        group_tags_merged = self.group_tag_df.merge(
            self.tag_text_df, 
            on='tag_id', 
            how='left'
        )
        
        # Aggregate tags for each group into a single text document
        group_documents = group_tags_merged.groupby('group_id')['tag_text'].apply(
            lambda x: ' '.join(x.dropna().astype(str))
        ).reset_index()
        group_documents.columns = ['group_id', 'tag_document']
        
        print(f"Groups with tag documents: {len(group_documents):,}")
        print("\nSample group documents:")
        for i, row in group_documents.head(3).iterrows():
            print(f"  Group {row['group_id']}: '{row['tag_document'][:80]}...'")
        
        # Store group IDs for later reference
        self.group_ids = group_documents['group_id'].values
        
        # TF-IDF Vectorization
        print("\nApplying TF-IDF Vectorization...")
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,           # Convert to lowercase
            stop_words='english',     # Remove stop words
            max_features=5000,        # Limit vocabulary size
            min_df=2,                 # Minimum document frequency
            max_df=0.95               # Maximum document frequency
        )
        
        self.item_feature_matrix = self.tfidf_vectorizer.fit_transform(
            group_documents['tag_document']
        )
        
        print(f"\nItem-Feature Matrix Shape: {self.item_feature_matrix.shape}")
        print(f"  - {self.item_feature_matrix.shape[0]} groups (items)")
        print(f"  - {self.item_feature_matrix.shape[1]} TF-IDF features")
        
        # Feature names (vocabulary)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"\nSample features (tags): {list(feature_names[:20])}")
        
        # ============================================================
        # SAVE STEP 3.1 RESULTS
        # ============================================================
        
        # Save group documents
        save_table(group_documents, "group_tag_documents.csv")
        
        # Save TF-IDF vocabulary stats
        vocab_df = pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names))
        })
        save_table(vocab_df, "tfidf_vocabulary.csv")
        
        # Save TF-IDF matrix statistics
        tfidf_stats = pd.DataFrame({
            'group_id': self.group_ids,
            'n_features': np.diff(self.item_feature_matrix.indptr),
            'max_tfidf': self.item_feature_matrix.max(axis=1).toarray().flatten(),
            'mean_tfidf': np.array(self.item_feature_matrix.mean(axis=1)).flatten()
        })
        save_table(tfidf_stats, "group_tfidf_stats.csv")
        
        # Plot TF-IDF distribution
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(tfidf_stats['n_features'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Non-Zero Features')
        plt.ylabel('Number of Groups')
        plt.title('TF-IDF Sparsity Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(tfidf_stats['max_tfidf'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Max TF-IDF Score')
        plt.ylabel('Number of Groups')
        plt.title('Max TF-IDF Distribution')
        
        plt.tight_layout()
        save_plot("tfidf_distribution.png")
        
        return self.item_feature_matrix
    
    def document_feature_selection(self):
        """
        3.2. Additional features: Categorical features (Tags)
        3.3. Create item-feature matrix and document feature selection
        """
        print("3.2 & 3.3 FEATURE SELECTION DOCUMENTATION")
        
        print("""
FEATURE SELECTION APPROACH:
---------------------------
1. Primary Features: Tag texts (categorical/text features)
   - Each group is described by a set of tags
   - Tags represent interests/topics (e.g., 'python', 'hiking', 'music')

2. TF-IDF Parameters:
   - lowercase=True: Normalize case
   - stop_words='english': Remove common words
   - max_features=5000: Top 5000 most informative terms
   - min_df=2: Term must appear in at least 2 groups
   - max_df=0.95: Ignore terms in >95% of groups

3. Item-Feature Matrix:
   - Rows: Groups (items to recommend)
   - Columns: TF-IDF weighted tag terms
   - Values: TF-IDF scores (0 to 1)
   - Sparse matrix format for efficiency
        """)
        
        # Matrix statistics
        matrix = self.item_feature_matrix
        non_zero = matrix.nnz
        total = matrix.shape[0] * matrix.shape[1]
        sparsity = (1 - non_zero / total) * 100
        
        print(f"ITEM-FEATURE MATRIX STATISTICS:")
        print(f"  Shape: {matrix.shape}")
        print(f"  Non-zero entries: {non_zero:,}")
        print(f"  Sparsity: {sparsity:.2f}%")
        print(f"  Memory: {matrix.data.nbytes / 1024:.2f} KB")
        
    def build_user_profiles(self, method='weighted_average'):
        """
        4.1. Build user profiles: Weighted average of rated item features
        """
        print("4. USER PROFILE CONSTRUCTION")
        
        # Get feature names from TF-IDF vectorizer
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Create user tag documents (similar to group documents)
        user_tags_merged = self.user_tag_df.merge(
            self.tag_text_df, 
            on='tag_id', 
            how='left'
        )
        
        # Count tag occurrences per user (implicit rating/weight)
        user_tag_counts = user_tags_merged.groupby(
            ['user_id', 'tag_text']
        ).size().reset_index(name='weight')
        
        # Aggregate weighted tag texts for each user
        user_documents = user_tags_merged.groupby('user_id')['tag_text'].apply(
            lambda x: ' '.join(x.dropna().astype(str))
        ).reset_index()
        user_documents.columns = ['user_id', 'tag_document']
        
        print(f"Building profiles for {len(user_documents):,} users...")
        
        # Transform user documents using the SAME TF-IDF vectorizer (fitted on groups)
        user_feature_matrix = self.tfidf_vectorizer.transform(
            user_documents['tag_document']
        )
        
        # Store user profiles
        self.user_ids = user_documents['user_id'].values
        self.user_feature_matrix = user_feature_matrix
        
        # Create user profile dictionary
        for idx, user_id in enumerate(self.user_ids):
            self.user_profiles[user_id] = user_feature_matrix[idx]
        
        print(f"User Profile Matrix Shape: {user_feature_matrix.shape}")
        print(f"  - {user_feature_matrix.shape[0]} users")
        print(f"  - {user_feature_matrix.shape[1]} features")
        
        # Show sample user profile
        print("\nSample User Profile (top tags by TF-IDF weight):")
        sample_user_idx = 0
        sample_user_id = self.user_ids[sample_user_idx]
        sample_profile = user_feature_matrix[sample_user_idx].toarray().flatten()
        top_indices = sample_profile.argsort()[-10:][::-1]
        print(f"  User {sample_user_id}:")
        for idx in top_indices:
            if sample_profile[idx] > 0:
                print(f"    - {feature_names[idx]}: {sample_profile[idx]:.4f}")
        
        # ============================================================
        # SAVE STEP 4.1 RESULTS
        # ============================================================
        
        # Save user profile statistics
        profile_stats = []
        for idx, user_id in enumerate(self.user_ids):
            profile = user_feature_matrix[idx].toarray().flatten()
            profile_stats.append({
                'user_id': user_id,
                'n_features': np.sum(profile > 0),
                'max_weight': np.max(profile),
                'mean_weight': np.mean(profile[profile > 0]) if np.sum(profile > 0) > 0 else 0
            })
        
        profile_stats_df = pd.DataFrame(profile_stats)
        save_table(profile_stats_df, "user_profile_stats.csv")
        
        # Plot user profile distribution
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(profile_stats_df['n_features'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Non-Zero Features')
        plt.ylabel('Number of Users')
        plt.title('User Profile Sparsity')
        
        plt.subplot(1, 2, 2)
        plt.hist(profile_stats_df['max_weight'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Max Profile Weight')
        plt.ylabel('Number of Users')
        plt.title('User Profile Intensity')
        
        plt.tight_layout()
        save_plot("user_profile_distribution.png")
        
        return self.user_profiles
    
    def handle_cold_start(self, strategy='popular_items'):
        """
        4.2. Handle cold-start users: Use popular item features
        """
        print("4.2 COLD-START HANDLING")
        print(f"Strategy: {strategy}")
        
        if strategy == 'popular_items':
            # Find most popular groups (most members)
            group_popularity = self.user_group_df['group_id'].value_counts()
            top_groups = group_popularity.head(100).index.tolist()
            
            # Get indices of top groups in item_feature_matrix
            group_id_to_idx = {gid: idx for idx, gid in enumerate(self.group_ids)}
            top_group_indices = [
                group_id_to_idx[gid] for gid in top_groups 
                if gid in group_id_to_idx
            ]
            
            # Average feature vector of popular groups
            if top_group_indices:
                popular_features = self.item_feature_matrix[top_group_indices].mean(axis=0)
                self.cold_start_profile = np.asarray(popular_features).flatten()
                print(f"Cold-start profile created from top {len(top_group_indices)} popular groups")
            else:
                self.cold_start_profile = np.zeros(self.item_feature_matrix.shape[1])
                print("Warning: Could not create cold-start profile")
        
        return self.cold_start_profile
    
    def compute_user_item_similarity(self, user_id):
        """
        5.1. Compute similarity: Cosine similarity between user profiles and items
        """
        # Get user profile
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
        else:
            # Cold-start: use default profile
            user_profile = self.cold_start_profile.reshape(1, -1)
        
        # Compute cosine similarity with all groups
        similarities = cosine_similarity(user_profile, self.item_feature_matrix)
        
        return similarities.flatten()
    
    def generate_recommendations(self, user_id, top_n=10, exclude_joined=True):
        """
        5.2. Generate top-N recommendations: Top-10 and Top-20
        """
        # Compute similarities
        similarities = self.compute_user_item_similarity(user_id)
        
        # Create group_id to similarity mapping
        group_similarities = list(zip(self.group_ids, similarities))
        
        # Get groups user already joined
        if exclude_joined:
            joined_groups = set(
                self.user_group_df[
                    self.user_group_df['user_id'] == user_id
                ]['group_id'].values
            )
        else:
            joined_groups = set()
        
        # Filter out joined groups and sort by similarity
        recommendations = [
            (gid, sim) for gid, sim in group_similarities 
            if gid not in joined_groups
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_n]
    
    def recommend_top_n(self, user_id):
        """
        5.2. Generate Top-10 and Top-20 Recommendations
        """
        print(f"5.1 SIMILARITY FOR USER {user_id}")
        
        # Top-10
        top_10 = self.generate_recommendations(user_id, top_n=10)
        print("\nTop-10 Recommendations:")
        for rank, (group_id, score) in enumerate(top_10, 1):
            # Get group tags
            group_tags = self.group_tag_df[
                self.group_tag_df['group_id'] == group_id
            ].merge(self.tag_text_df, on='tag_id')['tag_text'].head(3).tolist()
            tags_str = ', '.join(group_tags) if group_tags else 'N/A'
            print(f"  {rank:2d}. Group {group_id}: Score={score:.4f} | Tags: {tags_str}")
        
        # Top-20
        top_20 = self.generate_recommendations(user_id, top_n=20)
        print("\nTop-20 Recommendations:")
        for rank, (group_id, score) in enumerate(top_20, 1):
            print(f"  {rank:2d}. Group {group_id}: Score={score:.4f}")
        
        # ============================================================
        # SAVE TOP-10 AND TOP-20 RECOMMENDATIONS
        # ============================================================
        
        # Save Top-10 with tags
        top10_data = []
        for rank, (group_id, score) in enumerate(top_10, 1):
            group_tags = self.group_tag_df[
                self.group_tag_df['group_id'] == group_id
            ].merge(self.tag_text_df, on='tag_id')['tag_text'].head(5).tolist()
            top10_data.append({
                'rank': rank,
                'group_id': group_id,
                'score': score,
                'tags': ', '.join(group_tags) if group_tags else 'N/A'
            })
        save_table(pd.DataFrame(top10_data), f"top10_recommendations_user_{user_id}.csv")
        
        # Save Top-20
        top20_data = []
        for rank, (group_id, score) in enumerate(top_20, 1):
            group_tags = self.group_tag_df[
                self.group_tag_df['group_id'] == group_id
            ].merge(self.tag_text_df, on='tag_id')['tag_text'].head(5).tolist()
            top20_data.append({
                'rank': rank,
                'group_id': group_id,
                'score': score,
                'tags': ', '.join(group_tags) if group_tags else 'N/A'
            })
        save_table(pd.DataFrame(top20_data), f"top20_recommendations_user_{user_id}.csv")
        
        return top_10, top_20
    
    def build_item_knn(self, k_values=[10, 20]):
        """
        6.1. Implement item-based k-NN: k=10, k=20
        """
        print("6. k-NEAREST NEIGHBORS IMPLEMENTATION")
        
        self.knn_models = {}
        self.item_similarities = {}
        
        for k in k_values:
            print(f"\nBuilding k-NN model with k={k}...")
            
            # Fit NearestNeighbors model
            knn = NearestNeighbors(
                n_neighbors=k + 1,  # +1 because item is similar to itself
                metric='cosine',
                algorithm='brute'
            )
            knn.fit(self.item_feature_matrix)
            
            self.knn_models[k] = knn
            
            # Find neighbors for all items
            distances, indices = knn.kneighbors(self.item_feature_matrix)
            
            # Store similarities (1 - cosine distance)
            self.item_similarities[k] = {
                'distances': distances,
                'indices': indices,
                'similarities': 1 - distances
            }
            
            print(f"  - Found {k} nearest neighbors for {len(self.group_ids)} groups")
        
        # ============================================================
        # SAVE STEP 6.1 RESULTS
        # ============================================================
        
        for k in k_values:
            # Save sample neighbors for first 100 groups
            sample_neighbors = []
            for idx in range(min(100, len(self.group_ids))):
                group_id = self.group_ids[idx]
                neighbors = self.item_similarities[k]['indices'][idx][1:k+1]
                sims = self.item_similarities[k]['similarities'][idx][1:k+1]
                for n_idx, (neighbor_idx, sim) in enumerate(zip(neighbors, sims)):
                    sample_neighbors.append({
                        'group_id': group_id,
                        'neighbor_rank': n_idx + 1,
                        'neighbor_group_id': self.group_ids[neighbor_idx],
                        'similarity': sim
                    })
            
            neighbors_df = pd.DataFrame(sample_neighbors)
            save_table(neighbors_df, f"knn_neighbors_k{k}.csv")
        
        return self.knn_models
    
    def knn_predict_rating(self, user_id, group_id, k=10):
        """
        6.1. Predict rating using weighted average of similar items
        """
        if k not in self.item_similarities:
            raise ValueError(f"k={k} not available. Run build_item_knn first.")
        
        # Get group index
        group_id_to_idx = {gid: idx for idx, gid in enumerate(self.group_ids)}
        if group_id not in group_id_to_idx:
            return 0.0
        
        group_idx = group_id_to_idx[group_id]
        
        # Get similar items
        similar_indices = self.item_similarities[k]['indices'][group_idx][1:]  # Exclude self
        similar_sims = self.item_similarities[k]['similarities'][group_idx][1:]
        
        # Get user's joined groups
        user_groups = set(
            self.user_group_df[
                self.user_group_df['user_id'] == user_id
            ]['group_id'].values
        )
        
        # Calculate weighted average of user's interactions with similar items
        numerator = 0.0
        denominator = 0.0
        
        for sim_idx, similarity in zip(similar_indices, similar_sims):
            sim_group_id = self.group_ids[sim_idx]
            # Implicit rating: 1 if user joined, 0 otherwise
            rating = 1.0 if sim_group_id in user_groups else 0.0
            numerator += similarity * rating
            denominator += abs(similarity)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def knn_recommend(self, user_id, k=10, top_n=10):
        """
        6.1. Generate recommendations using k-NN
        """
        # Get groups user already joined
        joined_groups = set(
            self.user_group_df[
                self.user_group_df['user_id'] == user_id
            ]['group_id'].values
        )
        
        # Predict scores for all unjoined groups
        predictions = []
        for group_id in self.group_ids:
            if group_id not in joined_groups:
                score = self.knn_predict_rating(user_id, group_id, k)
                predictions.append((group_id, score))
        
        # Sort by predicted score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]
    
    def compare_approaches(self, user_id):
        """
        6.2. Compare content-based and k-NN approaches
        """
        print(f"6.2 COMPARISON: CONTENT-BASED vs k-NN (User {user_id})")
        
        # Content-Based recommendations
        cb_recs = self.generate_recommendations(user_id, top_n=10)
        
        # k-NN recommendations (k=10)
        knn_recs_10 = self.knn_recommend(user_id, k=10, top_n=10)
        
        # k-NN recommendations (k=20)
        knn_recs_20 = self.knn_recommend(user_id, k=20, top_n=10)
        
        print("\n{:<6} {:<20} {:<20} {:<20}".format(
            "Rank", "Content-Based", "k-NN (k=10)", "k-NN (k=20)"
        ))
        
        for i in range(10):
            cb = f"G{cb_recs[i][0]}({cb_recs[i][1]:.3f})" if i < len(cb_recs) else "-"
            knn10 = f"G{knn_recs_10[i][0]}({knn_recs_10[i][1]:.3f})" if i < len(knn_recs_10) else "-"
            knn20 = f"G{knn_recs_20[i][0]}({knn_recs_20[i][1]:.3f})" if i < len(knn_recs_20) else "-"
            print(f"{i+1:<6} {cb:<20} {knn10:<20} {knn20:<20}")
        
        # Calculate overlap
        cb_set = set([r[0] for r in cb_recs])
        knn10_set = set([r[0] for r in knn_recs_10])
        knn20_set = set([r[0] for r in knn_recs_20])
        
        print(f"\nOverlap Analysis:")
        print(f"  Content-Based ∩ k-NN(k=10): {len(cb_set & knn10_set)} items")
        print(f"  Content-Based ∩ k-NN(k=20): {len(cb_set & knn20_set)} items")
        print(f"  k-NN(k=10) ∩ k-NN(k=20): {len(knn10_set & knn20_set)} items")
        
        # ============================================================
        # SAVE COMPARISON TABLE
        # ============================================================
        
        comparison_data = []
        for i in range(10):
            comparison_data.append({
                'rank': i + 1,
                'cb_group_id': cb_recs[i][0] if i < len(cb_recs) else None,
                'cb_score': cb_recs[i][1] if i < len(cb_recs) else None,
                'knn10_group_id': knn_recs_10[i][0] if i < len(knn_recs_10) else None,
                'knn10_score': knn_recs_10[i][1] if i < len(knn_recs_10) else None,
                'knn20_group_id': knn_recs_20[i][0] if i < len(knn_recs_20) else None,
                'knn20_score': knn_recs_20[i][1] if i < len(knn_recs_20) else None
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        save_table(comparison_df, f"cb_vs_knn_comparison_user_{user_id}.csv")
        
        # Save overlap analysis
        overlap_df = pd.DataFrame({
            'comparison': ['CB ∩ k-NN(k=10)', 'CB ∩ k-NN(k=20)', 'k-NN(k=10) ∩ k-NN(k=20)'],
            'overlap_count': [len(cb_set & knn10_set), len(cb_set & knn20_set), len(knn10_set & knn20_set)]
        })
        save_table(overlap_df, f"cb_knn_overlap_analysis_user_{user_id}.csv")
        
        return cb_recs, knn_recs_10, knn_recs_20


def run_numerical_example():
    """
    7.1. Complete Numerical Example - Step-by-step showing:
         - Sample item descriptions
         - TF-IDF calculation for 3-5 sample items
         - User profile from 3-5 ratings
         - Similarity scores
         - Top-5 recommendations with scores
    """
    print("7. COMPLETE NUMERICAL EXAMPLE")
    
    # -------------------------------------------------------------------------
    # Step 1: Sample Item Descriptions (Groups with Tags)
    # -------------------------------------------------------------------------
    print("\nSTEP 1: SAMPLE ITEM DESCRIPTIONS")
    
    sample_groups = {
        'G1': "python programming coding data science machine learning",
        'G2': "hiking outdoor nature camping adventure travel",
        'G3': "python data analysis statistics visualization",
        'G4': "music rock concert live bands guitar",
        'G5': "outdoor running fitness marathon training"
    }
    
    print("\nSample Groups (Items):")
    for group_id, tags in sample_groups.items():
        print(f"  {group_id}: '{tags}'")
    
    # -------------------------------------------------------------------------
    # Step 2: TF-IDF Calculation
    # -------------------------------------------------------------------------
    print("\nSTEP 2: TF-IDF CALCULATION")
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(list(sample_groups.values()))
    feature_names = tfidf.get_feature_names_out()
    
    print("\nVocabulary (Features):")
    print(f"  {list(feature_names)}")
    
    print("\nTF-IDF Matrix:")
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=list(sample_groups.keys()),
        columns=feature_names
    ).round(3)
    
    # Show non-zero values for each group
    for group_id in sample_groups.keys():
        row = tfidf_df.loc[group_id]
        non_zero = row[row > 0].sort_values(ascending=False)
        print(f"\n  {group_id}:")
        for term, score in non_zero.items():
            print(f"    {term}: {score:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 3: User Profile from Ratings
    # -------------------------------------------------------------------------
    print("\nSTEP 3: USER PROFILE CONSTRUCTION")
    
    # User's ratings (implicit: joined groups)
    user_ratings = {
        'G1': 5,  # Strong interest in Python/ML
        'G3': 4,  # Good interest in data analysis
        'G5': 3   # Some interest in fitness
    }
    
    print("\nUser's Rated Items:")
    for group_id, rating in user_ratings.items():
        print(f"  {group_id}: Rating = {rating} -> '{sample_groups[group_id]}'")
    
    # Build user profile (weighted average)
    print("\nBuilding User Profile (Weighted Average):")
    print("  Formula: profile = Σ(rating × item_vector) / Σ(rating)")
    
    user_profile = np.zeros(len(feature_names))
    total_weight = 0
    
    for group_id, rating in user_ratings.items():
        group_idx = list(sample_groups.keys()).index(group_id)
        item_vector = tfidf_matrix[group_idx].toarray().flatten()
        user_profile += rating * item_vector
        total_weight += rating
        print(f"    + {rating} × vector({group_id})")
    
    user_profile = user_profile / total_weight
    print(f"\n  User Profile Vector (normalized):")
    
    profile_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': user_profile.round(4)
    })
    profile_df = profile_df[profile_df['Weight'] > 0].sort_values(
        'Weight', ascending=False
    )
    for _, row in profile_df.iterrows():
        print(f"    {row['Feature']}: {row['Weight']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Similarity Computation
    # -------------------------------------------------------------------------
    print("\nSTEP 4: SIMILARITY COMPUTATION (COSINE SIMILARITY)")
    
    # Compute cosine similarity between user profile and all items
    similarities = cosine_similarity(
        user_profile.reshape(1, -1), 
        tfidf_matrix
    ).flatten()
    
    print("\nCosine Similarity Scores:")
    print("  Formula: cos(θ) = (A · B) / (||A|| × ||B||)")
    print()
    
    for i, (group_id, tags) in enumerate(sample_groups.items()):
        status = "Already rated" if group_id in user_ratings else ""
        print(f"  {group_id}: {similarities[i]:.4f} {status}")
    
    # -------------------------------------------------------------------------
    # Step 5: Top-5 Recommendations
    # -------------------------------------------------------------------------
    print("\nSTEP 5: TOP-5 RECOMMENDATIONS")
    
    # Exclude already-rated items
    recommendations = []
    for i, group_id in enumerate(sample_groups.keys()):
        if group_id not in user_ratings:
            recommendations.append((group_id, similarities[i]))
    
    # Sort by similarity
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFinal Recommendations (excluding rated items):")
    for rank, (group_id, score) in enumerate(recommendations[:5], 1):
        print(f"  Rank {rank}: {group_id}")
        print(f"           Score: {score:.4f}")
        print(f"           Tags: '{sample_groups[group_id]}'")
        print()
    
    print("\nInterpretation:")
    print("  - G2 (hiking/outdoor) ranked highest due to overlap with")
    print("    user's interest in 'outdoor' from G5 (running/fitness)")
    print("  - G4 (music) ranked lowest as it has no feature overlap")
    
    return tfidf_matrix, user_profile, recommendations


# MAIN EXECUTION

def main():
    """Main function to run the complete Content-Based Recommender"""
    
    print("CONTENT-BASED RECOMMENDATION SYSTEM")
    
    # Initialize recommender
    recommender = ContentBasedRecommender(data_path='SECTION2_DomainRecommender/data/')
    
    # Load data (sample for development)
    recommender.load_data(sample_size=5000)
    
    # 3.1 Feature Extraction
    recommender.create_group_text_features()
    
    # 3.2 & 3.3 Document features
    recommender.document_feature_selection()
    
    # 4.1 Build user profiles
    recommender.build_user_profiles()
    
    # 4.2 Handle cold-start
    recommender.handle_cold_start(strategy='popular_items')
    
    # 5.2 Generate recommendations for a sample user
    sample_user = recommender.user_ids[0]
    recommender.recommend_top_n(sample_user)
    
    # 6.1 Build k-NN models
    recommender.build_item_knn(k_values=[10, 20])
    
    # 6.2 Compare approaches
    recommender.compare_approaches(sample_user)
    
    # 7. Numerical example
    run_numerical_example()
    
    print("CONTENT-BASED SYSTEM COMPLETE")
    
    return recommender


if __name__ == "__main__":
    recommender = main()
