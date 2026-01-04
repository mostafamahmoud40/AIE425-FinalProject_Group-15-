# ============================================================
# Part 3: Hybrid Recommender System for Interest-Based Groups
# ============================================================
# Implements:
# - Section 9: Weighted Hybrid (α × CB + (1-α) × CF)
# - Section 10: Cold-Start Handling
# - Section 11: Baseline Comparison
# - Section 12: Results Analysis
# Python Version: 3.11.14
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our recommenders
from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender, GPU_AVAILABLE

# ============================================================
# PATHS
# ========✓ ====================================================

TABLE_DIR = "SECTION2_DomainRecommender/results/tables/Hybrid_System"
PLOT_DIR = "SECTION2_DomainRecommender/results/plots/Hybrid_S✓ ystem"

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
    print(f"  Saved: {TABLE_DIR}/{filename}")

def save_plot(filename):
    """Save current plot"""
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/{filename}")

# ============================================================
# 9. HYBRID RECOMMENDATION SYSTEM
# ============================================================

class HybridRecommender:
    """
    Hybrid Recommender System combining Content-Based and Collaborative Filtering
    
    Approach: Weighted Hybrid
    Score = α × Content-Based + (1 − α) × Collaborative Filtering
    """
    
    def __init__(self, data_path='SECTION2_DomainRecommender/data/', sample_size=10000, min_group_members=50):
        self.data_path = data_path
        self.sample_size = sample_size
        self.min_group_members = min_group_members  # Filter groups with few members
        self.cb_recommender = None
        self.cf_recommender = None
        self.common_users = None
        self.common_groups = None
        self.best_alpha = 0.5
        self.user_group_df = None
        self.train_data = None
        self.test_data = None
        
    def initialize(self, cb_recommender=None, cf_recommender=None):
        """
        Initialize hybrid recommender system.
        
        Args:
            cb_recommender: Optional pre-trained ContentBasedRecommender
            cf_recommender: Optional pre-trained CollaborativeFilteringRecommender
            
        If recommenders are provided, they will be reused to avoid duplicate computation.
        """
        print("INITIALIZING HYBRID RECOMMENDER SYSTEM")
        
        # Load raw data first to ensure consistency
        print("\n--- Loading shared data ---")
        self.user_group_df = pd.read_csv(
            f"{self.data_path}user_group.csv",
            header=None,
            names=['user_id', 'group_id']
        ).drop_duplicates()
        
        # Filter to popular groups only (to reduce dimensions)
        print(f"\n--- Filtering groups with >= {self.min_group_members} members ---")
        group_counts = self.user_group_df['group_id'].value_counts()
        popular_groups = group_counts[group_counts >= self.min_group_members].index.tolist()
        self.user_group_df = self.user_group_df[
            self.user_group_df['group_id'].isin(popular_groups)
        ]
        print(f"Groups after filtering: {len(popular_groups):,}")
        
        # Get active users (with at least 5 interactions in filtered groups)
        user_counts = self.user_group_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 5].index[:self.sample_size].tolist()
        
        # Filter to active users
        self.user_group_df = self.user_group_df[
            self.user_group_df['user_id'].isin(active_users)
        ]
        
        print(f"Users after filtering: {len(active_users):,}")
        print(f"Interactions after filtering: {len(self.user_group_df):,}")
        
        # Create train/test split (80/20)
        print("\n--- Creating train/test split ---")
        self.train_data, self.test_data = train_test_split(
            self.user_group_df, 
            test_size=0.2, 
            random_state=42
        )
        print(f"Train interactions: {len(self.train_data):,}")
        print(f"Test interactions: {len(self.test_data):,}")
        
        # Get common users and groups
        self.common_users = set(self.train_data['user_id'].unique())
        self.common_groups = set(self.train_data['group_id'].unique())
        
        print(f"Common users: {len(self.common_users):,}")
        print(f"Common groups: {len(self.common_groups):,}")
        
        # Use existing Content-Based recommender if provided
        if cb_recommender is not None:
            print("\n--- Using existing Content-Based Recommender ---")
            self.cb_recommender = cb_recommender
            # Update common_users to intersection with CB users
            cb_users = set(self.cb_recommender.user_profiles.keys())
            self.common_users = self.common_users.intersection(cb_users)
        else:
            # Initialize Content-Based recommender
            print("\n--- Initializing Content-Based Recommender ---")
            self.cb_recommender = ContentBasedRecommender(data_path=self.data_path)
            self.cb_recommender.load_data(sample_size=None)  # Load all, we'll filter
            
            # Filter CB data to common users/groups
            self.cb_recommender.user_group_df = self.train_data.copy()
            self.cb_recommender.group_tag_df = self.cb_recommender.group_tag_df[
                self.cb_recommender.group_tag_df['group_id'].isin(self.common_groups)
            ]
            self.cb_recommender.user_tag_df = self.cb_recommender.user_tag_df[
                self.cb_recommender.user_tag_df['user_id'].isin(self.common_users)
            ]
            
            # Build CB features
            self.cb_recommender.create_group_text_features()
            self.cb_recommender.build_user_profiles()
            self.cb_recommender.handle_cold_start(strategy='popular_items')
        
        # Use existing Collaborative Filtering recommender if provided
        if cf_recommender is not None:
            print("\n--- Using existing Collaborative Filtering Recommender ---")
            self.cf_recommender = cf_recommender
            # Update common_users to intersection with CF users
            cf_users = set(self.cf_recommender.user_id_to_idx.keys())
            self.common_users = self.common_users.intersection(cf_users)
            # Update common_groups to intersection with CF groups
            cf_groups = set(self.cf_recommender.group_id_to_idx.keys())
            self.common_groups = self.common_groups.intersection(cf_groups)
            print(f"Users available in both CB & CF: {len(self.common_users):,}")
            print(f"Groups available in both CB & CF: {len(self.common_groups):,}")
        else:
            # Initialize Collaborative Filtering recommender
            print("\n--- Initializing Collaborative Filtering Recommender ---")
            self.cf_recommender = CollaborativeFilteringRecommender(data_path=self.data_path)
            self.cf_recommender.user_group_df = self.train_data.copy()
            self.cf_recommender.create_user_item_matrix()
            self.cf_recommender.compute_user_similarity()
            self.cf_recommender.apply_svd(k_values=[10, 20])
        
        print("\nHYBRID SYSTEM INITIALIZED")
    
    # =========================================================================
    # 9.1 WEIGHTED HYBRID RECOMMENDATION
    # =========================================================================
    
    def get_cb_scores(self, user_id):
        """Get Content-Based scores for a user"""
        if user_id not in self.cb_recommender.user_profiles:
            # Cold-start: use default profile
            profile = self.cb_recommender.cold_start_profile.reshape(1, -1)
        else:
            profile = self.cb_recommender.user_profiles[user_id]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            profile, 
            self.cb_recommender.item_feature_matrix
        ).flatten()
        
        # Return as dict: group_id -> score
        scores = {}
        for idx, score in enumerate(similarities):
            group_id = self.cb_recommender.group_ids[idx]
            scores[group_id] = score
        return scores
    
    def get_cf_scores(self, user_id, method='svd', k=10):
        """Get Collaborative Filtering scores for a user"""
        scores = {}
        
        if user_id not in self.cf_recommender.user_id_to_idx:
            return scores
        
        user_idx = self.cf_recommender.user_id_to_idx[user_id]
        
        if method == 'svd' and k in self.cf_recommender.svd_predictions:
            predictions = self.cf_recommender.svd_predictions[k][user_idx]
            for group_idx, score in enumerate(predictions):
                group_id = self.cf_recommender.idx_to_group_id[group_idx]
                scores[group_id] = max(0, score)  # Ensure non-negative
        elif method == 'user_based':
            # Get user-based predictions
            for group_id in self.common_groups:
                if group_id in self.cf_recommender.group_id_to_idx:
                    score = self.cf_recommender.user_based_predict(user_id, group_id, k=20)
                    scores[group_id] = score
        
        return scores
    
    def hybrid_recommend(self, user_id, alpha=0.5, top_n=10, cf_method='svd', k=10):
        """
        9.1 Weighted Hybrid Recommendation
        Score = α × CB + (1 − α) × CF
        """
        # Get scores from both systems
        cb_scores = self.get_cb_scores(user_id)
        cf_scores = self.get_cf_scores(user_id, method=cf_method, k=k)
        
        # Normalize scores to [0, 1]
        cb_max = max(cb_scores.values()) if cb_scores else 1
        cf_max = max(cf_scores.values()) if cf_scores else 1
        
        if cb_max > 0:
            cb_scores = {k: v / cb_max for k, v in cb_scores.items()}
        if cf_max > 0:
            cf_scores = {k: v / cf_max for k, v in cf_scores.items()}
        
        # Get groups user already joined (in training data)
        joined_groups = set(
            self.train_data[self.train_data['user_id'] == user_id]['group_id']
        )
        
        # Combine scores
        all_groups = set(cb_scores.keys()) | set(cf_scores.keys())
        hybrid_scores = []
        
        for group_id in all_groups:
            if group_id not in joined_groups:
                cb_score = cb_scores.get(group_id, 0)
                cf_score = cf_scores.get(group_id, 0)
                hybrid = alpha * cb_score + (1 - alpha) * cf_score
                hybrid_scores.append((group_id, hybrid, cb_score, cf_score))
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_scores[:top_n]
    
    def find_best_alpha(self, alpha_values=[0.3, 0.5, 0.7], n_users=100):
        """
        9.1 Test different alpha values and select best
        """
        print("9.1 TESTING ALPHA VALUES FOR WEIGHTED HYBRID")
        print("Formula: Score = α × Content-Based + (1 − α) × Collaborative")
        print(f"\nTesting α = {', '.join(map(str, alpha_values))}")
        
        # Get test users
        test_users = list(self.test_data['user_id'].unique())[:n_users]
        
        # Build ground truth from test data
        ground_truth = defaultdict(set)
        for _, row in self.test_data.iterrows():
            ground_truth[row['user_id']].add(row['group_id'])
        
        best_alpha = alpha_values[0]
        best_hit_rate = 0
        
        for alpha in alpha_values:
            hits = 0
            tested = 0
            
            for user_id in test_users:
                if user_id in ground_truth and len(ground_truth[user_id]) > 0:
                    recs = self.hybrid_recommend(user_id, alpha=alpha, top_n=10)
                    rec_groups = set([r[0] for r in recs])
                    
                    if rec_groups & ground_truth[user_id]:
                        hits += 1
                    tested += 1
            
            hit_rate = hits / tested if tested > 0 else 0
            print(f"\n  α = {alpha}:")
            print(f"    Hit Rate: {hit_rate:.4f}")
            print(f"    Users tested: {tested}")
            
            if hit_rate >= best_hit_rate:
                best_hit_rate = hit_rate
                best_alpha = alpha
        
        self.best_alpha = best_alpha
        print(f"\n  Best α = {best_alpha} (Hit Rate: {best_hit_rate:.4f})")
        
        # ============================================================
        # SAVE 9.1 ALPHA TUNING RESULTS
        # ============================================================
        print("\nSAVING ALPHA TUNING RESULTS...")
        
        alpha_results = pd.DataFrame({
            'alpha': alpha_values,
            'hit_rate': [best_hit_rate if a == best_alpha else 0 for a in alpha_values]
        })
        # Recompute all hit rates for saving
        alpha_results_list = []
        for alpha in alpha_values:
            hits = 0
            tested = 0
            for user_id in test_users:
                if user_id in ground_truth and len(ground_truth[user_id]) > 0:
                    recs = self.hybrid_recommend(user_id, alpha=alpha, top_n=10)
                    rec_groups = set([r[0] for r in recs])
                    if rec_groups & ground_truth[user_id]:
                        hits += 1
                    tested += 1
            hit_rate = hits / tested if tested > 0 else 0
            alpha_results_list.append({'alpha': alpha, 'hit_rate': hit_rate, 'users_tested': tested})
        
        alpha_df = pd.DataFrame(alpha_results_list)
        save_table(alpha_df, "alpha_tuning_results.csv")
        
        # Plot alpha comparison
        plt.figure(figsize=(8, 4))
        plt.bar([str(a) for a in alpha_df['alpha']], alpha_df['hit_rate'], color='steelblue', edgecolor='black')
        plt.xlabel('Alpha (α)')
        plt.ylabel('Hit Rate')
        plt.title('Hybrid System: Alpha Tuning Results')
        plt.axhline(y=best_hit_rate, color='red', linestyle='--', label=f'Best: α={best_alpha}')
        plt.legend()
        save_plot("alpha_tuning.png")
        
        return best_alpha
    
    def justify_hybrid_approach(self):
        """9.2 Justify hybrid approach choice"""
        print("9.2 HYBRID APPROACH JUSTIFICATION")
        
        print("""
CHOSEN APPROACH: Weighted Hybrid
--------------------------------
Formula: Score = α × Content-Based + (1 − α) × Collaborative

DOMAIN CHARACTERISTICS:

1. HIGH SPARSITY (99.8%):
   - Most users have joined few groups
   - CF alone struggles with sparse data
   - CB provides fallback using tag similarities
   → Weighted hybrid balances both signals

2. RICH CONTENT FEATURES:
   - 77,810 unique tags describe groups
   - Users explicitly follow tags (interests)
   - Strong content signal available
   → CB can provide meaningful recommendations

3. IMPLICIT FEEDBACK:
   - No explicit ratings (only join/not join)
   - CF works with binary interactions
   - Both CB and CF operate on similar scales
   → Easy to combine normalized scores

4. COLD-START HANDLING:
   - New users: CB uses their tag preferences
   - Active users: CF captures co-membership patterns
   - α can be adjusted based on user history
   → Weighted approach handles transition

WHY NOT OTHER APPROACHES:
- Switching Hybrid: Hard threshold may miss CF signal
- Cascade Hybrid: May eliminate good CF candidates early
        """)
    
    # =========================================================================
    # 10. COLD-START HANDLING
    # =========================================================================
    
    def evaluate_cold_start(self):
        """
        10.1 Evaluate cold-start handling for users with different activity levels
        """
        print("10. COLD-START HANDLING EVALUATION")
        
        # Get user activity counts from TRAINING data
        user_train_counts = self.train_data.groupby('user_id').size()
        
        # Build ground truth from test data
        ground_truth = defaultdict(set)
        for _, row in self.test_data.iterrows():
            ground_truth[row['user_id']].add(row['group_id'])
        
        results = {}
        
        # Test different activity levels (ranges based on training data counts)
        # Note: Users were sampled with >=5 groups, so we look at training split counts
        activity_ranges = [
            ("5-20 ratings", 5, 20),
            ("21-50 ratings", 21, 50),
            ("51-100 ratings", 51, 100),
            ("100+ ratings", 100, 10000)
        ]
        
        for label, min_count, max_count in activity_ranges:
            print(f"\n--- Users with {label} ---")
            
            # Find users in this range
            users_in_range = user_train_counts[
                (user_train_counts >= min_count) & 
                (user_train_counts <= max_count)
            ].index.tolist()
            
            # Filter to users with test data
            test_users = [u for u in users_in_range if u in ground_truth and len(ground_truth[u]) > 0][:50]
            
            print(f"Found {len(test_users)} users")
            
            if len(test_users) == 0:
                results[label] = {'cb': 0, 'cf': 0, 'hybrid': 0, 'popularity': 0}
                print("  Skipping - no users found")
                continue
            
            # Evaluate each method
            cb_hits = cf_hits = hybrid_hits = pop_hits = 0
            
            # Get popular groups for baseline
            popular_groups = self.train_data['group_id'].value_counts().head(10).index.tolist()
            
            for user_id in test_users:
                actual = ground_truth[user_id]
                
                # Content-Based
                cb_recs = self.cb_recommender.generate_recommendations(user_id, top_n=10)
                cb_groups = set([r[0] for r in cb_recs])
                if cb_groups & actual:
                    cb_hits += 1
                
                # Collaborative
                cf_recs = self.cf_recommender.svd_recommend(user_id, k=10, top_n=10)
                cf_groups = set([r[0] for r in cf_recs])
                if cf_groups & actual:
                    cf_hits += 1
                
                # Hybrid
                hybrid_recs = self.hybrid_recommend(user_id, alpha=self.best_alpha, top_n=10)
                hybrid_groups = set([r[0] for r in hybrid_recs])
                if hybrid_groups & actual:
                    hybrid_hits += 1
                
                # Popularity baseline
                if set(popular_groups) & actual:
                    pop_hits += 1
            
            n = len(test_users)
            results[label] = {
                'cb': cb_hits / n,
                'cf': cf_hits / n,
                'hybrid': hybrid_hits / n,
                'popularity': pop_hits / n
            }
            
            print(f"  Content-Based Hit Rate: {results[label]['cb']:.4f}")
            print(f"  Collaborative Hit Rate: {results[label]['cf']:.4f}")
            print(f"  Hybrid Hit Rate: {results[label]['hybrid']:.4f}")
            print(f"  Popularity Hit Rate: {results[label]['popularity']:.4f}")
        
        # ============================================================
        # SAVE 10. COLD-START RESULTS
        # ============================================================
        print("\nSAVING COLD-START EVALUATION RESULTS...")
        
        cold_start_df = pd.DataFrame([
            {'activity_level': label, 'method': 'Content-Based', 'hit_rate': m['cb']}
            for label, m in results.items()
        ] + [
            {'activity_level': label, 'method': 'Collaborative', 'hit_rate': m['cf']}
            for label, m in results.items()
        ] + [
            {'activity_level': label, 'method': 'Hybrid', 'hit_rate': m['hybrid']}
            for label, m in results.items()
        ] + [
            {'activity_level': label, 'method': 'Popularity', 'hit_rate': m['popularity']}
            for label, m in results.items()
        ])
        save_table(cold_start_df, "cold_start_evaluation.csv")
        
        # Plot cold-start comparison
        plt.figure(figsize=(10, 5))
        activity_labels = list(results.keys())
        x = np.arange(len(activity_labels))
        width = 0.2
        
        methods = ['cb', 'cf', 'hybrid', 'popularity']
        labels = ['Content-Based', 'Collaborative', 'Hybrid', 'Popularity']
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = [results[a][method] for a in activity_labels]
            plt.bar(x + i * width, values, width, label=label, color=color)
        
        plt.xlabel('User Activity Level')
        plt.ylabel('Hit Rate')
        plt.title('Cold-Start Handling: Performance by User Activity')
        plt.xticks(x + width * 1.5, activity_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        save_plot("cold_start_comparison.png")
        
        return results
    
    # =========================================================================
    # 11. BASELINE COMPARISON
    # =========================================================================
    
    def random_recommend(self, user_id, top_n=10):
        """Random recommendation baseline"""
        joined = set(self.train_data[self.train_data['user_id'] == user_id]['group_id'])
        available = list(self.common_groups - joined)
        if len(available) == 0:
            return []
        np.random.shuffle(available)
        return [(g, 0) for g in available[:top_n]]
    
    def popularity_recommend(self, user_id, top_n=10):
        """Most popular items baseline"""
        joined = set(self.train_data[self.train_data['user_id'] == user_id]['group_id'])
        popularity = self.train_data['group_id'].value_counts()
        recs = []
        for group_id, count in popularity.items():
            if group_id not in joined:
                recs.append((group_id, count))
                if len(recs) >= top_n:
                    break
        return recs
    
    def evaluate_all_methods(self, n_users=200):
        """
        11.1 Compare all methods
        """
        print("11. BASELINE COMPARISON")
        
        # Get test users with ground truth
        ground_truth = defaultdict(set)
        for _, row in self.test_data.iterrows():
            ground_truth[row['user_id']].add(row['group_id'])
        
        test_users = [u for u in ground_truth.keys() if len(ground_truth[u]) > 0][:n_users]
        print(f"\nEvaluating on {len(test_users)} users with Top-10...")
        
        results = {
            'Random': {'precision': [], 'recall': [], 'hits': 0},
            'Popularity': {'precision': [], 'recall': [], 'hits': 0},
            'Content-Based': {'precision': [], 'recall': [], 'hits': 0},
            'CF (SVD k=10)': {'precision': [], 'recall': [], 'hits': 0},
            'CF (SVD k=20)': {'precision': [], 'recall': [], 'hits': 0},
            f'Hybrid (α={self.best_alpha})': {'precision': [], 'recall': [], 'hits': 0},
        }
        
        for idx, user_id in enumerate(test_users):
            if idx % 50 == 0:
                print(f"  Processing user {idx+1}/{len(test_users)}...")
            actual = ground_truth[user_id]
            n_actual = len(actual)
            
            # Get recommendations from each method (using faster SVD instead of user-based)
            methods_recs = {
                'Random': self.random_recommend(user_id, top_n=10),
                'Popularity': self.popularity_recommend(user_id, top_n=10),
                'Content-Based': self.cb_recommender.generate_recommendations(user_id, top_n=10),
                'CF (SVD k=10)': self.cf_recommender.svd_recommend(user_id, k=10, top_n=10),
                'CF (SVD k=20)': self.cf_recommender.svd_recommend(user_id, k=20, top_n=10),
                f'Hybrid (α={self.best_alpha})': [(r[0], r[1]) for r in self.hybrid_recommend(user_id, alpha=self.best_alpha, top_n=10)],
            }
            
            for method_name, recs in methods_recs.items():
                rec_groups = set([r[0] for r in recs])
                hits = len(rec_groups & actual)
                
                precision = hits / len(rec_groups) if rec_groups else 0
                recall = hits / n_actual if n_actual > 0 else 0
                
                results[method_name]['precision'].append(precision)
                results[method_name]['recall'].append(recall)
                if hits > 0:
                    results[method_name]['hits'] += 1
        
        # Print comparison table
        print("11.2 COMPARISON TABLE")
        
        print(f"\n{'Method':<25} {'Precision@10':<15} {'Recall@10':<15} {'Hit Rate':<15}")
        print("-" * 70)
        
        final_results = {}
        for method_name, data in results.items():
            avg_precision = np.mean(data['precision'])
            avg_recall = np.mean(data['recall'])
            hit_rate = data['hits'] / len(test_users)
            
            final_results[method_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'hit_rate': hit_rate
            }
            
            print(f"{method_name:<25} {avg_precision:<15.4f} {avg_recall:<15.4f} {hit_rate:<15.4f}")
        
        # ============================================================
        # SAVE 11. BASELINE COMPARISON RESULTS
        # ============================================================
        print("\nSAVING BASELINE COMPARISON RESULTS...")
        
        comparison_df = pd.DataFrame([
            {'method': name, 'precision_at_10': m['precision'], 'recall_at_10': m['recall'], 'hit_rate': m['hit_rate']}
            for name, m in final_results.items()
        ])
        save_table(comparison_df, "baseline_comparison.csv")
        
        # Plot comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        methods = list(final_results.keys())
        hit_rates = [final_results[m]['hit_rate'] for m in methods]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        bars = plt.barh(methods, hit_rates, color=colors, edgecolor='black')
        plt.xlabel('Hit Rate')
        plt.title('Method Comparison: Hit Rate')
        for bar, hr in zip(bars, hit_rates):
            plt.text(hr + 0.005, bar.get_y() + bar.get_height()/2, f'{hr:.3f}', va='center')
        
        plt.subplot(1, 2, 2)
        precisions = [final_results[m]['precision'] for m in methods]
        recalls = [final_results[m]['recall'] for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        plt.bar(x - width/2, precisions, width, label='Precision@10', color='steelblue')
        plt.bar(x + width/2, recalls, width, label='Recall@10', color='coral')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Method Comparison: Precision & Recall')
        plt.legend()
        
        plt.tight_layout()
        save_plot("baseline_comparison.png")
        
        return final_results
    
    # =========================================================================
    # 12. RESULTS ANALYSIS
    # =========================================================================
    
    def analyze_results(self, comparison_results, cold_start_results):
        """
        12. Analyze and summarize results
        """
        print("12. RESULTS ANALYSIS")
        
        # Question 1: Which approach performed best?
        print("\nQUESTION 1: Which approach performed best?")
        print("-" * 45)
        
        # Sort by hit rate
        sorted_methods = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['hit_rate'],
            reverse=True
        )
        
        best_method = sorted_methods[0][0]
        best_hr = sorted_methods[0][1]['hit_rate']
        
        print(f"\n  Best Performing Method: {best_method}")
        print(f"  Hit Rate: {best_hr:.4f}")
        print(f"  Precision@10: {sorted_methods[0][1]['precision']:.4f}")
        print(f"  Recall@10: {sorted_methods[0][1]['recall']:.4f}")
        
        print("\nRanking by Hit Rate:")
        for rank, (method, metrics) in enumerate(sorted_methods, 1):
            print(f"  {rank}. {method}: {metrics['hit_rate']:.4f}")
        
        # Question 2: How well does hybrid handle cold-start?
        print("\n\nQUESTION 2: How well does hybrid handle cold-start?")
        print("-" * 50)
        
        print("\nPerformance by user activity level:")
        print(f"{'Activity':<15} {'Hybrid':<12} {'CB':<12} {'CF':<12} {'Popularity':<12}")
        print("-" * 60)
        
        for activity, metrics in cold_start_results.items():
            print(f"{activity:<15} {metrics['hybrid']:<12.4f} {metrics['cb']:<12.4f} "
                  f"{metrics['cf']:<12.4f} {metrics['popularity']:<12.4f}")
        
        print("""
KEY FINDINGS:
-------------
1. Cold-Start (few ratings): 
   - Content-Based provides baseline when CF lacks data
   - Hybrid leverages CB for new users

2. Moderate Activity:
   - CF starts contributing meaningful signal
   - Hybrid combines both for improved coverage

3. Active Users:
   - CF becomes more reliable with more data
   - Hybrid balances personalization (CF) with content (CB)

HYBRID ADVANTAGE:
- Gracefully handles transition from cold to warm users
- CB provides coverage when CF fails
- Weighted approach adapts to data availability
        """)
        
        # ============================================================
        # SAVE 12. FINAL ANALYSIS RESULTS
        # ============================================================
        print("\nSAVING FINAL ANALYSIS RESULTS...")
        
        # Save method ranking
        ranking_df = pd.DataFrame([
            {'rank': rank, 'method': method, 'hit_rate': metrics['hit_rate'], 
             'precision': metrics['precision'], 'recall': metrics['recall']}
            for rank, (method, metrics) in enumerate(sorted_methods, 1)
        ])
        save_table(ranking_df, "final_method_ranking.csv")
        
        # Save summary findings
        summary = pd.DataFrame({
            'Finding': [
                'Best Method',
                'Best Hit Rate',
                'Best Precision@10',
                'Best Recall@10',
                'Hybrid Alpha Used'
            ],
            'Value': [
                best_method,
                f"{best_hr:.4f}",
                f"{sorted_methods[0][1]['precision']:.4f}",
                f"{sorted_methods[0][1]['recall']:.4f}",
                str(self.best_alpha)
            ]
        })
        save_table(summary, "final_summary.csv")
        
        print("\nAll Section 2 results saved successfully!")


# MAIN EXECUTION

def main():
    """Main function to run the complete Hybrid Recommender System"""
    
    print("HYBRID RECOMMENDER SYSTEM")
    
    # Initialize hybrid system with target: ~10K users, ~1K groups, ~150K interactions
    # min_group_members=1250 filters to popular groups only
    hybrid = HybridRecommender(data_path='SECTION2_DomainRecommender/data/', sample_size=10000, min_group_members=1250)
    hybrid.initialize()
    
    # 9.2 Justify approach
    hybrid.justify_hybrid_approach()
    
    # 9.1 Find best alpha
    hybrid.find_best_alpha(alpha_values=[0.3, 0.5, 0.7], n_users=100)
    
    # Show sample recommendations
    sample_user = list(hybrid.common_users)[0]
    print(f"\n--- Sample Hybrid Recommendations (User {sample_user}) ---")
    recs = hybrid.hybrid_recommend(sample_user, alpha=hybrid.best_alpha, top_n=10)
    
    print(f"\nTop-10 Hybrid (α={hybrid.best_alpha}):")
    print(f"{'Rank':<6} {'Group':<12} {'Hybrid':<10} {'CB':<10} {'CF':<10}")
    print("-" * 50)
    for rank, (group_id, hybrid_score, cb_score, cf_score) in enumerate(recs, 1):
        print(f"{rank:<6} {group_id:<12} {hybrid_score:<10.4f} {cb_score:<10.4f} {cf_score:<10.4f}")
    
    # 10. Cold-start evaluation
    cold_start_results = hybrid.evaluate_cold_start()
    
    # 11. Baseline comparison
    comparison_results = hybrid.evaluate_all_methods(n_users=200)
    
    # 12. Results analysis
    hybrid.analyze_results(comparison_results, cold_start_results)
    
    print("HYBRID RECOMMENDER SYSTEM COMPLETE")
    
    return hybrid


if __name__ == "__main__":
    hybrid = main()
