# ============================================================
# Main Entry Point - Interest-Based Group Recommendation
# ============================================================
# Complete workflow for Section 2:
# 1. Data Preprocessing
# 2. Unified Data Sampling (ensures CB & CF use same users)
# 3. Content-Based Filtering
# 4. Collaborative Filtering
# 5. Hybrid System
# 6. Full Evaluation
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import load_datasets, compute_statistics
from content_based import ContentBasedRecommender, run_numerical_example
from collaborative import CollaborativeFilteringRecommender
from hybrid import HybridRecommender
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "SECTION2_DomainRecommender/data/"
RESULTS_DIR = "SECTION2_DomainRecommender/results/"

# Create all result directories
os.makedirs(f"{RESULTS_DIR}tables/Data_Preprocessing", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}tables/Content_Based", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}tables/Collaborative_Filtering", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}tables/Hybrid_System", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}plots/Data_Preprocessing", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}plots/Content_Based", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}plots/Collaborative_Filtering", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}plots/Hybrid_System", exist_ok=True)

print("RESULTS WILL BE SAVED TO:")
print(f"  Tables: {RESULTS_DIR}tables/")
print(f"  Plots:  {RESULTS_DIR}plots/")

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

def run_preprocessing():
    """Run data preprocessing and statistics"""
    print("\nSTEP 1: DATA PREPROCESSING\n")
    datasets = load_datasets()
    stats = compute_statistics(datasets)
    return datasets, stats

# ============================================================
# 2. UNIFIED DATA SAMPLING
# ============================================================

def create_unified_sample(sample_size=20000, min_group_members=50, min_user_interactions=5):
    """
    Create a unified sample ensuring CB and CF use the SAME users and groups.
    This fixes the user mismatch problem between CB and CF.
    """
    print("\nSTEP 2: UNIFIED DATA SAMPLING")
    
    DATA_DIR = "SECTION2_DomainRecommender/data/"
    
    # Load all data
    print("\nLoading datasets...")
    user_group = pd.read_csv(f"{DATA_DIR}user_group.csv", header=None, names=['user_id', 'group_id']).drop_duplicates()
    user_tag = pd.read_csv(f"{DATA_DIR}user_tag.csv", header=None, names=['user_id', 'tag_id'])
    group_tag = pd.read_csv(f"{DATA_DIR}group_tag.csv", header=None, names=['group_id', 'tag_id'])
    tag_text = pd.read_csv(f"{DATA_DIR}tag_text.csv", header=None, names=['tag_id', 'tag_text'])
    
    print(f"Total user-group interactions: {len(user_group):,}")
    print(f"Total users: {user_group['user_id'].nunique():,}")
    print(f"Total groups: {user_group['group_id'].nunique():,}")
    
    # Step 1: Filter to popular groups
    print(f"\n--- Filtering groups with >= {min_group_members} members ---")
    group_counts = user_group['group_id'].value_counts()
    popular_groups = group_counts[group_counts >= min_group_members].index.tolist()
    user_group = user_group[user_group['group_id'].isin(popular_groups)]
    print(f"Groups after filtering: {len(popular_groups):,}")
    
    # Step 2: Get users who have BOTH group memberships AND tag preferences
    print("\n--- Finding users with both group memberships AND tag preferences ---")
    users_with_groups = set(user_group['user_id'].unique())
    users_with_tags = set(user_tag['user_id'].unique())
    users_with_both = users_with_groups.intersection(users_with_tags)
    print(f"Users with group memberships: {len(users_with_groups):,}")
    print(f"Users with tag preferences: {len(users_with_tags):,}")
    print(f"Users with BOTH: {len(users_with_both):,}")
    
    # Step 3: Filter to active users (minimum interactions)
    print(f"\n--- Filtering users with >= {min_user_interactions} interactions ---")
    user_group_filtered = user_group[user_group['user_id'].isin(users_with_both)]
    user_counts = user_group_filtered['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_user_interactions].index[:sample_size].tolist()
    print(f"Active users selected: {len(active_users):,}")
    
    # Step 4: Final filtering
    user_group_final = user_group[user_group['user_id'].isin(active_users)]
    final_groups = user_group_final['group_id'].unique().tolist()
    
    # Filter related dataframes
    user_tag_final = user_tag[user_tag['user_id'].isin(active_users)]
    group_tag_final = group_tag[group_tag['group_id'].isin(final_groups)]
    
    print(f"\n--- UNIFIED SAMPLE STATISTICS ---")
    print(f"Final users: {len(active_users):,}")
    print(f"Final groups: {len(final_groups):,}")
    print(f"Final interactions: {len(user_group_final):,}")
    print(f"User-tag preferences: {len(user_tag_final):,}")
    print(f"Group-tag associations: {len(group_tag_final):,}")
    
    # Create train/test split
    print("\n--- Creating train/test split (80/20) ---")
    train_data, test_data = train_test_split(user_group_final, test_size=0.2, random_state=42)
    print(f"Train interactions: {len(train_data):,}")
    print(f"Test interactions: {len(test_data):,}")
    
    return {
        'user_group': user_group_final,
        'user_tag': user_tag_final,
        'group_tag': group_tag_final,
        'tag_text': tag_text,
        'train_data': train_data,
        'test_data': test_data,
        'users': active_users,
        'groups': final_groups
    }

# ============================================================
# 3. CONTENT-BASED FILTERING (with unified data)
# ============================================================

def run_content_based_unified(unified_data):
    """Run content-based filtering with unified data"""
    print("\nSTEP 3: CONTENT-BASED FILTERING")
    
    DATA_DIR = "SECTION2_DomainRecommender/data/"
    cb = ContentBasedRecommender(data_path=DATA_DIR)
    
    # Use unified data directly (don't load from files)
    cb.user_group_df = unified_data['train_data'].copy()
    cb.user_tag_df = unified_data['user_tag'].copy()
    cb.group_tag_df = unified_data['group_tag'].copy()
    cb.tag_text_df = unified_data['tag_text'].copy()
    
    print(f"\nUsing unified data:")
    print(f"  Users: {cb.user_tag_df['user_id'].nunique():,}")
    print(f"  Groups: {cb.group_tag_df['group_id'].nunique():,}")
    
    # Build features (3.1, 3.2, 3.3)
    cb.create_group_text_features()
    cb.build_user_profiles()
    cb.handle_cold_start(strategy='popular_items')
    
    # 5.2 Top-10 and Top-20 recommendations
    test_user = unified_data['users'][0]
    cb.recommend_top_n(test_user)
    
    # 6.1 & 6.2 k-NN Implementation and Comparison
    print("\n6. k-NEAREST NEIGHBORS (k-NN)")
    cb.build_item_knn(k_values=[10, 20])
    cb.compare_approaches(test_user)
    
    # Sample CB recommendation output
    recommendations = cb.generate_recommendations(test_user, top_n=10)
    print(f"\nSample CB recommendations for user {test_user}:")
    for i, (group_id, score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}: {score:.4f}")
    
    return cb

# ============================================================
# 4. COLLABORATIVE FILTERING (with unified data)
# ============================================================

def run_collaborative_unified(unified_data):
    """Run collaborative filtering with unified data"""
    print("\nSTEP 4: COLLABORATIVE FILTERING")
    
    DATA_DIR = "SECTION2_DomainRecommender/data/"
    cf = CollaborativeFilteringRecommender(data_path=DATA_DIR)
    
    # Use unified data directly
    cf.user_group_df = unified_data['train_data'].copy()
    
    print(f"\nUsing unified data:")
    print(f"  Users: {cf.user_group_df['user_id'].nunique():,}")
    print(f"  Groups: {cf.user_group_df['group_id'].nunique():,}")
    
    # Build models
    cf.create_user_item_matrix()
    cf.compute_user_similarity()
    cf.apply_svd(k_values=[10, 20])  # k=10, k=20 only (memory safe)
    
    # Test recommendation
    test_user = cf.user_ids[0]
    recommendations = cf.svd_recommend(test_user, k=20, top_n=10)
    print(f"\nSample CF recommendations for user {test_user}:")
    for i, (group_id, score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}: {score:.4f}")
    
    return cf

# ============================================================
# 5. HYBRID SYSTEM (with unified data)
# ============================================================

def run_hybrid_unified(unified_data, cb_recommender, cf_recommender):
    """Run hybrid recommender system with unified data"""
    print("\nSTEP 5: HYBRID SYSTEM")
    
    DATA_DIR = "SECTION2_DomainRecommender/data/"
    RESULTS_DIR = "SECTION2_DomainRecommender/results/"
    
    hybrid = HybridRecommender(data_path=DATA_DIR, sample_size=len(unified_data['users']), min_group_members=50)
    
    # Set unified data directly
    hybrid.train_data = unified_data['train_data']
    hybrid.test_data = unified_data['test_data']
    hybrid.cb_recommender = cb_recommender
    hybrid.cf_recommender = cf_recommender
    
    # All users are now common (since we used unified sampling)
    hybrid.common_users = set(unified_data['users'])
    hybrid.common_groups = set(unified_data['groups'])
    
    # Verify CB and CF alignment
    cb_users = set(cb_recommender.user_profiles.keys())
    cf_users = set(cf_recommender.user_id_to_idx.keys())
    actual_common = hybrid.common_users.intersection(cb_users).intersection(cf_users)
    
    print(f"\n--- ALIGNMENT CHECK ---")
    print(f"Unified users: {len(hybrid.common_users):,}")
    print(f"CB users with profiles: {len(cb_users):,}")
    print(f"CF users in matrix: {len(cf_users):,}")
    print(f"Users available in ALL systems: {len(actual_common):,}")
    
    # Update common_users to actual intersection
    hybrid.common_users = actual_common
    
    if len(actual_common) < 100:
        print("\nWARNING: Very few common users. Check data alignment!")
    else:
        print(f"\nGood alignment: {len(actual_common):,} users available for hybrid evaluation")
    
    # Alpha tuning with more users
    print("\n")
    hybrid.find_best_alpha(alpha_values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], n_users=min(200, len(actual_common)))
    
    # Sample recommendations
    test_user = list(hybrid.common_users)[0]
    recommendations = hybrid.hybrid_recommend(test_user, alpha=hybrid.best_alpha, top_n=10)
    
    print(f"\nSample hybrid recommendations for user {test_user}:")
    for i, (group_id, hybrid_score, cb_score, cf_score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}:")
        print(f"      Hybrid: {hybrid_score:.4f} | CB: {cb_score:.4f} | CF: {cf_score:.4f}")
    
    # Save sample recommendations
    recs_df = pd.DataFrame(
        [(group_id, hybrid_score, cb_score, cf_score) 
         for group_id, hybrid_score, cb_score, cf_score in recommendations],
        columns=['group_id', 'hybrid_score', 'content_based_score', 'collaborative_score']
    )
    recs_df.insert(0, 'user_id', test_user)
    recs_df.insert(1, 'rank', range(1, len(recs_df) + 1))
    recs_df.to_csv(f"{RESULTS_DIR}tables/Hybrid_System/sample_hybrid_recommendations.csv", index=False)
    
    # ============================================================
    # FULL EVALUATION
    # ============================================================
    print("\nSTEP 6: FULL EVALUATION WITH MULTIPLE METRICS")
    
    # Cold-start evaluation
    cold_start_results = hybrid.evaluate_cold_start()
    
    # Baseline comparison with all metrics
    comparison_results = hybrid.evaluate_all_methods(n_users=min(300, len(actual_common)))
    
    # Results analysis
    hybrid.analyze_results(comparison_results, cold_start_results)
    
    return hybrid

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\nSECTION 2: INTEREST-BASED GROUP RECOMMENDATION")
    
    # Configuration
    SAMPLE_SIZE = 20000
    MIN_GROUP_MEMBERS = 50
    MIN_USER_INTERACTIONS = 5
    
    # Step 1: Data Preprocessing
    datasets, stats = run_preprocessing()
    
    # Step 2: Create unified sample (ensures CB & CF use same users)
    unified_data = create_unified_sample(
        sample_size=SAMPLE_SIZE,
        min_group_members=MIN_GROUP_MEMBERS,
        min_user_interactions=MIN_USER_INTERACTIONS
    )
    
    # Step 3: Content-Based Filtering
    cb_recommender = run_content_based_unified(unified_data)
    
    # Step 3b: Run Numerical Example (Section 7)
    print("\nSTEP 7: COMPLETE NUMERICAL EXAMPLE")
    run_numerical_example()
    
    # Step 4: Collaborative Filtering
    cf_recommender = run_collaborative_unified(unified_data)
    
    # Step 5 & 6: Hybrid System & Evaluation
    hybrid_recommender = run_hybrid_unified(unified_data, cb_recommender, cf_recommender)
    
    print("\nALL STEPS COMPLETED SUCCESSFULLY")
    print("\nRESULTS SAVED IN:")
    RESULTS_DIR = "SECTION2_DomainRecommender/results/"
    print(f"  Data Preprocessing: {RESULTS_DIR}tables/Data_Preprocessing/")
    print(f"  Content-Based:      {RESULTS_DIR}tables/Content_Based/")
    print(f"  Collaborative:      {RESULTS_DIR}tables/Collaborative_Filtering/")
    print(f"  Hybrid System:      {RESULTS_DIR}tables/Hybrid_System/")
