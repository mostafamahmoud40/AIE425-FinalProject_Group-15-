# ============================================================
# Main Entry Point - Interest-Based Group Recommendation
# ============================================================
# Complete workflow for Section 2:
# 1. Data Preprocessing
# 2. Content-Based Filtering
# 3. Collaborative Filtering
# 4. Hybrid System
# ============================================================

import os
import pandas as pd
import numpy as np
from data_preprocessing import load_datasets, compute_statistics
from content_based import ContentBasedRecommender
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

print("=" * 60)
print("RESULTS WILL BE SAVED TO:")
print("=" * 60)
print(f"  Tables: {RESULTS_DIR}tables/")
print(f"  Plots:  {RESULTS_DIR}plots/")
print("=" * 60)

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
# 2. CONTENT-BASED FILTERING
# ============================================================

def run_content_based(sample_size=5000):
    """Run content-based filtering"""
    print("\nSTEP 2: CONTENT-BASED FILTERING\n")
    cb = ContentBasedRecommender(data_path=DATA_DIR)
    cb.load_data(sample_size=sample_size)
    cb.create_group_text_features()
    cb.build_user_profiles()
    cb.handle_cold_start(strategy='popular_items')
    
    test_user = cb.user_tag_df['user_id'].iloc[0]
    recommendations = cb.generate_recommendations(test_user, top_n=10)
    print(f"\nSample recommendations for user {test_user}:")
    for i, (group_id, score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}: {score:.4f}")
    return cb

# ============================================================
# 3. COLLABORATIVE FILTERING
# ============================================================

def run_collaborative(sample_size=5000):
    """Run collaborative filtering"""
    print("\nSTEP 3: COLLABORATIVE FILTERING\n")
    cf = CollaborativeFilteringRecommender(data_path=DATA_DIR)
    cf.load_data(sample_size=sample_size)
    cf.create_user_item_matrix()
    cf.compute_user_similarity()
    cf.apply_svd(k_values=[10, 20])
    
    test_user = cf.user_ids[0]
    recommendations = cf.user_based_recommend(test_user, top_n=10)
    print(f"\nSample recommendations for user {test_user}:")
    for i, (group_id, score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}: {score:.4f}")
    return cf

# ============================================================
# 4. HYBRID SYSTEM
# ============================================================

def run_hybrid(sample_size=10000, cb_recommender=None, cf_recommender=None):
    """Run hybrid recommender system"""
    print("\nSTEP 4: HYBRID SYSTEM\n")
    hybrid = HybridRecommender(
        data_path=DATA_DIR,
        sample_size=sample_size,
        min_group_members=50
    )
    # Pass existing recommenders to avoid duplicate computation
    hybrid.initialize(cb_recommender=cb_recommender, cf_recommender=cf_recommender)
    hybrid.find_best_alpha(alpha_values=[0.3, 0.5, 0.7], n_users=100)
    
    test_user = list(hybrid.common_users)[0]
    recommendations = hybrid.hybrid_recommend(
        test_user,
        alpha=hybrid.best_alpha,
        top_n=10
    )
    print(f"\nSample hybrid recommendations for user {test_user}:")
    for i, (group_id, hybrid_score, cb_score, cf_score) in enumerate(recommendations[:5], 1):
        print(f"  {i}. Group {group_id}:")
        print(f"      Hybrid: {hybrid_score:.4f} | CB: {cb_score:.4f} | CF: {cf_score:.4f}")
    
    # Save sample recommendations to CSV
    print("\nSAVING SAMPLE HYBRID RECOMMENDATIONS...")
    recs_df = pd.DataFrame(
        [(group_id, hybrid_score, cb_score, cf_score) 
         for group_id, hybrid_score, cb_score, cf_score in recommendations],
        columns=['group_id', 'hybrid_score', 'content_based_score', 'collaborative_score']
    )
    recs_df.insert(0, 'user_id', test_user)
    recs_df.insert(1, 'rank', range(1, len(recs_df) + 1))
    recs_df.to_csv(f"{RESULTS_DIR}tables/Hybrid_System/sample_hybrid_recommendations.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}tables/Hybrid_System/sample_hybrid_recommendations.csv")
    
    return hybrid

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\nSECTION 2: INTEREST-BASED GROUP RECOMMENDATION\n")
    
    # Increased sample sizes for 16GB RAM system
    # Safe: 20K users, ~1500 groups, ~300K interactions
    # Maximum: 30K users, ~2000 groups, ~500K interactions
    SAMPLE_SIZE = 20000  # Increased from 10000 (safe for 16GB RAM)
    
    datasets, stats = run_preprocessing()
    cb_recommender = run_content_based(sample_size=SAMPLE_SIZE)
    cf_recommender = run_collaborative(sample_size=SAMPLE_SIZE)
    # Pass existing recommenders to hybrid to avoid duplicate 8.1/8.2 computation
    hybrid_recommender = run_hybrid(sample_size=SAMPLE_SIZE, cb_recommender=cb_recommender, cf_recommender=cf_recommender)
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nRESULTS SAVED IN:")
    print("-" * 60)
    print("Data Preprocessing:")
    print(f"   Tables: {RESULTS_DIR}tables/Data_Preprocessing/")
    print(f"   Plots:  {RESULTS_DIR}plots/Data_Preprocessing/")
    print("\nContent-Based Filtering:")
    print(f"   Tables: {RESULTS_DIR}tables/Content_Based/")
    print(f"   Plots:  {RESULTS_DIR}plots/Content_Based/")
    print("\nCollaborative Filtering:")
    print(f"   Tables: {RESULTS_DIR}tables/Collaborative_Filtering/")
    print(f"   Plots:  {RESULTS_DIR}plots/Collaborative_Filtering/")
    print("\nHybrid System:")
    print(f"   Tables: {RESULTS_DIR}tables/Hybrid_System/")
    print(f"   Plots:  {RESULTS_DIR}plots/Hybrid_System/")
    print("=" * 60)
