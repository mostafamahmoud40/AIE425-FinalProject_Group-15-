# ============================================================
# Part 1: Data Preprocessing
# Domain: Interest-Based Group Formation (Meetup.com)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "SECTION2_DomainRecommender/data"
TABLE_DIR = "SECTION2_DomainRecommender/results/tables/Data_Preprocessing"
PLOT_DIR = "SECTION2_DomainRecommender/results/plots/Data_Preprocessing"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

pd.options.display.float_format = "{:.2f}".format

def save_table(df, filename):
    """Save DataFrame to CSV with 2 decimal formatting"""
    df2 = df.copy()
    float_cols = df2.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        df2[float_cols] = df2[float_cols].round(2)
    df2.to_csv(f"{TABLE_DIR}/{filename}", index=False)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_datasets():
    """Load all datasets and return as dictionary"""
    print("LOADING DATASETS")
    
    datasets = {}
    
    # Load event_group.csv
    datasets['event_group'] = pd.read_csv(
        f"{DATA_DIR}/event_group.csv",
        header=None,
        names=['event_id', 'group_id']
    )
    print(f"Event-Group mappings: {len(datasets['event_group']):,}")
    
    # Load group_tag.csv
    datasets['group_tag'] = pd.read_csv(
        f"{DATA_DIR}/group_tag.csv",
        header=None,
        names=['group_id', 'tag_id']
    )
    print(f"Group-Tag mappings: {len(datasets['group_tag']):,}")
    
    # Load tag_text.csv
    datasets['tag_text'] = pd.read_csv(
        f"{DATA_DIR}/tag_text.csv",
        header=None,
        names=['tag_id', 'tag_text']
    )
    print(f"Tags: {len(datasets['tag_text']):,}")
    
    # Load user_event.csv
    datasets['user_event'] = pd.read_csv(
        f"{DATA_DIR}/user_event.csv",
        header=None,
        names=['user_id', 'event_id']
    )
    print(f"User-Event interactions: {len(datasets['user_event']):,}")
    
    # Load user_group.csv
    datasets['user_group'] = pd.read_csv(
        f"{DATA_DIR}/user_group.csv",
        header=None,
        names=['user_id', 'group_id']
    )
    datasets['user_group'] = datasets['user_group'].drop_duplicates()
    print(f"User-Group memberships: {len(datasets['user_group']):,}")
    
    # Load user_tag.csv
    datasets['user_tag'] = pd.read_csv(
        f"{DATA_DIR}/user_tag.csv",
        header=None,
        names=['user_id', 'tag_id']
    )
    print(f"User-Tag preferences: {len(datasets['user_tag']):,}")
    
    return datasets

# ============================================================
# 2. DATA STATISTICS
# ============================================================

def compute_statistics(datasets):
    """Compute and display dataset statistics"""
    print("\nDATA STATISTICS")
    
    # User statistics
    n_users = datasets['user_group']['user_id'].nunique()
    print(f"\nTotal unique users: {n_users:,}")
    
    # Group statistics
    n_groups = datasets['user_group']['group_id'].nunique()
    print(f"Total unique groups: {n_groups:,}")
    
    # Tag statistics
    n_tags = datasets['tag_text']['tag_id'].nunique()
    print(f"Total unique tags: {n_tags:,}")
    
    # User-Group interaction statistics
    user_counts = datasets['user_group']['user_id'].value_counts()
    print(f"\nUser-Group memberships:")
    print(f"  Mean: {user_counts.mean():.2f}")
    print(f"  Median: {user_counts.median():.2f}")
    print(f"  Min: {user_counts.min()}")
    print(f"  Max: {user_counts.max()}")
    
    # Group popularity statistics
    group_counts = datasets['user_group']['group_id'].value_counts()
    print(f"\nGroup popularity (members per group):")
    print(f"  Mean: {group_counts.mean():.2f}")
    print(f"  Median: {group_counts.median():.2f}")
    print(f"  Min: {group_counts.min()}")
    print(f"  Max: {group_counts.max()}")
    
    # Data sparsity
    total_possible = n_users * n_groups
    actual_interactions = len(datasets['user_group'])
    sparsity = (1 - actual_interactions / total_possible) * 100
    print(f"\nData sparsity: {sparsity:.4f}%")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    # Save user statistics
    user_stats_df = user_counts.reset_index()
    user_stats_df.columns = ['user_id', 'n_groups_joined']
    save_table(user_stats_df, "user_group_counts.csv")
    
    # Save group statistics
    group_stats_df = group_counts.reset_index()
    group_stats_df.columns = ['group_id', 'n_members']
    save_table(group_stats_df, "group_member_counts.csv")
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Total Users', 'Total Groups', 'Total Tags', 'Total Interactions', 'Sparsity (%)'],
        'Value': [n_users, n_groups, n_tags, actual_interactions, sparsity]
    })
    save_table(summary_df, "dataset_summary.csv")
    
    # Save user-group interaction distribution plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(user_counts.values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Groups Joined')
    plt.ylabel('Number of Users')
    plt.title('Distribution of User Activity')
    
    plt.subplot(1, 2, 2)
    plt.hist(group_counts.values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Members')
    plt.ylabel('Number of Groups')
    plt.title('Distribution of Group Popularity')
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/data_distribution.png", dpi=150)
    plt.close()
    
    return {
        'n_users': n_users,
        'n_groups': n_groups,
        'n_tags': n_tags,
        'sparsity': sparsity
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("SECTION 2: DATA PREPROCESSING")
    
    # Load datasets
    datasets = load_datasets()
    
    # Compute statistics
    stats = compute_statistics(datasets)
    
    print("\nData preprocessing completed successfully.")
