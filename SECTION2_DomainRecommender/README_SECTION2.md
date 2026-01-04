# Section 2: Interest-Based Group Recommendation System

## Overview

This section implements a **domain-specific recommender system** for recommending interest-based groups to users. The system combines three approaches:

1. **Content-Based Filtering** - Using TF-IDF on group tags
2. **Collaborative Filtering** - User-based CF with SVD
3. **Hybrid System** - Weighted combination of both approaches

---

## Project Structure

```
SECTION2_DomainRecommender/
├── code/
│   ├── main.py                 # Main entry point (runs all steps)
│   ├── data_preprocessing.py   # Data loading and statistics
│   ├── content_based.py        # Content-Based Filtering (TF-IDF)
│   ├── collaborative.py        # Collaborative Filtering (User-Based + SVD)
│   └── hybrid.py               # Hybrid Recommender System
├── data/
│   ├── user_group.csv          # User-Group memberships
│   ├── user_tag.csv            # User tag preferences
│   ├── group_tag.csv           # Group-Tag associations
│   ├── tag_text.csv            # Tag text descriptions
│   ├── user_event.csv          # User-Event interactions
│   └── event_group.csv         # Event-Group mappings
└── results/
    ├── tables/                 # CSV output files
    └── plots/                  # Visualization images
```

---

## Dataset Description

| File | Columns | Description |
|------|---------|-------------|
| `user_group.csv` | user_id, group_id | Which groups each user has joined |
| `user_tag.csv` | user_id, tag_id | Tags that users are interested in |
| `group_tag.csv` | group_id, tag_id | Tags associated with each group |
| `tag_text.csv` | tag_id, tag_text | Text description of each tag |
| `user_event.csv` | user_id, event_id | Events users have attended |
| `event_group.csv` | event_id, group_id | Which group hosts each event |

---

## Pipeline Steps

### Step 1: Data Preprocessing
**File:** `data_preprocessing.py`

- Load all datasets
- Compute statistics:
  - Total users, groups, tags
  - User activity distribution
  - Group popularity distribution
  - **Data Sparsity** (~99.8%)

**Outputs:**
- `tables/Data_Preprocessing/dataset_summary.csv`
- `tables/Data_Preprocessing/user_group_counts.csv`
- `tables/Data_Preprocessing/group_member_counts.csv`
- `plots/Data_Preprocessing/data_distribution.png`

---

### Step 2: Content-Based Filtering
**File:** `content_based.py`

#### 3.1 TF-IDF Feature Extraction
- Concatenate all tags for each group into a document
- Apply TF-IDF vectorization:
  ```python
  TfidfVectorizer(
      lowercase=True,
      stop_words='english',
      max_features=5000,
      min_df=2,
      max_df=0.95
  )
  ```
- Create **Item-Feature Matrix** (Groups × TF-IDF Features)

#### 4. User Profile Construction
- Aggregate user tag preferences
- Transform using the same TF-IDF vectorizer
- Build **User Profile Matrix**

#### 5. Recommendation Generation
- Compute **Cosine Similarity** between user profiles and group features
- Rank groups by similarity score

#### 6. Cold-Start Handling
- Strategy: Recommend popular items for new users

**Outputs:**
- `tables/Content_Based/group_tag_documents.csv`
- `tables/Content_Based/tfidf_vocabulary.csv`
- `tables/Content_Based/user_profile_stats.csv`
- `plots/Content_Based/tfidf_distribution.png`
- `plots/Content_Based/user_profile_distribution.png`

---

### Step 3: Collaborative Filtering
**File:** `collaborative.py`

#### 8.1 User-Based Collaborative Filtering
- Create **User-Item Matrix** (binary: 1 if joined, 0 otherwise)
- Compute **User-User Similarity** using Cosine Similarity
- Predict ratings using weighted average of k-nearest neighbors:

$$\hat{r}_{u,i} = \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot r_{v,i}}{\sum_{v \in N_k(u)} sim(u,v)}$$

#### 8.2 SVD Matrix Factorization
- Apply SVD with k = [10, 20] latent factors
- GPU acceleration using PyTorch (if available)
- Reconstruct ratings matrix:

$$\hat{R} = U_k \Sigma_k V_k^T$$

**Outputs:**
- `tables/Collaborative_Filtering/user_item_matrix_stats.csv`
- `tables/Collaborative_Filtering/user_similarity_stats.csv`
- `tables/Collaborative_Filtering/top_similar_user_pairs.csv`
- `tables/Collaborative_Filtering/svd_singular_values_k*.csv`
- `plots/Collaborative_Filtering/user_similarity_distribution.png`
- `plots/Collaborative_Filtering/svd_variance_explained_k*.png`

---

### Step 4: Hybrid System
**File:** `hybrid.py`

#### 9.1 Weighted Hybrid Recommendation
Combines both approaches using a weighted formula:

$$Score = \alpha \times CB + (1 - \alpha) \times CF$$

Where:
- **α** = Weight for Content-Based (tuned automatically)
- **CB** = Normalized Content-Based score
- **CF** = Normalized Collaborative Filtering score

#### Alpha Tuning
- Test α values: [0.3, 0.5, 0.7]
- Evaluate using **Hit Rate** on test set
- Select best α based on performance

#### 9.2 Justification for Hybrid Approach

| Domain Characteristic | Why Hybrid Works |
|----------------------|------------------|
| High Sparsity (99.8%) | CB provides fallback when CF fails |
| Rich Content Features | 77,810+ unique tags available |
| Implicit Feedback | Both CB and CF work on same scale |
| Cold-Start Problem | CB uses tag preferences for new users |

#### 10. Cold-Start Evaluation
- New users: Rely more on Content-Based
- Active users: Rely more on Collaborative Filtering

#### 11. Baseline Comparison
Compare hybrid system against:
- Content-Based only
- Collaborative Filtering only
- Random recommendations
- Most Popular items

**Outputs:**
- `tables/Hybrid_System/alpha_tuning_results.csv`
- `tables/Hybrid_System/sample_hybrid_recommendations.csv`
- `plots/Hybrid_System/alpha_tuning.png`

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
cd SECTION2_DomainRecommender/code
python main.py
```

### Run Individual Components
```bash
python data_preprocessing.py   # Step 1 only
python content_based.py        # Step 2 only
python collaborative.py        # Step 3 only
python hybrid.py               # Step 4 only
```

---

## Configuration

In `main.py`, you can adjust:

```python
SAMPLE_SIZE = 20000  # Number of users to sample (adjust based on RAM)
```

**Memory Guidelines:**
| Sample Size | RAM Required |
|-------------|--------------|
| 10,000 users | ~4 GB |
| 20,000 users | ~8 GB |
| 30,000 users | ~16 GB |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate** | % of users with at least one correct recommendation in top-N |
| **Precision@K** | Fraction of recommended items that are relevant |
| **Recall@K** | Fraction of relevant items that are recommended |

---

## Key Algorithms

### TF-IDF (Term Frequency-Inverse Document Frequency)
$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

### Cosine Similarity
$$\text{sim}(A,B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### SVD Decomposition
$$R = U \Sigma V^T$$

### Weighted Hybrid Score
$$\text{Score} = \alpha \times \text{CB}_{norm} + (1-\alpha) \times \text{CF}_{norm}$$

---

## Output Summary

| Directory | Contents |
|-----------|----------|
| `results/tables/Data_Preprocessing/` | Dataset statistics |
| `results/tables/Content_Based/` | TF-IDF features, user profiles |
| `results/tables/Collaborative_Filtering/` | User similarity, SVD results |
| `results/tables/Hybrid_System/` | Alpha tuning, final recommendations |
| `results/plots/` | All visualizations |

---
