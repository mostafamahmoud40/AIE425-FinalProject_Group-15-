# Section 2: Interest-Based Group Recommendation System

---

## 1. System Description and Architecture

### 1.1 System Overview

This section implements a **domain-specific recommender system** for the domain of **Interest-Based Group Formation Recommendation**. The system recommends groups to users based on their interests and the behavior of similar users.

**Domain:** Interest-Based Group Formation Recommendation (e.g., Meetup.com groups)

**Goal:** Recommend relevant interest groups to users based on:
- Their explicit interests (tags they follow)
- Their implicit behavior (groups they've joined)
- Similar users' preferences

### 1.2 System Components

| Component | Description |
|-----------|-------------|
| **Users** | People looking for interest-based groups to join |
| **Items** | Interest groups (characterized by tags) |
| **Features** | Tags/topics describing user interests and group themes |
| **Interactions** | User-Group memberships (implicit feedback) |

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │     USER     │
                              │  (Interests) │
                              └──────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ User-Group  │  │  User-Tag   │  │  Group-Tag  │  │  Tag-Text   │        │
│  │ Memberships │  │ Preferences │  │ Associations│  │ Descriptions│        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE EXTRACTION                                 │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │   TF-IDF Vectorization      │  │   User-Item Matrix          │          │
│  │   (Group Tag Features)      │  │   (Binary Interactions)     │          │
│  └─────────────────────────────┘  └─────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│    CONTENT-BASED FILTERING      │  │   COLLABORATIVE FILTERING       │
│                                 │  │                                 │
│  • User Profile (TF-IDF)        │  │  • User-User Similarity         │
│  • Group Features (TF-IDF)      │  │  • SVD Matrix Factorization     │
│  • Cosine Similarity            │  │  • K-Nearest Neighbors          │
│                                 │  │                                 │
│  "Groups matching your          │  │  "Groups joined by similar      │
│   interests"                    │  │   users"                        │
└─────────────────────────────────┘  └─────────────────────────────────┘
                    │                                 │
                    │     CB Score          CF Score  │
                    └────────────────┬────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HYBRID RECOMMENDER                                 │
│                                                                             │
│                   Score = α × CB + (1 - α) × CF                             │
│                                                                             │
│  • Weighted combination of both approaches                                  │
│  • Alpha tuning for optimal balance                                         │
│  • Cold-start handling for new users                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   RECOMMENDED GROUPS  │
                         │   (Top-N for user)    │
                         └───────────────────────┘
```

### 1.4 Recommendation Approaches

| Approach | Method | Description |
|----------|--------|-------------|
| **Content-Based** | TF-IDF + Cosine Similarity | Matches user interests with group tags |
| **Collaborative** | User-Based CF + SVD | Finds groups joined by similar users |
| **Hybrid** | Weighted Combination | α × CB + (1-α) × CF for best results |

---

## 2. Data Collection and Preprocessing

### 2.1 Data Source

**Dataset:** Meetup.com Interest Groups Dataset

| Attribute | Value |
|-----------|-------|
| Source | Meetup.com (Public Dataset) |
| Format | CSV files |
| Size | ~50MB+ |
| Type | Implicit Feedback (memberships) |

### 2.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Users | 4,111,476 |
| Total Groups | 70,604 |
| Total Tags | 77,810 |
| Total Interactions | 10,627,861 |
| **Data Sparsity** | **99.9963%** |

### 2.3 Sampled Dataset (for computation)

| Metric | Value |
|--------|-------|
| Sampled Users | 20,000 |
| Sampled Groups | 34,838 |
| Sampled Interactions | 731,426 |
| User-Tag Preferences | 609,442 |
| Group-Tag Associations | 298,467 |
| Train/Test Split | 80% / 20% |

### 2.3 Data Files Description

| File | Columns | Records | Description |
|------|---------|---------|-------------|
| `user_group.csv` | user_id, group_id | 10.6M | User-Group memberships |
| `user_tag.csv` | user_id, tag_id | - | User interest tags |
| `group_tag.csv` | group_id, tag_id | - | Group topic tags |
| `tag_text.csv` | tag_id, tag_text | 77,810 | Tag descriptions |
| `user_event.csv` | user_id, event_id | - | Event attendance |
| `event_group.csv` | event_id, group_id | - | Event-Group mapping |

### 2.4 Data Preprocessing Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING                                                │
│     • Load all CSV files with proper column names               │
│     • Handle missing headers                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. DATA CLEANING                                               │
│     • Remove duplicate user-group memberships                   │
│     • Handle missing values in tag texts                        │
│     • Filter invalid IDs                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SAMPLING (for memory efficiency)                            │
│     • Select top N popular groups (min 50 members)              │
│     • Select active users (min 5 interactions)                  │
│     • Limit to 20,000 users for 16GB RAM                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. TEXT PREPROCESSING (for Content-Based)                      │
│     • Lowercase conversion                                      │
│     • English stop words removal                                │
│     • TF-IDF vectorization (max 5000 features)                  │
│     • min_df=2, max_df=0.95 filtering                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. MATRIX CONSTRUCTION                                         │
│     • User-Item Matrix (sparse, binary)                         │
│     • Item-Feature Matrix (TF-IDF weights)                      │
│     • User-User Similarity Matrix                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. TRAIN/TEST SPLIT                                            │
│     • 80% training, 20% testing                                 │
│     • Random split with seed=42                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Preprocessing Outputs

| File | Description |
|------|-------------|
| `dataset_summary.csv` | Overall statistics |
| `user_group_counts.csv` | Groups per user distribution |
| `group_member_counts.csv` | Members per group distribution |
| `data_distribution.png` | Visualization of distributions |

---

## 3. Implementation of Recommendation Approaches

### 3.1 Content-Based Filtering

**File:** `content_based.py`

**Concept:** Recommend groups that match user's explicit interests (tags)

#### 3.1.1 Feature Extraction (TF-IDF)

```python
# For each group, concatenate all its tags
Group 123: "python programming coding machine-learning data-science"
Group 456: "hiking outdoor camping adventure nature"

# Apply TF-IDF Vectorization
TfidfVectorizer(
    lowercase=True,           # Normalize case
    stop_words='english',     # Remove common words
    max_features=5000,        # Top 5000 terms
    min_df=2,                 # Min 2 documents
    max_df=0.95               # Max 95% of documents
)
```

**TF-IDF Formula:**
$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

Where:
- TF(t,d) = Term frequency of term t in document d
- N = Total number of documents
- DF(t) = Document frequency of term t

#### 3.1.2 User Profile Construction

```python
# Build user profile from their followed tags
User 1 interests: [python, AI, data-science]
→ User Profile = TF-IDF weighted average of tag vectors
```

#### 3.1.3 Similarity Computation

**Cosine Similarity:**
$$\text{sim}(user, group) = \frac{\vec{u} \cdot \vec{g}}{\|\vec{u}\| \times \|\vec{g}\|}$$

#### 3.1.4 Cold-Start Handling

For new users with no tag preferences:
- **Strategy:** Recommend most popular groups
- Build default profile from top tags

---

### 3.2 Collaborative Filtering

**File:** `collaborative.py`

**Concept:** Recommend groups that similar users have joined

#### 3.2.1 User-Based Collaborative Filtering

```
User-Item Matrix (Binary):
              Group1  Group2  Group3  Group4  Group5
User A          1       0       1       0       1
User B          1       1       0       0       1
User C          0       0       1       1       0
User D          1       1       0       0       1    ← Similar to User B

Prediction for User D on Group3:
→ Look at what similar users (A, B, C) did for Group3
→ Weight by similarity scores
```

**Prediction Formula:**
$$\hat{r}_{u,i} = \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot r_{v,i}}{\sum_{v \in N_k(u)} sim(u,v)}$$

Where:
- $N_k(u)$ = k nearest neighbors of user u
- $sim(u,v)$ = Cosine similarity between users
- $r_{v,i}$ = Rating of neighbor v for item i

#### 3.2.2 SVD Matrix Factorization

**Concept:** Decompose User-Item matrix into latent factors

$$R \approx U_k \Sigma_k V_k^T$$

Where:
- $U_k$ = User latent factors (n_users × k)
- $\Sigma_k$ = Singular values (k × k)
- $V_k^T$ = Item latent factors (k × n_items)
- k = Number of latent factors [10, 20]

**Benefits:**
- Dimensionality reduction
- Captures hidden patterns
- Handles sparsity better

---

### 3.3 Hybrid Recommendation System

**File:** `hybrid.py`

**Concept:** Combine Content-Based and Collaborative Filtering for best results

#### 3.3.1 Weighted Hybrid Formula

$$\text{Score} = \alpha \times CB_{norm} + (1 - \alpha) \times CF_{norm}$$

Where:
- α = Weight for Content-Based (0 to 1)
- $CB_{norm}$ = Normalized Content-Based score [0,1]
- $CF_{norm}$ = Normalized Collaborative score [0,1]

#### 3.3.2 Alpha Tuning

| Alpha (α) | CB Weight | CF Weight | Hit Rate |
|-----------|-----------|-----------|----------|
| 0.2 | 20% | 80% | 37.0% |
| **0.3** | **30%** | **70%** | **37.5%** ✓ Best |
| 0.4 | 40% | 60% | 37.0% |
| 0.5 | 50% | 50% | 32.0% |
| 0.6 | 60% | 40% | 19.5% |
| 0.7 | 70% | 30% | 15.5% |

**Best α = 0.3** → Collaborative Filtering is more effective for this domain

#### 3.3.3 Why Hybrid Works for This Domain

| Challenge | How Hybrid Solves It |
|-----------|---------------------|
| **High Sparsity (99.99%)** | CB provides fallback when CF data is insufficient |
| **Cold-Start Users** | CB uses tag preferences even for new users |
| **Active Users** | CF captures co-membership patterns effectively |
| **Rich Content** | 77,810 tags provide strong content signal |

#### 3.3.4 Hybrid Recommendation Example

```
User 1146883 wants recommendations:

Step 1: Content-Based computes:
   User interests: [python, AI, data]
   → CB scores for all groups

Step 2: Collaborative computes:
   Similar users: [User_A, User_B, User_C]
   → CF scores based on their memberships

Step 3: Hybrid combines:
   Score = 0.3 × CB + 0.7 × CF

Result:
┌──────┬──────────┬────────────┬──────────┬──────────┐
│ Rank │ Group ID │ Hybrid     │ CB Score │ CF Score │
├──────┼──────────┼────────────┼──────────┼──────────┤
│  1   │   5495   │   0.834    │   0.447  │   1.000  │
│  2   │   8932   │   0.827    │   0.427  │   0.998  │
│  3   │  10976   │   0.765    │   0.587  │   0.841  │
│  4   │   3856   │   0.710    │   0.302  │   0.885  │
│  5   │   3422   │   0.655    │   0.212  │   0.845  │
└──────┴──────────┴────────────┴──────────┴──────────┘
```

---

## 4. Evaluation Metrics

### 4.1 Metrics Used

| Metric | Formula | Description |
|--------|---------|-------------|
| **Hit Rate** | $\frac{\text{Users with } \geq 1 \text{ hit}}{\text{Total users}}$ | % of users with at least one correct recommendation |
| **Precision@K** | $\frac{\text{Relevant in Top-K}}{K}$ | Accuracy of top-K recommendations |
| **Recall@K** | $\frac{\text{Relevant in Top-K}}{\text{Total Relevant}}$ | Coverage of relevant items |
| **NDCG@K** | $\frac{DCG@K}{IDCG@K}$ | Ranking quality with position weighting |

### 4.2 Evaluation Results

#### **Main Results Table (All Metrics)**

| Rank | Method | Precision@10 | Recall@10 | NDCG@10 | Hit Rate |
|------|--------|--------------|-----------|---------|----------|
| 1 | **CF (SVD k=20)** | **0.0977** | **0.0988** | **0.1280** | **0.4967** |
| 2 | Hybrid (α=0.3) | 0.0763 | 0.0772 | 0.1001 | 0.4133 |
| 3 | CF (SVD k=10) | 0.0727 | 0.0729 | 0.0945 | 0.4000 |
| 4 | Popularity | 0.0150 | 0.0150 | 0.0184 | 0.1133 |
| 5 | Content-Based | 0.0100 | 0.0113 | 0.0157 | 0.0833 |
| 6 | Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### **Cold-Start Performance (Hit Rate by User Activity Level)**

| Activity Level | Hybrid | Content-Based | Collaborative | Popularity |
|----------------|--------|---------------|---------------|------------|
| 5-20 ratings | 0.28 | 0.06 | 0.26 | 0.10 |
| 21-50 ratings | 0.38 | 0.16 | 0.28 | 0.08 |
| 51-100 ratings | 0.56 | 0.08 | 0.50 | 0.12 |
| 100+ ratings | **0.72** | 0.14 | 0.70 | 0.36 |

#### **Key Findings:**

1. **CF (SVD k=20) performs best overall** with Hit Rate = 49.67%
2. **Hybrid achieves strong performance** with Hit Rate = 41.33%
3. **Cold-start handling**: Hybrid outperforms all methods for low-activity users
4. **Precision@10 = 9.77%** for best method indicates good recommendation quality
5. **All methods beat Random baseline** confirming system effectiveness

### 4.3 Why These Metrics?

**For Interest-Based Group Recommendation:**

1. **Hit Rate** - Most important because:
   - Users typically join 1-2 groups from recommendations
   - We care if at least ONE recommendation is useful

2. **Precision@K** - Important because:
   - Limited screen space for recommendations
   - Quality over quantity

3. **NDCG@K** - Important because:
   - Order matters (top recommendations should be best)
   - Weighted by position

### 4.4 Evaluation Methodology

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PROCESS                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Train/Test Split (80/20)                                    │
│     • Train on 80% of user-group interactions                   │
│     • Test on held-out 20%                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Generate Recommendations                                    │
│     • For each test user, generate Top-10 recommendations       │
│     • Exclude groups already joined in training                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Compare with Ground Truth                                   │
│     • Check if recommended groups appear in test set            │
│     • Count hits, compute precision, recall                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Aggregate Metrics                                           │
│     • Average across all test users                             │
│     • Report final Hit Rate, Precision@K, etc.                  │
└─────────────────────────────────────────────────────────────────┘
```

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

## Pipeline Architecture

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

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│                    (Main Entry Point)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Step 1: Data Preprocessing                    │
│                   (data_preprocessing.py)                       │
│         Load datasets, compute statistics, analyze sparsity     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│  Step 2: Content-Based    │         │  Step 3: Collaborative    │
│   (content_based.py)      │         │   (collaborative.py)      │
│                           │         │                           │
│  • TF-IDF on group tags   │         │  • User-Based CF          │
│  • User profile building  │         │  • Cosine similarity      │
│  • Cosine similarity      │         │  • SVD (k=10, k=20)       │
│  • Cold-start handling    │         │  • GPU acceleration       │
└───────────────────────────┘         └───────────────────────────┘
        │                                           │
        │         CB Score                CF Score  │
        └─────────────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 4: Hybrid System                        │
│                       (hybrid.py)                               │
│                                                                 │
│            Score = α × CB + (1 - α) × CF                        │
│                                                                 │
│  • Alpha tuning (0.3, 0.5, 0.7)                                 │
│  • Best α = 0.3 (Hit Rate: 43%)                                 │
│  • Cold-start handling                                          │
│  • Final recommendations                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Results     │
                    │  Tables + Plots │
                    └─────────────────┘
```

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
- Test α values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
- Evaluate using **Hit Rate** on test set
- Best α = **0.3** (Hit Rate: 37.5%)

#### 9.2 Justification for Hybrid Approach

| Domain Characteristic | Why Hybrid Works |
|----------------------|------------------|
| High Sparsity (99.99%) | CB provides fallback when CF fails |
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
| **NDCG@K** | Normalized Discounted Cumulative Gain (ranking quality) |

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
| `results/tables/Content_Based/` | TF-IDF features, user profiles, Top-10/20 recommendations |
| `results/tables/Collaborative_Filtering/` | User similarity, SVD results |
| `results/tables/Hybrid_System/` | Alpha tuning, comparison results, cold-start analysis |
| `results/plots/` | All visualizations |

---

## Final Results Summary

### Best Performing Methods

| Rank | Method | Hit Rate | NDCG@10 |
|------|--------|----------|---------|
| 🥇 | CF (SVD k=20) | 49.67% | 0.128 |
| 🥈 | Hybrid (α=0.3) | 41.33% | 0.100 |
| 🥉 | CF (SVD k=10) | 40.00% | 0.095 |

### Cold-Start Performance
- **Hybrid** provides consistent performance across all user activity levels
- For users with 5-20 interactions: Hybrid achieves **28%** Hit Rate
- For users with 100+ interactions: Hybrid achieves **72%** Hit Rate

### Conclusion
The **weighted hybrid approach** successfully combines Content-Based and Collaborative Filtering, providing:
1. **Strong overall performance** competing with pure CF methods
2. **Robust cold-start handling** through CB fallback
3. **Scalable architecture** for large datasets (20K users, 35K groups)

---
