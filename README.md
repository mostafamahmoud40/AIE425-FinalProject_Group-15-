# AIE425 Final Project - Group 15
# Recommender Systems: Dimensionality Reduction & Domain-Based Recommendation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-red.svg)

</div>

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Section 1: Dimensionality Reduction](#-section-1-dimensionality-reduction-for-collaborative-filtering)
4. [Section 2: Domain-Based Recommender System](#-section-2-interest-based-group-recommendation-system)
5. [Installation & Setup](#-installation--setup)
6. [How to Run](#-how-to-run)
7. [Results Summary](#-results-summary)
8. [Key Formulas](#-key-formulas)
9. [Contributors](#-contributors)
10. [License](#-license)

---

## Project Overview

This project implements and evaluates **recommender system techniques** through two comprehensive sections:

| Section | Domain | Dataset | Objective |
|---------|--------|---------|-----------|
| **Section 1** | Movie Recommendations | MovieLens 20M | Predict missing ratings using PCA & SVD |
| **Section 2** | Group Recommendations | Meetup.com | Recommend interest groups to users |

### Project Goals

1. **Understand** dimensionality reduction techniques (PCA, SVD) for collaborative filtering
2. **Implement** content-based, collaborative filtering, and hybrid recommendation approaches
3. **Evaluate** and compare different recommendation strategies
4. **Handle** real-world challenges like sparsity and cold-start problems

---

## Project Structure

```
AIE425 FinalProject_Group[15]/
│
├── README.md                           # This file
├── requirements.txt                    # Global dependencies
│
├── SECTION1_DimensionalityReduction/   # Movie recommendations
│   ├── README_SECTION1.md              # Section 1 documentation
│   ├── code/
│   │   ├── utils.py                    # Statistical analysis utilities
│   │   ├── pca_mean_filling.py         # PCA with mean-filling approach
│   │   ├── PCA_Method_with_MLE.py      # PCA with MLE estimation
│   │   └── svd.py                      # SVD collaborative filtering
│   ├── data/
│   │   ├── ratings.csv                 # User-Movie ratings
│   │   └── movies.csv                  # Movie metadata
│   └── results/
│       ├── tables/                     # CSV output files
│       └── plots/                      # Visualization images
│
└── SECTION2_DomainRecommender/         # Group recommendations
    ├── README_SECTION2.md              # Section 2 documentation
    ├── requirements.txt                # Section-specific dependencies
    ├── code/
    │   ├── main.py                     # Main entry point
    │   ├── data_preprocessing.py       # Data loading and analysis
    │   ├── content_based.py            # Content-based filtering (TF-IDF)
    │   ├── collaborative.py            # Collaborative filtering (SVD)
    │   └── hybrid.py                   # Hybrid recommendation system
    ├── data/
    │   ├── user_group.csv              # User-Group memberships
    │   ├── user_tag.csv                # User tag preferences
    │   ├── group_tag.csv               # Group-Tag associations
    │   └── tag_text.csv                # Tag descriptions
    └── results/
        ├── tables/                     # CSV output files
        └── plots/                      # Visualization images
```

---

## Section 1: Dimensionality Reduction for Collaborative Filtering

### Dataset: MovieLens 20M

| Metric | Value |
|--------|-------|
| **Users** | ~138,000 |
| **Movies** | ~27,000 |
| **Ratings** | ~20 million |
| **Rating Scale** | 1-5 (explicit) |
| **Sparsity** | ~99.5% |

### Implemented Methods

#### 1. Statistical Analysis
- User and item rating distributions
- Average ratings per user and per item
- Item popularity analysis (Low/Medium/High)
- Target user and item selection
- Co-rating analysis with threshold definition

#### 2. PCA with Mean-Filling
Handles missing values by filling with item mean ratings before applying PCA.

**Process:**
1. Calculate average rating for target items
2. Fill missing ratings with item mean
3. Compute item-item covariance matrix
4. Select peer items (Top-5 and Top-10)
5. Apply PCA via eigen-decomposition
6. Project users to reduced latent space
7. Predict ratings and compare peer sizes

#### 3. PCA with Maximum Likelihood Estimation (MLE)
Estimates covariance using only observed entries (no imputation).

**Process:**
1. Compute MLE covariance for target items
2. Select peer items based on covariance
3. Apply PCA using true eigen-decomposition
4. Project users and predict ratings
5. Compare with mean-filling approach

#### 4. SVD-Based Collaborative Filtering
Matrix factorization approach for rating prediction.

**Process:**
1. Construct dense rating matrix (item-mean filling)
2. Apply truncated SVD: $R \approx U_k \Sigma_k V_k^T$
3. Select optimal rank using reconstruction error
4. Predict ratings for target users
5. Analyze cold-start sensitivity
6. Interpret latent factors

### Section 1 Key Findings

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| **PCA Mean-Filling** | Simple, handles all users | Introduces bias from imputation |
| **PCA MLE** | Uses only real data | Requires co-rated items |
| **SVD** | Captures latent factors | Computationally expensive |

---

## Section 2: Interest-Based Group Recommendation System

### Dataset: Meetup.com

| Metric | Full Dataset | Sampled |
|--------|--------------|---------|
| **Users** | 4,111,476 | 20,000 |
| **Groups** | 70,604 | 34,838 |
| **Interactions** | 10,627,861 | 731,426 |
| **Tags** | 77,810 | - |
| **Sparsity** | 99.9963% | 99.91% |

### Interaction Type
- **Type:** Implicit Feedback (Binary)
- **Signal:** Group Membership (1 = joined, 0 = not joined)

### System Architecture

```
                         ┌──────────────┐
                         │     USER     │
                         └──────┬───────┘
                                │
           ┌────────────────────┴────────────────────┐
           ▼                                         ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│  CONTENT-BASED (CB)     │             │  COLLABORATIVE (CF)     │
│                         │             │                         │
│  • TF-IDF Vectorization │             │  • User-User Similarity │
│  • User Profile (Tags)  │             │  • SVD (k=10, k=20)     │
│  • Cosine Similarity    │             │  • Matrix Factorization │
└─────────────────────────┘             └─────────────────────────┘
           │                                         │
           └────────────────────┬────────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │     HYBRID SYSTEM        │
                  │                          │
                  │  Score = α×CB + (1-α)×CF │
                  │  Best α = 0.3            │
                  └──────────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │   TOP-10 RECOMMENDATIONS │
                  └──────────────────────────┘
```

### Implemented Approaches

#### 1. Content-Based Filtering (TF-IDF)

**Feature Extraction:**
```python
TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    min_df=2,
    max_df=0.95
)
```

**User Profile Construction:**
- Aggregate user's tag preferences
- Create weighted TF-IDF vector
- Compute cosine similarity with groups

#### 2. Collaborative Filtering (SVD)

**Matrix Factorization:**
$$R \approx U_k \Sigma_k V_k^T$$

- $U_k$: User latent factors (n_users × k)
- $\Sigma_k$: Singular values (k × k)
- $V_k^T$: Item latent factors (k × n_items)

**Configurations Tested:**
- k = 10 latent factors
- k = 20 latent factors

#### 3. Hybrid Recommender

**Weighted Combination:**
$$\text{Score} = \alpha \times CB_{norm} + (1 - \alpha) \times CF_{norm}$$

**Alpha Tuning Results:**

| α | CB Weight | CF Weight | Hit Rate |
|---|-----------|-----------|----------|
| 0.2 | 20% | 80% | 37.0% |
| **0.3** | **30%** | **70%** | **37.5%** ✓ |
| 0.4 | 40% | 60% | 37.0% |
| 0.5 | 50% | 50% | 32.0% |

### Section 2 Evaluation Results

| Rank | Method | Precision@10 | Recall@10 | NDCG@10 | Hit Rate |
|------|--------|--------------|-----------|---------|----------|
| 1 | **CF (SVD k=20)** | **0.0977** | **0.0988** | **0.1280** | **49.67%** |
| 2 | Hybrid (α=0.3) | 0.0763 | 0.0772 | 0.1001 | 41.33% |
| 3 | CF (SVD k=10) | 0.0727 | 0.0729 | 0.0945 | 40.00% |
| 4 | Popularity | 0.0150 | 0.0150 | 0.0184 | 11.33% |
| 5 | Content-Based | 0.0100 | 0.0113 | 0.0157 | 8.33% |
| 6 | Random | 0.0000 | 0.0000 | 0.0000 | 0.00% |

### Cold-Start Performance

| Activity Level | Hybrid | CB | CF | Popularity |
|----------------|--------|----|----|------------|
| 5-20 ratings | **28%** | 6% | 26% | 10% |
| 21-50 ratings | **38%** | 16% | 28% | 8% |
| 51-100 ratings | **56%** | 8% | 50% | 12% |
| 100+ ratings | **72%** | 14% | 70% | 36% |

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation Steps

```bash
# Clone or download the project
cd "AIE425 FinalProject_Group[15]"

# Install global dependencies
pip install -r requirements.txt

# For Section 2 (optional GPU support)
cd SECTION2_DomainRecommender
pip install -r requirements.txt
```

### Required Packages

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
torch  # Optional: for GPU acceleration in Section 2
```

### Memory Requirements

| Configuration | RAM Required |
|---------------|--------------|
| Section 1 (8K users) | ~4 GB |
| Section 2 (10K users) | ~4 GB |
| Section 2 (20K users) | ~8 GB |
| Section 2 (30K users) | ~16 GB |

---

## How to Run

### Section 1: Dimensionality Reduction

```bash
cd SECTION1_DimensionalityReduction/code

# Step 1: Statistical Analysis
python utils.py

# Step 2: PCA with Mean-Filling
python pca_mean_filling.py

# Step 3: PCA with MLE
python PCA_Method_with_MLE.py

# Step 4: SVD Collaborative Filtering
python svd.py
```

### Section 2: Domain Recommender

```bash
cd SECTION2_DomainRecommender/code

# Run complete pipeline
python main.py

# Or run individual components:
python data_preprocessing.py   # Data analysis
python content_based.py        # Content-Based
python collaborative.py        # Collaborative Filtering
python hybrid.py               # Hybrid System
```

### Configuration (Section 2)

In `main.py`, adjust based on your RAM:
```python
SAMPLE_SIZE = 20000  # Number of users
```

---

## Results Summary

### Section 1 Outputs

| Directory | Contents |
|-----------|----------|
| `results/tables/Statistical Analysis/` | User/item statistics, popularity groups |
| `results/tables/PCA Method with Mean-Filling/` | Predictions, peer selections, covariance |
| `results/tables/PCA Method with MLE/` | MLE estimates, comparisons |
| `results/tables/svd/` | SVD decomposition, eigenpairs |

### Section 2 Outputs

| Directory | Contents |
|-----------|----------|
| `results/tables/Data_Preprocessing/` | Dataset statistics |
| `results/tables/Content_Based/` | TF-IDF features, user profiles |
| `results/tables/Collaborative_Filtering/` | User similarity, SVD results |
| `results/tables/Hybrid_System/` | Alpha tuning, comparisons |

---

## Key Formulas

### TF-IDF (Term Frequency - Inverse Document Frequency)
$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

### Cosine Similarity
$$\text{sim}(A,B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### SVD Decomposition
$$R = U \Sigma V^T$$

### PCA via Eigen-Decomposition
$$C = \frac{1}{n-1} X^T X$$
$$C \cdot v = \lambda \cdot v$$

### Weighted Hybrid Score
$$\text{Score} = \alpha \times \text{CB}_{norm} + (1-\alpha) \times \text{CF}_{norm}$$

### Evaluation Metrics

**Precision@K:**
$$\text{Precision@K} = \frac{\text{Relevant items in Top-K}}{K}$$

**Recall@K:**
$$\text{Recall@K} = \frac{\text{Relevant items in Top-K}}{\text{Total Relevant items}}$$

**NDCG@K:**
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

---

## Contributors

| Name | Contributions |
|------|--------------|
| **Group 15 Members** | Full implementation and documentation |

---

## AI Assistance Disclosure

> **Disclosure:** This project used AI tools (ChatGPT/GitHub Copilot) for:
> - Learning and understanding recommendation system concepts
> - Code explanation and debugging assistance
> - Documentation formatting and organization
>
> All code was reviewed, understood, and validated by team members.

---

## License

This project is intended for **academic and educational use only**.

**Course:** AIE425 - Recommender Systems  
**Institution:** [Your University]  
**Semester:** [Current Semester]

---

<div align="center">

### Key Takeaways

| Finding | Details |
|---------|---------|
| **Best Overall Method** | CF with SVD (k=20) achieves 49.67% Hit Rate |
| **Cold-Start Solution** | Hybrid approach performs best for new users |
| **Sparsity Handling** | SVD effectively captures latent patterns |
| **Computational Trade-off** | More latent factors = better accuracy but slower |

</div>

---

<div align="center">
<i>AIE425 Final Project - Group 15</i>
</div>
