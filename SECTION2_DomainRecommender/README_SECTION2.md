# Section 2: Interest-Based Group Recommendation System

**Project Name:** Interest-Based Group Recommendation System  
**Domain:** Social Group Formation (Meetup.com)

---

## 1️⃣ Project Overview

| Attribute | Value |
|-----------|-------|
| **Project Name** | Interest-Based Group Recommendation System |
| **Domain** | Social Group Formation (Meetup.com) |
| **Objective** | Recommend relevant interest groups to users |
| **Approaches** | Content-Based, Collaborative Filtering, Hybrid |

---

## 2️⃣ Problem Description

### The Problem We're Solving
Users on platforms like Meetup.com struggle to discover relevant interest groups among thousands of options. Manual browsing is time-consuming and often leads to missing groups that match their interests.

### Why It's Important
- **Information Overload:** 70,000+ groups make manual discovery impractical
- **User Engagement:** Better recommendations increase user satisfaction and platform engagement
- **Community Building:** Connecting users with the right groups strengthens communities

### Target Users
| User Type | Description |
|-----------|-------------|
| **New Users** | Need content-based recommendations based on their stated interests |
| **Active Users** | Benefit from collaborative filtering based on similar users' behavior |
| **All Users** | Get best results from hybrid approach combining both methods |

---

## 3️⃣ Dataset Description

### Data Source
**Meetup.com Interest Groups Dataset** - Public dataset containing user-group interactions and tag information.

### Dataset Statistics

| Metric | Full Dataset | Sampled Dataset |
|--------|--------------|-----------------|
| **Users** | 4,111,476 | 20,000 |
| **Groups (Items)** | 70,604 | 34,838 |
| **Interactions** | 10,627,861 | 731,426 |
| **Tags** | 77,810 | - |
| **Data Sparsity** | 99.9963% | 99.91% |

### Interaction Type
- **Type:** Implicit Feedback (Binary)
- **Signal:** Group Membership (1 = joined, 0 = not joined)
- **No explicit ratings** - users either join a group or don't

### Data Files

| File | Columns | Description |
|------|---------|-------------|
| `user_group.csv` | user_id, group_id | User-Group memberships |
| `user_tag.csv` | user_id, tag_id | User interest tags |
| `group_tag.csv` | group_id, tag_id | Group topic tags |
| `tag_text.csv` | tag_id, tag_text | Tag descriptions |

---

## 4️⃣ System Architecture

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
│  • Cosine Similarity            │  │  • k=10, k=20 latent factors    │
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
│                         (Best α = 0.3)                                      │
│                                                                             │
│  • Weighted combination of both approaches                                  │
│  • Alpha tuning: tested 0.2, 0.3, 0.4, 0.5, 0.6, 0.7                        │
│  • Cold-start handling for new users                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   RECOMMENDED GROUPS  │
                         │   (Top-10 for user)   │
                         └───────────────────────┘
```

---

## 5️⃣ Content-Based Approach

### Feature Extraction: TF-IDF

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency)

```python
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

### User Profile Construction

Each user's profile is built from their tag preferences:

```
User interests: [python, AI, data-science, machine-learning]
     ↓
User Profile = Weighted average of tag TF-IDF vectors
```

### Similarity Metric: Cosine Similarity

$$\text{sim}(user, group) = \frac{\vec{u} \cdot \vec{g}}{\|\vec{u}\| \times \|\vec{g}\|}$$

### Example Recommendation

```
User 219356 interests: singles, dating, social events

Top-5 Content-Based Recommendations:
┌──────┬──────────┬─────────┬─────────────────────────────────┐
│ Rank │ Group ID │ Score   │ Tags                            │
├──────┼──────────┼─────────┼─────────────────────────────────┤
│  1   │  58271   │ 0.4094  │ dance, singles, diningout       │
│  2   │  39095   │ 0.3923  │ singles, newlysingle, parents   │
│  3   │  52010   │ 0.3759  │ singles, speed-dating           │
│  4   │  62415   │ 0.3737  │ dance, singles, salsa           │
│  5   │  21547   │ 0.3721  │ singles, speed-dating           │
└──────┴──────────┴─────────┴─────────────────────────────────┘
```

---

## 6️⃣ Collaborative Filtering

### Method: User-Based CF + SVD Matrix Factorization

| Component | Description |
|-----------|-------------|
| **User-Based CF** | Find similar users using cosine similarity |
| **SVD** | Matrix factorization with k=10 and k=20 latent factors |
| **Prediction** | Weighted average of similar users' ratings |

### SVD Formula

$$R \approx U_k \Sigma_k V_k^T$$

Where:
- $U_k$ = User latent factors (n_users × k)
- $\Sigma_k$ = Singular values (k × k)
- $V_k^T$ = Item latent factors (k × n_items)

### Similarity: Cosine Similarity

$$\text{sim}(u, v) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \times \|\vec{v}\|}$$

### Cold-Start Handling

| User Type | Strategy |
|-----------|----------|
| **New users (no history)** | Use Content-Based recommendations |
| **Users with few interactions** | Hybrid with higher CB weight |
| **Active users** | Full Collaborative Filtering |

---

## 7️⃣ Hybrid Recommendation

### Combination Method: Weighted Hybrid

$$\text{Score} = \alpha \times CB_{norm} + (1 - \alpha) \times CF_{norm}$$

Where:
- **α** = Weight for Content-Based (tuned automatically)
- **CB_norm** = Normalized Content-Based score [0, 1]
- **CF_norm** = Normalized Collaborative Filtering score [0, 1]

### Alpha Tuning Results

| Alpha (α) | CB Weight | CF Weight | Hit Rate |
|-----------|-----------|-----------|----------|
| 0.2 | 20% | 80% | 37.0% |
| **0.3** | **30%** | **70%** | **37.5%** ✓ Best |
| 0.4 | 40% | 60% | 37.0% |
| 0.5 | 50% | 50% | 32.0% |
| 0.6 | 60% | 40% | 19.5% |
| 0.7 | 70% | 30% | 15.5% |

### Why This Hybrid Strategy?

| Domain Characteristic | How Hybrid Solves It |
|----------------------|----------------------|
| **High Sparsity (99.99%)** | CB provides fallback when CF data is insufficient |
| **Cold-Start Users** | CB uses tag preferences even for new users |
| **Active Users** | CF captures co-membership patterns effectively |
| **Rich Content (77K tags)** | Strong content signal for CB |

---

## 8️⃣ Evaluation

### Metrics Used

| Metric | Formula | Description |
|--------|---------|-------------|
| **Hit Rate** | Users with ≥1 hit / Total users | % of users with at least one correct recommendation |
| **Precision@K** | Relevant in Top-K / K | Accuracy of top-K recommendations |
| **Recall@K** | Relevant in Top-K / Total Relevant | Coverage of relevant items |
| **NDCG@K** | DCG@K / IDCG@K | Ranking quality with position weighting |

### Main Results Comparison

| Rank | Method | Precision@10 | Recall@10 | NDCG@10 | Hit Rate |
|------|--------|--------------|-----------|---------|----------|
| 🥇 | **CF (SVD k=20)** | **0.0977** | **0.0988** | **0.1280** | **49.67%** |
| 🥈 | Hybrid (α=0.3) | 0.0763 | 0.0772 | 0.1001 | 41.33% |
| 🥉 | CF (SVD k=10) | 0.0727 | 0.0729 | 0.0945 | 40.00% |
| 4 | Popularity | 0.0150 | 0.0150 | 0.0184 | 11.33% |
| 5 | Content-Based | 0.0100 | 0.0113 | 0.0157 | 8.33% |
| 6 | Random | 0.0000 | 0.0000 | 0.0000 | 0.00% |

### Cold-Start Performance (Hit Rate by Activity Level)

| Activity Level | Hybrid | Content-Based | Collaborative | Popularity |
|----------------|--------|---------------|---------------|------------|
| 5-20 ratings | **28%** | 6% | 26% | 10% |
| 21-50 ratings | **38%** | 16% | 28% | 8% |
| 51-100 ratings | **56%** | 8% | 50% | 12% |
| 100+ ratings | **72%** | 14% | 70% | 36% |

### Key Findings

1. **CF (SVD k=20) achieves best overall performance** with 49.67% Hit Rate
2. **Hybrid provides consistent performance** across all user activity levels
3. **Content-Based handles cold-start** better than pure CF for new users
4. **All methods significantly beat Random baseline** confirming system effectiveness

---

## 9️⃣ How to Run the Code

### Prerequisites

```bash
cd SECTION2_DomainRecommender
pip install -r requirements.txt
```

### Required Packages

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
torch  # Optional: for GPU acceleration
```

### Run Complete Pipeline

```bash
cd SECTION2_DomainRecommender/code
python main.py
```

### Run Individual Components

```bash
python data_preprocessing.py   # Step 1: Data analysis
python content_based.py        # Step 2: Content-Based
python collaborative.py        # Step 3: Collaborative Filtering
python hybrid.py               # Step 4: Hybrid System
```

### Configuration

In `main.py`, adjust based on your RAM:

```python
SAMPLE_SIZE = 20000  # Number of users (adjust based on RAM)
```

| Sample Size | RAM Required |
|-------------|--------------|
| 10,000 users | ~4 GB |
| 20,000 users | ~8 GB |
| 30,000 users | ~16 GB |

---

## 🔟 Results

### Output Tables

| Directory | Contents |
|-----------|----------|
| `results/tables/Data_Preprocessing/` | Dataset statistics, distributions |
| `results/tables/Content_Based/` | TF-IDF features, user profiles, Top-10/20 recommendations |
| `results/tables/Collaborative_Filtering/` | User similarity, SVD results |
| `results/tables/Hybrid_System/` | Alpha tuning, comparison results, cold-start analysis |

### Output Plots

| Plot | Location | Description |
|------|----------|-------------|
| `data_distribution.png` | Data_Preprocessing/ | User/Group activity distributions |
| `tfidf_distribution.png` | Content_Based/ | TF-IDF feature statistics |
| `user_profile_distribution.png` | Content_Based/ | User profile statistics |
| `svd_variance_explained.png` | Collaborative_Filtering/ | SVD component analysis |
| `user_similarity_distribution.png` | Collaborative_Filtering/ | User-user similarity |
| `alpha_tuning.png` | Hybrid_System/ | Alpha parameter optimization |
| `baseline_comparison.png` | Hybrid_System/ | Method comparison |
| `cold_start_comparison.png` | Hybrid_System/ | Cold-start analysis |

### Sample Recommendations

```
User 5 - Hybrid Recommendations (α=0.3):

┌──────┬──────────┬────────────┬──────────┬──────────┐
│ Rank │ Group ID │ Hybrid     │ CB Score │ CF Score │
├──────┼──────────┼────────────┼──────────┼──────────┤
│  1   │  25471   │   0.4030   │   0.000  │   0.576  │
│  2   │  51404   │   0.3186   │   0.000  │   0.455  │
│  3   │  28042   │   0.3116   │   0.000  │   0.445  │
│  4   │  18726   │   0.3010   │   1.000  │   0.001  │
│  5   │  20735   │   0.2890   │   0.960  │   0.002  │
└──────┴──────────┴────────────┴──────────┴──────────┘
```

---

## 1️⃣1️⃣ Contributors

| Name | Role |
|------|------|
| **[Member 1]** | Data Preprocessing, Content-Based Filtering |
| **[Member 2]** | Collaborative Filtering, SVD Implementation |
| **[Member 3]** | Hybrid System, Evaluation Metrics |
| **[Member 4]** | Documentation, Testing, Visualization |

---

## 1️⃣2️⃣ AI Assistance Disclosure

> **Disclosure:** This project used AI tools (ChatGPT/GitHub Copilot) for:
> - Learning and understanding recommendation system concepts
> - Code explanation and debugging assistance
> - Documentation formatting and organization
>
> All code was reviewed, understood, and validated by team members.

---

## Project Structure

```
SECTION2_DomainRecommender/
├── README_SECTION2.md          # This file
├── requirements.txt            # Python dependencies
├── code/
│   ├── main.py                 # Main entry point
│   ├── data_preprocessing.py   # Data loading and statistics
│   ├── content_based.py        # Content-Based Filtering
│   ├── collaborative.py        # Collaborative Filtering
│   └── hybrid.py               # Hybrid Recommender
├── data/
│   ├── user_group.csv          # User-Group memberships
│   ├── user_tag.csv            # User tag preferences
│   ├── group_tag.csv           # Group-Tag associations
│   └── tag_text.csv            # Tag descriptions
└── results/
    ├── tables/                 # CSV output files
    └── plots/                  # Visualization images
```

---

## Key Formulas

### TF-IDF
$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

### Cosine Similarity
$$\text{sim}(A,B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### SVD Decomposition
$$R = U \Sigma V^T$$

### Weighted Hybrid Score
$$\text{Score} = \alpha \times \text{CB}_{norm} + (1-\alpha) \times \text{CF}_{norm}$$

---
