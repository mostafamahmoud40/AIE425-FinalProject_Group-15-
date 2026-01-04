# Dimensionality Reduction and Matrix Factorization for Recommender Systems

This project implements and compares dimensionality reduction and matrix factorization techniques for collaborative filtering using the MovieLens 20M dataset. The objective is to predict missing user–item ratings and analyze latent representations produced by PCA-based and SVD-based methods.

## Project Structure

data/  
 ├─ ratings.csv  
 ├─ movies.csv  

results/  
 ├─ tables/  
 │   ├─ Statistical Analysis/  
 │   ├─ PCA Method with Mean-Filling/  
 │   ├─ PCA Method with MLE/  
 │   └─ svd/  
 └─ plots/  
     ├─ Statistical Analysis/  
     ├─ PCA Method with Mean-Filling/  
     ├─ PCA Method with MLE/  
     └─ svd/  

code/  
 ├─ statistical_analysis.py  
 ├─ pca_mean_filling.py  
 ├─ pca_mle.py  
 └─ svd_collaborative_filtering.py  

## Implemented Methods

### Statistical Analysis
- User and item rating distributions  
- Average ratings per user and per item  
- Item popularity analysis and grouping  
- Rating percentile-based grouping  
- Target user and target item selection  
- Co-rating analysis and threshold definition  

### PCA with Mean-Filling
- Mean-filling of missing ratings for target items  
- Item mean-centering  
- Item–item covariance computation  
- Peer selection (Top-5 and Top-10)  
- PCA via covariance eigen-decomposition  
- Reduced latent user space construction  
- Rating prediction and peer-size comparison  

### PCA with Maximum Likelihood Estimation (MLE)
- Covariance estimation using only observed entries  
- PCA using eigen-decomposition  
- Latent space projection  
- Rating prediction  
- Comparison with mean-filling PCA results  

### Singular Value Decomposition (SVD)
- Dense rating matrix construction using item-mean filling  
- Full and truncated SVD  
- Rank selection using reconstruction error  
- Rating prediction  
- Cold-start user analysis  
- Sensitivity to missing data  
- Latent factor interpretation  

## Dataset

MovieLens 20M Dataset  
- Explicit ratings on a 1–5 scale  
- Approximately 20 million ratings  
- More than 100,000 users  
- More than 27,000 items  

## How to Run

1. Place ratings.csv and movies.csv inside the data/ directory  
2. Install dependencies:  
   pip install -r requirements.txt  
3. Run scripts sequentially:  
   python code/statistical_analysis.py  
   python code/pca_mean_filling.py  
   python code/pca_mle.py  
   python code/svd_collaborative_filtering.py  

All generated tables and plots are automatically saved to the results/ directory.

## Reproducibility

All experiments use fixed random seeds. Intermediate tables and figures are saved to ensure full reproducibility of results.

## Purpose

This project provides a systematic comparison of PCA-based and SVD-based collaborative filtering methods, highlighting their practical behavior, strengths, and limitations when applied to large-scale recommender system data.

## License

This project is intended for academic and educational use only.
