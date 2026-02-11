# Clustering of Categorical Time Series

## Overview
Python implementation of unsupervised clustering algorithms for categorical time series data. 

## Project Description
This project implements and compares multiple clustering approaches for categorical temporal sequences:

### Algorithms
- **K-Medoids Clustering**: Partition-based clustering using medoid representatives
- **Hierarchical Clustering**: Agglomerative clustering with Ward's linkage method

### Distance Metrics
- **Dynamic Hamming Distance (DHD)**: Position-wise difference count
- **Optimal Matching (OM)**: Minimum-cost sequence transformation using dynamic programming

### Evaluation Metrics
- **Adjusted Rand Index (ARI)**: Agreement between predicted and true labels
- **Silhouette Score**: Intrinsic clustering quality assessment

## Data

- **Source**: Generated using Markov chains with different transition matrix distributions (random)
- **Format**: 150 sequences × 10 time steps
- **States**: 6 categorical states
  - `sain_non_vaccine` (healthy unvaccinated)
  - `retabli` (recovered)
  - `contamine` (contaminated)
  - `mort` (deceased)
  - `sain_vaccine` (healthy vaccinated)
  - `infecte` (infected)

### Data Generation
Three transition matrices generated with different distributions:
- M1: Uniform distribution
- M2: Absolute Gaussian distribution
- M3: Beta distribution



### Best Performing Method
**Hierarchical Clustering + Optimal Matching**
- ARI: 0.1241
- Silhouette Score: 0.0672

### Ranking
1. Hierarchical (OM): ARI=0.124, Silhouette=0.067
2. K-Medoids (OM): ARI=0.094, Silhouette=0.054
3. Hierarchical (DHD): ARI=0.005, Silhouette=0.018
4. K-Medoids (DHD): ARI=0.017, Silhouette=0.040

**Insight**: Optimal Matching distance captures sequence similarity better than Hamming distance for this data.








