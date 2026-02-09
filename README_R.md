# INSERM Clustering Project - R Implementation

## Overview
Original R implementation of clustering algorithms for categorical time series data from INSERM internship project.

## Project Structure
```
R/
├── data_generation.R          # Generate sequences using Markov chains
├── distance_metrics.R         # Calculate Hamming and Optimal Matching distances
├── clustering.R               # K-Medoids and Hierarchical clustering
├── evaluation.R               # ARI and Silhouette evaluation metrics

```

## Dependencies
```
install.packages("cluster")      # For PAM clustering
install.packages("fossil")       # For Adjusted Rand Index
```

## Usage

### Run all steps in sequence
```
source("data_generation.R")
source("distance_metrics.R")
source("clustering.R")
source("evaluation.R")
```

### Or run individual steps

**Step 1: Generate Data**
```
source("data_generation.R")
```
Generates 150 sequences using Markov chains.

**Step 2: Calculate Distance Matrices**
```
source("distance_metrics.R")
```
Computes Hamming and Optimal Matching distances.

**Step 3: Perform Clustering**
```
source("clustering.R")
```
Applies K-Medoids and Hierarchical clustering with visualizations.

**Step 4: Evaluate Results**
```
source("evaluation.R")
```
Computes ARI and Silhouette scores.

## Key Results

**Best Method**: Hierarchical Clustering + Optimal Matching
- ARI: 0.1241
- Silhouette Score: 0.0672

## Data

- **150 sequences** generated using Markov chains
- **10 time steps** per sequence
- **6 categorical states** representing disease progression

## Algorithms

### Clustering Methods
- **K-Medoids (PAM)**: Partition-based clustering using medoid representatives
- **Hierarchical (Ward)**: Agglomerative clustering minimizing within-cluster variance

### Distance Metrics
- **Hamming Distance**: Position-wise differences
- **Optimal Matching**: Minimum-cost sequence transformation

### Evaluation
- **Adjusted Rand Index**: Agreement with ground truth
- **Silhouette Score**: Intrinsic clustering quality

## Outputs

### Data Files
- `sequences_m1.csv`: Generated sequences
- `distance_matrix_hamming.rds`: Hamming distance matrix
- `distance_matrix_om.rds`: Optimal Matching distance matrix
- `labels_*.rds`: Cluster assignments for each method
- `evaluation_results.csv`: Comparison table

### Visualizations
- `dendrogram_hamming.png`: Hierarchical clustering dendrogram (Hamming)
- `dendrogram_om.png`: Hierarchical clustering dendrogram (OM)
- `clusters_hc_hamming.png`: 2D cluster visualization (Hamming)
- `clusters_hc_om.png`: 2D cluster visualization (OM)
- `silhouette_hc_hamming.png`: Silhouette plot (Hamming)
- `silhouette_hc_om.png`: Silhouette plot (OM)

## Notes

- This is the **original R implementation** using TraMineR and cluster packages
- A **Python version** is also available in the parent directory for reproducibility and modern tooling
- Both implementations produce equivalent results

## Author

Wijdane GARAB  
INSERM Internship Project  
Paris Cité University

## References

- Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
