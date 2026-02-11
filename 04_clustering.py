import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
import matplotlib.pyplot as plt



def kmedoids_clustering_simple(dist_matrix, n_clusters=3, max_iterations=100, random_state=42):

    np.random.seed(random_state)
    n_samples = dist_matrix.shape[0]
    
 
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    
    for iteration in range(max_iterations):
       
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances_to_medoids = dist_matrix[i, medoid_indices]
            labels[i] = np.argmin(distances_to_medoids)
        
        
        total_cost = 0
        for i in range(n_samples):
            medoid = medoid_indices[labels[i]]
            total_cost += dist_matrix[i, medoid]
        
        old_cost = total_cost
        
        
        improved = False
        for m_idx, medoid in enumerate(medoid_indices):
            for i in range(n_samples):
                if i in medoid_indices:
                    continue
                
               
                new_medoids = medoid_indices.copy()
                new_medoids[m_idx] = i
                
               
                new_cost = 0
                for j in range(n_samples):
                    distances = dist_matrix[j, new_medoids]
                    new_cost += np.min(distances)
                
               
                if new_cost < old_cost:
                    medoid_indices = new_medoids
                    old_cost = new_cost
                    improved = True
                    break
            
            if improved:
                break
        
       
        if not improved:
            break
    
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        distances_to_medoids = dist_matrix[i, medoid_indices]
        labels[i] = np.argmin(distances_to_medoids)
    
    return labels, medoid_indices


def hierarchical_clustering(dist_matrix, n_clusters=3, linkage_method='ward'):
   
    
    condensed_dist = squareform(dist_matrix)
    
    
    linkage_matrix = linkage(condensed_dist, method=linkage_method)
    
   
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1  # -1 pour 0-indexing
    
    return labels, linkage_matrix


def plot_dendrogram(linkage_matrix, title="Dendrogram", filename=None):

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"✓ Dendrogram sauvegardé: {filename}")
    else:
        plt.show()





def plot_clusters_2d(dist_matrix, labels, title="Clusters", filename=None):

   
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(dist_matrix)
   
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    for cluster in np.unique(labels):
        mask = labels == cluster
        plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                   c=colors[cluster % len(colors)], 
                   label=f'Cluster {cluster}', 
                   s=100, alpha=0.6)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("MDS Dimension 1", fontsize=12)
    plt.ylabel("MDS Dimension 2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"✓ Plot sauvegardé: {filename}")
    else:
        plt.show()




if __name__ == "__main__":
    import pandas as pd
    
   
    sequences = pd.read_csv("sequences_m1.csv").values
    dist_hamming = np.load("distance_matrix_hamming.npy")
    dist_om = np.load("distance_matrix_om.npy")
    
    print("=" * 60)
    print("CLUSTERING")
    print("=" * 60)
    
   
    print("\n1. K-MEDOIDS + HAMMING DISTANCE")
    print("-" * 60)
    labels_kmed_hamming, medoids_hamming = kmedoids_clustering_simple(dist_hamming, n_clusters=3)
    print(f"Labels: {np.unique(labels_kmed_hamming)}")
    print(f"Cluster sizes: {np.bincount(labels_kmed_hamming)}")
    print(f"Medoids: {medoids_hamming}")
    
  
    print("\n2. K-MEDOIDS + OPTIMAL MATCHING")
    print("-" * 60)
    labels_kmed_om, medoids_om = kmedoids_clustering_simple(dist_om, n_clusters=3)
    print(f"Labels: {np.unique(labels_kmed_om)}")
    print(f"Cluster sizes: {np.bincount(labels_kmed_om)}")
    print(f"Medoids: {medoids_om}")
    
   
    print("\n3. HIERARCHICAL + HAMMING DISTANCE")
    print("-" * 60)
    labels_hc_hamming, linkage_hc_hamming = hierarchical_clustering(dist_hamming, n_clusters=3, linkage_method='ward')
    print(f"Labels: {np.unique(labels_hc_hamming)}")
    print(f"Cluster sizes: {np.bincount(labels_hc_hamming)}")
    
   
    print("\n4. HIERARCHICAL + OPTIMAL MATCHING")
    print("-" * 60)
    labels_hc_om, linkage_hc_om = hierarchical_clustering(dist_om, n_clusters=3, linkage_method='ward')
    print(f"Labels: {np.unique(labels_hc_om)}")
    print(f"Cluster sizes: {np.bincount(labels_hc_om)}")
    
   
    print("\n5. VISUALIZATIONS")
    print("-" * 60)
    plot_dendrogram(linkage_hc_hamming, 
                   title="Hierarchical Clustering Dendrogram (Hamming)",
                   filename="dendrogram_hamming.png")
    
    plot_dendrogram(linkage_hc_om,
                   title="Hierarchical Clustering Dendrogram (OM)",
                   filename="dendrogram_om.png")
    
    plot_clusters_2d(dist_hamming, labels_hc_hamming,
                    title="HC Clusters (Hamming)",
                    filename="clusters_hc_hamming.png")
    
    plot_clusters_2d(dist_om, labels_hc_om,
                    title="HC Clusters (OM)",
                    filename="clusters_hc_om.png")
    
 
    np.save("labels_kmedoids_hamming.npy", labels_kmed_hamming)
    np.save("labels_kmedoids_om.npy", labels_kmed_om)
    np.save("labels_hierarchical_hamming.npy", labels_hc_hamming)
    np.save("labels_hierarchical_om.npy", labels_hc_om)
    
    print("✓ Labels sauvegardés")
