import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# CLUSTERING ALGORITHMS

# ============================================
# 1. K-MEDOIDS CLUSTERING

def kmedoids_clustering(dist_matrix, n_clusters=3, method='pam'):
    """
    Appliquer K-Medoids clustering
    
    Args:
        dist_matrix: matrice de distances (n_samples, n_samples)
        n_clusters: nombre de clusters
        method: 'pam' (default) ou 'alternate'
    
    Returns:
        labels: array de labels de cluster (0, 1, 2, ...)
        medoids: indices des medoids
    """
    # Convertir matrice de distances en condensed form
    condensed_dist = squareform(dist_matrix)
    
    # Appliquer KMedoids
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method=method, random_state=42)
    labels = kmedoids.fit_predict(dist_matrix)
    medoids = kmedoids.medoid_indices_
    
    return labels, medoids



# 2. HIERARCHICAL CLUSTERING


def hierarchical_clustering(dist_matrix, n_clusters=3, linkage_method='ward'):
    """
    Appliquer Hierarchical Clustering (Agglomerative)
    
    Args:
        dist_matrix: matrice de distances (n_samples, n_samples)
        n_clusters: nombre de clusters
        linkage_method: 'ward', 'complete', 'average', 'single'
    
    Returns:
        labels: array de labels de cluster
        linkage_matrix: matrice de linkage (pour dendrogram)
    """
    # Convertir en condensed distance matrix
    condensed_dist = squareform(dist_matrix)
    
    # Calculer linkage
    linkage_matrix = linkage(condensed_dist, method=linkage_method)
    
    # Couper le dendrogram à n_clusters
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1  # -1 pour 0-indexing
    
    return labels, linkage_matrix


def plot_dendrogram(linkage_matrix, title="Dendrogram", filename=None):
    """
    Tracer le dendrogram
    
    Args:
        linkage_matrix: matrice de linkage
        title: titre du graphique
        filename: si fourni, sauvegarder le graphique
    """
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


# 3. VISUALISER LES CLUSTERS


def plot_clusters_2d(dist_matrix, labels, title="Clusters", filename=None):
    """
    Tracer les clusters en 2D (en utilisant MDS pour réduction dimensionnalité)
    
    Args:
        dist_matrix: matrice de distances
        labels: labels des clusters
        title: titre du graphique
        filename: si fourni, sauvegarder
    """
    from sklearn.manifold import MDS
    
    # MDS pour réduire en 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(dist_matrix)
    
    # Plot
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



# TEST


if __name__ == "__main__":
    import pandas as pd
    
    # Charger données
    sequences = pd.read_csv("sequences_m1.csv").values
    dist_hamming = np.load("distance_matrix_hamming.npy")
    dist_om = np.load("distance_matrix_om.npy")
    
    print("=" * 60)
    print("CLUSTERING")
    print("=" * 60)
    
    # K-MEDOIDS avec Hamming
    print("\n1. K-MEDOIDS + HAMMING DISTANCE")
    print("-" * 60)
    labels_kmed_hamming, medoids_hamming = kmedoids_clustering(dist_hamming, n_clusters=3)
    print(f"Labels: {np.unique(labels_kmed_hamming)}")
    print(f"Cluster sizes: {np.bincount(labels_kmed_hamming)}")
    print(f"Medoids: {medoids_hamming}")
    
    # K-MEDOIDS avec OM
    print("\n2. K-MEDOIDS + OPTIMAL MATCHING")
    print("-" * 60)
    labels_kmed_om, medoids_om = kmedoids_clustering(dist_om, n_clusters=3)
    print(f"Labels: {np.unique(labels_kmed_om)}")
    print(f"Cluster sizes: {np.bincount(labels_kmed_om)}")
    print(f"Medoids: {medoids_om}")
    
    # HIERARCHICAL avec Hamming
    print("\n3. HIERARCHICAL + HAMMING DISTANCE")
    print("-" * 60)
    labels_hc_hamming, linkage_hc_hamming = hierarchical_clustering(dist_hamming, n_clusters=3, linkage_method='ward')
    print(f"Labels: {np.unique(labels_hc_hamming)}")
    print(f"Cluster sizes: {np.bincount(labels_hc_hamming)}")
    
    # HIERARCHICAL avec OM
    print("\n4. HIERARCHICAL + OPTIMAL MATCHING")
    print("-" * 60)
    labels_hc_om, linkage_hc_om = hierarchical_clustering(dist_om, n_clusters=3, linkage_method='ward')
    print(f"Labels: {np.unique(labels_hc_om)}")
    print(f"Cluster sizes: {np.bincount(labels_hc_om)}")
    
    # VISUALIZATIONS
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
    
    # Sauvegarder labels
    np.save("labels_kmedoids_hamming.npy", labels_kmed_hamming)
    np.save("labels_kmedoids_om.npy", labels_kmed_om)
    np.save("labels_hierarchical_hamming.npy", labels_hc_hamming)
    np.save("labels_hierarchical_om.npy", labels_hc_om)
    
    print("✓ Labels sauvegardés")
