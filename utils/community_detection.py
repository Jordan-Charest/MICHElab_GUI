import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.graph_objects as go  # If using Plotly for visualization
from tqdm import tqdm
import bct
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward, leaves_list
from copy import copy

def compute_coassignment_probability_louvain(W, N_iters=1000, gamma_min=0.7, gamma_max=2.2, threshold=True):
    if np.array_equal(W, W.T):
        directed = False
    else:
        directed = True

    gammas = np.linspace(gamma_min, gamma_max, N_iters)
    communities = np.zeros((W.shape[0], N_iters))

    for i in range(N_iters):
        if directed:
            communities[:, i], _ = bct.modularity_louvain_dir(W, gamma=gammas[i])
        else:
            communities[:, i], _ = bct.modularity_louvain_und(W, gamma=gammas[i])

    coassignment_matrix = compute_consensus_matrix_from_labels(communities)

    if threshold:
        communities_null = shuffle_communities(communities)
        coassignment_matrix_null = compute_consensus_matrix_from_labels(communities_null)
        coassignment_matrix -= np.mean(coassignment_matrix_null)
        coassignment_matrix[coassignment_matrix < 0] = 0

    return coassignment_matrix

def compute_coassignment_probability_kmeans(data, N_iters=1000, k_min=2, k_max=30, threshold=False):
       
    k = np.linspace(k_min, k_max, N_iters, endpoint=True).astype('int')
    clusters = np.zeros((data.shape[0], N_iters))

    for i in range(N_iters):
        clustering = KMeans(n_clusters=k[i], n_init='auto').fit(data)
        clusters[:, i] =  clustering.labels_

    coassignment_matrix = compute_consensus_matrix_from_labels(clusters)

    if threshold:
        communities_null = shuffle_communities(clusters)
        coassignment_matrix_null = compute_consensus_matrix_from_labels(communities_null)
        coassignment_matrix -= np.mean(coassignment_matrix_null)
        coassignment_matrix[coassignment_matrix < 0] = 0

    return coassignment_matrix

def shuffle_communities(communities): # TODO: implement 
    return communities

def compute_consensus_matrix_from_labels(communities):
    R = communities.shape[0]
    coassignment_matrix = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            coassignment_matrix[i, j] = len(np.where((communities[i, :] == communities[j, :]) == True)[0]) / \
                                        communities.shape[1]
            coassignment_matrix[j, i] = coassignment_matrix[i, j]
    return coassignment_matrix

# -------------------
# NUMBER OF CLUSTERS USING CLASSICAL METRICS
# -------------------
def plot_clustering_metrics(data_matrix, ks=range(2, 11), filename="clustering_metrics_analysis.svg", savefig=False, show=True, suptitle=None):
    """
    Computes and visualizes multiple clustering evaluation metrics to determine the optimal number of clusters (k).
    This function applies k-means clustering to the given dataset for different values of k and computes four widely used
    clustering validation metrics:
    - Inertia (Elbow Method): Measures the sum of squared distances of data points to their assigned cluster centers.
      Lower inertia suggests better clustering, but it typically decreases with k. The "elbow" point indicates the optimal k.
    - Silhouette Score: Evaluates cluster separation by measuring how similar each point is to its own cluster versus
      other clusters. Ranges from -1 (poor clustering) to 1 (well-separated clusters). A peak suggests the best k.
    - Davies-Bouldin Index (DBI): Measures cluster compactness and separation. Lower values indicate well-formed,
      distinct clusters.
    - Calinski-Harabasz Score: Assesses the ratio of between-cluster variance to within-cluster variance. Higher values
      indicate well-separated, compact clusters.
    Parameters:
        data_matrix (numpy.ndarray or pd.DataFrame): The dataset used for clustering (e.g., consensus matrix or z-scored features).
        ks (range or list, optional): Range of cluster numbers to evaluate. Default is 2 to 10.
        filename (str, optional): Name of the file to save the plot. Default is 'clustering_metrics_analysis.svg'.
    """
    # Convert DataFrame to NumPy array if necessary
    if hasattr(data_matrix, 'values'):
        data_matrix = data_matrix.values
    # Initialize lists for storing evaluation metrics
    inertia = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    # Compute clustering metrics for each k
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_matrix)
        # Compute metrics
        inertia.append(kmeans.inertia_)  # Sum of squared distances
        silhouette_scores.append(silhouette_score(data_matrix, labels))  # Silhouette score
        davies_bouldin_scores.append(davies_bouldin_score(data_matrix, labels))  # Davies-Bouldin index
        calinski_harabasz_scores.append(calinski_harabasz_score(data_matrix, labels))  # Calinski-Harabasz index
    # Ensure matching lengths before plotting
    min_len = min(len(ks), len(inertia), len(silhouette_scores), len(davies_bouldin_scores), len(calinski_harabasz_scores))
    ks = list(ks)[:min_len]
    inertia = inertia[:min_len]
    silhouette_scores = silhouette_scores[:min_len]
    davies_bouldin_scores = davies_bouldin_scores[:min_len]
    calinski_harabasz_scores = calinski_harabasz_scores[:min_len]
    # -------------------
    # PLOT CLUSTERING METRICS
    # -------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    if suptitle is not None: fig.suptitle(suptitle)
    # Elbow Method (Inertia)
    axes[0, 0].plot(ks, inertia, marker='o', label="Inertia", color="b")
    axes[0, 0].set_xlabel("Number of clusters (k)")
    axes[0, 0].set_ylabel("Inertia (sum of squared distances)")
    axes[0, 0].set_title("Elbow Method")
    axes[0, 0].legend()
    axes[0, 0].grid()
    # Silhouette Score
    axes[0, 1].plot(ks, silhouette_scores, marker='o', label="Silhouette Score", color="g")
    axes[0, 1].set_xlabel("Number of clusters (k)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].set_title("Silhouette Analysis")
    axes[0, 1].legend()
    axes[0, 1].grid()
    # Davies-Bouldin Index (lower is better)
    axes[1, 0].plot(ks, davies_bouldin_scores, marker='o', label="Davies-Bouldin Index", color="r")
    axes[1, 0].set_xlabel("Number of clusters (k)")
    axes[1, 0].set_ylabel("Davies-Bouldin Index")
    axes[1, 0].set_title("Davies-Bouldin Index (Lower is Better)")
    axes[1, 0].legend()
    axes[1, 0].grid()
    # Calinski-Harabasz Index (higher is better)
    axes[1, 1].plot(ks, calinski_harabasz_scores, marker='o', label="Calinski-Harabasz Score", color="purple")
    axes[1, 1].set_xlabel("Number of clusters (k)")
    axes[1, 1].set_ylabel("Calinski-Harabasz Score")
    axes[1, 1].set_title("Calinski-Harabasz Score (Higher is Better)")
    axes[1, 1].legend()
    axes[1, 1].grid()
    # Adjust layout
    plt.tight_layout()
    # Save figure
    if savefig:
        save_figure_with_date(plt.gcf(), filename)
        print(f"Clustering metrics plot saved as '{filename}'")
    # Show plots
    if show: plt.show()


def save_figure_with_date(fig, filename):
    """
    Saves a figure with a filename that includes the current date.
    Supports both Matplotlib and Plotly figures.
    Args:
        fig: The figure to save (matplotlib.figure.Figure or plotly.graph_objects.Figure).
        filename (str): The full filename including extension.
    Returns:
        str: The saved file path.
    """
    
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    base, ext = os.path.splitext(filename)
    file_path = f"{base}_{date_str}{ext}"
    if ext not in [".png", ".pdf", ".svg", ".html"]:
        raise ValueError("Unsupported file format. Use 'png', 'pdf', 'svg', or 'html'.")
    if isinstance(fig, plt.Figure):  # Matplotlib figure
        if ext == ".png":
            fig.savefig(file_path, dpi=600, bbox_inches='tight')
        else:
            fig.savefig(file_path, bbox_inches='tight')  # No dpi for vector formats
    elif isinstance(fig, go.Figure):  # Plotly figure
        if ext == ".html":
            fig.write_html(file_path)
        else:
            raise ValueError("Plotly figures can only be saved as .html files.")
    else:
        raise TypeError("Unsupported figure type. Provide a Matplotlib or Plotly figure.")
    return file_path


def hierarchical_clustering(mat, max_clusters=20, method='ward', show=False):

    linkage_mat = linkage(1 - mat, method=method) # See other methods to build dendrogram
    clusters = fcluster(linkage_mat, t=max_clusters, criterion='maxclust') # See other criteria to split the hierarchy

    if show:
        sb.clustermap(mat, col_linkage=linkage_mat, row_linkage=linkage_mat, cmap="viridis")

    return linkage_mat, clusters

def communities_colormap(communities, base_cmap=plt.cm.Dark2):
    
    min_community = np.min(communities)  # Lowest integer in data
    max_community = np.max(communities)  # Highest integer in data
    num_communities = max_community - min_community + 1  # Total unique classes

    # Extract `num_classes` colors from Dark2 in order
    colors = [plt.cm.Dark2(i) for i in range(num_communities)]
    custom_cmap = mcolors.ListedColormap(colors, name="custom_dark2")

    # Define boundaries from min_class to max_class (inclusive)
    bounds = np.arange(min_community, max_community + 2)  # +2 ensures last bin is included
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    return custom_cmap, norm

def communities_to_window(community_labels, elements_mask, shape):
    final_mat = np.full(shape, np.nan)  # Initialize with NaNs
    valid_indices = np.flatnonzero(elements_mask)  # Get positions of valid data points
    final_mat.flat[valid_indices] = community_labels  # Assign community labels
    return final_mat


def intercommunity(FC, communities, include_diagonal=False):
    
    communities = communities - 1 # Start at 0 instead of 1
    n_communities = len(set(communities)) # Number of communities

    community_connectivity = np.zeros((n_communities, n_communities))
    community_counter = np.zeros_like(community_connectivity)

    offset = 0 if include_diagonal else 1
    for row in range(FC.shape[0]):
        for col in range(row + offset, FC.shape[1]):
            
            community_row = communities[row]
            community_col = communities[col]

            if community_row > community_col:
                community_row, community_col = community_col, community_row # Make sure we're always filling the upper triangle

            community_connectivity[community_row,community_col] += FC[row,col]
            community_counter[community_row,community_col] += 1

    community_connectivity = np.divide(community_connectivity, community_counter)

    return community_connectivity

def reorder_communities(mat, communities):
    
    order = np.argsort(communities)
    reordered_mat = mat[order,:][:,order]

    return reordered_mat


def plot_reordered_and_communities(FC, communities, elements_mask, window_shape, include_diagonal=False, title="", cmap="viridis", communities_cmap=None, coassignment_mat=None):

    reordered_FC = reorder_communities(FC, communities)
    window_communities = communities_to_window(communities, elements_mask, window_shape)
    intercommunity_mat = intercommunity(FC, communities, include_diagonal=include_diagonal)

    if communities_cmap is None:
        communities_cmap = communities_colormap(communities)[0]

    if coassignment_mat is not None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fig.suptitle(title, fontsize=14)
    
    im1 = axes[0].imshow(reordered_FC, cmap=cmap, aspect='equal')
    axes[0].set_title("reordered FC")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(intercommunity_mat.T, cmap=cmap, aspect='equal')
    axes[1].set_title("inter-community")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(window_communities, cmap=communities_cmap, aspect='equal')
    axes[2].set_title("Communities")
    fig.colorbar(im3, ax=axes[2])

    if coassignment_mat is not None:
        im4 = axes[3].imshow(coassignment_mat, cmap=cmap, aspect='equal')
        axes[3].set_title("Coassignment matrix")
        fig.colorbar(im4, ax=axes[3])

    return fig, axes

def plot_reordered_and_communities_batch(FC_list, communities_list, elements_mask, window_shape, include_diagonal=False, title="", cmap="viridis", communities_cmap=None, coassignment_mat_list=None, row_titles=None):

    n_elems = len(FC_list)

    if coassignment_mat_list is not None:
        fig, axes = plt.subplots(n_elems, 4, figsize=(24, 6*n_elems))
    else:
        fig, axes = plt.subplots(n_elems, 3, figsize=(18, 6*n_elems))

    fig.suptitle(title, fontsize=14)

    for i in range(len(FC_list)):
    
        reordered_FC = reorder_communities(FC_list[i], communities_list[i])
        window_communities = communities_to_window(communities_list[i], elements_mask, window_shape)
        intercommunity_mat = intercommunity(FC_list[i], communities_list[i], include_diagonal=include_diagonal)

        if communities_cmap is None:
            communities_cmap = communities_colormap(communities_list[i])[0]
        
        im1 = axes[i, 0].imshow(reordered_FC, cmap=cmap, aspect='equal')
        axes[i, 0].set_title(row_titles[i])
        fig.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(intercommunity_mat.T, cmap=cmap, aspect='equal')
        axes[i, 1].set_title("inter-community")
        fig.colorbar(im2, ax=axes[i, 1])

        im3 = axes[i, 2].imshow(window_communities, cmap=communities_cmap, aspect='equal')
        axes[i, 2].set_title("Communities")
        fig.colorbar(im3, ax=axes[i, 2])

        if coassignment_mat_list is not None:
            im4 = axes[i, 3].imshow(coassignment_mat_list[i], cmap=cmap, aspect='equal')
            axes[i, 3].set_title("Coassignment matrix")
            fig.colorbar(im4, ax=axes[i, 3])

        fig.tight_layout(pad=4)

    return fig, axes

def set_diagonal(mat, val):

    new_mat = copy(mat)
    np.fill_diagonal(new_mat, val)

    return new_mat