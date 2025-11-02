"""
Graph utility functions for FireGNN framework.
Provides functions for graph construction, fuzzy rule computation,
and auxiliary task generation.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import scipy.stats
from tqdm import tqdm
import pickle
import os


def compute_topological_features(G):
    """
    Compute topological features for all nodes in the graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        dict: Dictionary containing topological features for each node
    """
    print("Computing topological features...")
    
    # Basic features
    degrees = np.array([G.degree(n) for n in G.nodes()])
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
    
    # Two-hop agreement
    two_hop_agreement = []
    for n in tqdm(G.nodes(), desc="Computing two-hop agreement"):
        d2 = set(nx.single_source_shortest_path_length(G, n, cutoff=2).keys())
        d2.discard(n)
        if d2:
            lbls = [G.nodes[m]['y'] for m in d2]
            two_hop_agreement.append(lbls.count(G.nodes[n]['y']) / len(lbls))
        else:
            two_hop_agreement.append(0.0)
    two_hop_agreement = np.array(two_hop_agreement)
    
    # Centrality measures
    eigenvector_centrality = np.array(list(nx.eigenvector_centrality_numpy(G, weight='weight').values()))
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    
    # Average edge weight
    avg_edge_weight = []
    for n in G.nodes():
        if G.degree(n) > 0:
            weights = [G.edges[n, m]['weight'] for m in G.neighbors(n)]
            avg_edge_weight.append(np.mean(weights))
        else:
            avg_edge_weight.append(0.0)
    avg_edge_weight = np.array(avg_edge_weight)
    
    return {
        'degrees': degrees,
        'clustering': clustering,
        'two_hop_agreement': two_hop_agreement,
        'eigenvector_centrality': eigenvector_centrality,
        'degree_centrality': degree_centrality,
        'avg_edge_weight': avg_edge_weight
    }


def compute_auxiliary_features(G, num_classes):
    """
    Compute auxiliary features for multi-task learning.
    
    Args:
        G: NetworkX graph object
        num_classes: Number of classes in the dataset
        
    Returns:
        dict: Dictionary containing auxiliary features
    """
    print("Computing auxiliary features...")
    
    # Similarity aggregation scores
    sim_scores = []
    homophily = []
    entropies = []
    
    for i in tqdm(G.nodes(), desc="Computing auxiliary features"):
        neighbors = list(G.neighbors(i))
        neighbors_2hop = set([k for j in neighbors for k in G.neighbors(j) if k != i])
        
        score = np.zeros(num_classes)
        denom = 0.0
        same_label_sum = 0.0
        total_sum = 0.0
        
        # Direct neighbors
        for j in neighbors:
            if not G.nodes[j].get('test', False):
                label = G.nodes[j]['y']
                weight = G.edges[i, j]['weight']
                score[label] += weight
                denom += weight
                total_sum += weight
                if label == G.nodes[i]['y']:
                    same_label_sum += weight
        
        # 2-hop neighbors
        for k in neighbors_2hop:
            if not G.nodes[k].get('test', False):
                label = G.nodes[k]['y']
                weight = G.edges[i, k]['weight'] if (i, k) in G.edges else 0.3
                score[label] += 0.3 * weight
                denom += 0.3 * weight
                total_sum += 0.3 if (i, k) not in G.edges else G.edges[i, k]['weight']
                if label == G.nodes[i]['y']:
                    same_label_sum += 0.3 if (i, k) not in G.edges else G.edges[i, k]['weight']
        
        # Normalize scores
        score = score / (denom + 1e-6)
        score = np.where(np.isnan(score) | (score <= 0), 1e-6, score)
        score = np.exp(score / 0.5) / np.sum(np.exp(score / 0.5))
        
        # Compute entropy
        entropy = scipy.stats.entropy(score) / np.log(num_classes)
        
        sim_scores.append(score)
        entropies.append(entropy)
        h = same_label_sum / (total_sum + 1e-6)
        homophily.append(h)
    
    return {
        'sim_scores': np.array(sim_scores),
        'homophily': np.array(homophily),
        'entropies': np.array(entropies)
    }


def create_fuzzy_rules(features, num_rules=10):
    """
    Initialize fuzzy rule centers and widths using K-means clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        num_rules: Number of fuzzy rules
        
    Returns:
        tuple: (centers, widths) for fuzzy rules
    """
    # Use K-means to find rule centers
    kmeans = KMeans(n_clusters=num_rules, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_
    
    # Compute widths based on cluster standard deviations
    widths = np.zeros_like(centers)
    for i in range(num_rules):
        cluster_points = features[cluster_labels == i]
        if len(cluster_points) > 0:
            widths[i] = np.std(cluster_points, axis=0) + 1e-6
        else:
            widths[i] = np.std(features, axis=0) + 1e-6
    
    return centers, widths


def build_knn_graph_from_features(features, labels, k=8, metric='cosine', 
                                 add_label_edges=True, rewire_edges=True):
    """
    Build k-NN graph from feature vectors with optional label-based enhancements.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        k: Number of neighbors
        metric: Distance metric for k-NN
        add_label_edges: Whether to add edges between same-label nodes
        rewire_edges: Whether to rewire edges for better homophily
        
    Returns:
        NetworkX graph object
    """
    print(f"Building k-NN graph with k={k}, metric={metric}")
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    dists, inds = nbrs.kneighbors(features)
    
    # Create initial edges
    edge_list = []
    for i in tqdm(range(len(features)), desc="Building initial edges"):
        for r in range(1, k):  # Skip self (r=0)
            j = inds[i, r]
            if i < j:  # Avoid duplicate edges
                sim = 1 - dists[i, r]  # Convert distance to similarity
                edge_list.append((i, j, sim))
    
    initial_edge_cnt = len(edge_list)
    label_based_add_cnt = label_based_add_success_cnt = 0
    # Add label-based edges
    if add_label_edges:
        print("Adding label-based edges...")
        for i in tqdm(range(len(features)), desc="Adding label edges"):
            same_label = [j for j in range(len(features)) if labels[j] == labels[i] and j != i]
            if same_label:
                knn_neighbors = inds[i, 1:]
                valid_neighbors = [j for j in same_label if j in knn_neighbors]
                if valid_neighbors:
                    sims = [1.0 - dists[i, np.where(inds[i, :] == j)[0][0]] for j in valid_neighbors]
                    top_k = np.argsort(sims)[-10:]
                    for idx in top_k:
                        j = valid_neighbors[idx]
                        if i < j:
                            label_based_add_cnt += 1
                            for a0, b0, _ in edge_list:
                              if a0 == i and b0 == j:
                                break
                            else:
                              label_based_add_success_cnt += 1
                            edge_list.append((i, j, sims[idx]))

    print(f'label_based_add_cnt: {label_based_add_cnt}')
    print(f'label_based_add_success_cnt: {label_based_add_success_cnt}')
    
    # Rewire edges for better homophily
    if rewire_edges:
        print("Rewiring edges for better homophily...")
        rewired_count = 0
        for i in tqdm(range(len(features)), desc="Rewiring edges"):
            neighbors = inds[i, 1:]
            to_remove = []
            for j in neighbors:
                if labels[j] != labels[i] and (1.0 - dists[i, np.where(inds[i, :] == j)[0][0]]) < 0.65:
                    to_remove.append(j)
            
            same_label = [j for j in range(len(features)) if labels[j] == labels[i] and j != i and j not in neighbors]
            if to_remove and same_label:
                sims = [1.0 - dists[i, np.where(inds[i, :] == j)[0][0]] if j in inds[i, :] else 0.0 for j in same_label]
                top_k = np.argsort(sims)[-len(to_remove):]
                for idx, j in zip(top_k, [same_label[k] for k in top_k]):
                    if i < j and sims[idx] > 0.0:
                        edge_list.append((i, j, sims[idx]))
                        rewired_count += 1
                edge_list = [(u, v, w) for u, v, w in edge_list if not (u == i and v in to_remove) or (v == i and u in to_remove)]
        
        print(f"Rewired {rewired_count} edges")  
    
    # Remove duplicates
    edge_list = list(set(edge_list))

    print(f'initial_edge_cnt: {initial_edge_cnt}')
    print(f'edge_cnt: {len(edge_list)}')
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in tqdm(range(len(features)), desc="Adding nodes"):
        G.add_node(i, x=features[i], y=labels[i])
    
    # Add edges
    for edge in tqdm(edge_list, desc="Adding edges"):
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    return G


def prepare_pytorch_geometric_data(G, topological_features=None, auxiliary_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph object
        topological_features: Optional pre-computed topological features
        auxiliary_features: Optional pre-computed auxiliary features
        
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Extract basic data
    x = np.array([G.nodes[n]['x'] for n in G.nodes()])
    y = np.array([G.nodes[n]['y'] for n in G.nodes()])
    
    # Create edge_index and edge_attr
    edge_list = []
    edge_attr_list = []
    for u, v, d in G.edges(data=True):
        edge_list.append([u, v])
        edge_list.append([v, u])
        edge_attr_list.append(d['weight'])
        edge_attr_list.append(d['weight'])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    # Create masks
    train_mask = np.array([G.nodes[n].get('train', True) for n in G.nodes()])
    val_mask = np.array([G.nodes[n].get('val', False) for n in G.nodes()])
    test_mask = np.array([G.nodes[n].get('test', False) for n in G.nodes()])
    
    # Compute topological features if not provided
    if topological_features is None:
        topological_features = compute_topological_features(G)
    
    # Stack and standardize topological features
    feature_names = ['degrees', 'clustering', 'two_hop_agreement', 
                    'eigenvector_centrality', 'degree_centrality', 'avg_edge_weight']
    
    topo_features = np.vstack([topological_features[name] for name in feature_names]).T
    scaler = StandardScaler()
    topo_features = scaler.fit_transform(topo_features)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        topo_features=torch.tensor(topo_features, dtype=torch.float),
        topo_scaler_mean=torch.tensor(scaler.mean_, dtype=torch.float),
        topo_scaler_scale=torch.tensor(scaler.scale_, dtype=torch.float)
    )
    
    # Add auxiliary features if provided
    if auxiliary_features is not None:
        data.sim_scores = torch.tensor(auxiliary_features['sim_scores'], dtype=torch.float)
        data.homophily = torch.tensor(auxiliary_features['homophily'], dtype=torch.float)
        data.entropies = torch.tensor(auxiliary_features['entropies'], dtype=torch.float)
    
    return data


def save_graph(G, filename):
    """
    Save graph to pickle file.
    
    Args:
        G: NetworkX graph object
        filename: Output filename
    """
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def load_graph(filename):
    """
    Load graph from pickle file.
    
    Args:
        filename: Input filename
        
    Returns:
        NetworkX graph object
    """
    with open(filename, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph from {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_device():
    """
    Get the best available device (MPS, CUDA, or CPU).
    
    Returns:
        torch.device: Device object
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 