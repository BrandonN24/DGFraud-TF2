import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy import stats
import pandas as pd
import argparse

from utils.utils import preprocess_feature
from utils.data_loader import load_data_yelp
import json

def analyze_yelp_graph_structure(adj_list, features, labels):
    """
    Analyzes the graph structure of the Yelp dataset to determine key properties
    that affect sampling strategies.
    
    Args:
        adj_list: List of adjacency matrices
        features: Node feature matrix
        labels: Node labels (fraud/non-fraud)
        
    Returns:
        dict: Dictionary containing graph structure metrics
    """
    results = {}
    
    # Create NetworkX graphs for analysis
    graphs = []
    for adj_mat in adj_list:
        G = nx.from_scipy_sparse_array(adj_mat)
        graphs.append(G)
    
    # 1. Degree distribution analysis
    for i, G in enumerate(graphs):
        degrees = [d for _, d in G.degree()]
        results[f'graph_{i}_avg_degree'] = np.mean(degrees)
        results[f'graph_{i}_max_degree'] = np.max(degrees)
        results[f'graph_{i}_degree_std'] = np.std(degrees)
        
        # Plot degree distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(degrees, bins=50, kde=True)
        plt.title(f"Degree Distribution for Graph {i}")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.savefig(f"degree_dist_graph_{i}.png")
        plt.close()
        
        # Check if degree distribution follows power law (scale-free)
        if len(degrees) > 10:  # Need enough data points
            # Filter out zeros
            non_zero_degrees = [d for d in degrees if d > 0]
            if len(non_zero_degrees) > 10:
                # Fit power law
                fit = stats.powerlaw.fit(non_zero_degrees)
                alpha = fit[0]
                results[f'graph_{i}_powerlaw_alpha'] = alpha
                results[f'graph_{i}_is_scale_free'] = alpha > 1 and alpha < 3
    
    # 2. Homophily analysis (do similar nodes connect to each other?)
    for i, G in enumerate(graphs):
        homophily_score = compute_homophily(G, labels)
        results[f'graph_{i}_homophily'] = homophily_score
    
    # 3. Clustering coefficient (community structure)
    for i, G in enumerate(graphs):
        results[f'graph_{i}_clustering_coef'] = nx.average_clustering(G)
    
    # 4. Connected components
    for i, G in enumerate(graphs):
        components = list(nx.connected_components(G))
        results[f'graph_{i}_num_components'] = len(components)
        if len(components) > 0:
            results[f'graph_{i}_largest_component_size'] = len(max(components, key=len))
    
    # 5. Dense subgraph detection (potential fraud rings)
    for i, G in enumerate(graphs):
        densest_subgraph = find_densest_subgraph(G)
        results[f'graph_{i}_densest_subgraph_size'] = len(densest_subgraph)
        results[f'graph_{i}_densest_subgraph_density'] = compute_density(G, densest_subgraph)
    
    return results

def compute_homophily(G, labels):
    """
    Computes homophily score for the graph based on node labels.
    
    Args:
        G: NetworkX graph
        labels: Node labels
        
    Returns:
        float: Homophily score between -1 (heterophily) and 1 (homophily)
    """
    same_label_edges = 0
    diff_label_edges = 0
    
    for u, v in G.edges():
        if u < len(labels) and v < len(labels):
            if labels[u] == labels[v]:
                same_label_edges += 1
            else:
                diff_label_edges += 1
    
    total_edges = same_label_edges + diff_label_edges
    if total_edges == 0:
        return 0
    
    return (same_label_edges - diff_label_edges) / total_edges

def find_densest_subgraph(G, min_size=3, max_size=20):
    """
    Uses a greedy approach to find approximately the densest subgraph.
    
    Args:
        G: NetworkX graph
        min_size: Minimum subgraph size to consider
        max_size: Maximum subgraph size to consider
        
    Returns:
        set: Nodes in the densest subgraph
    """
    # Simplistic greedy approach for smaller graphs
    if len(G) <= 1000:
        best_density = 0
        best_subgraph = set()
        
        # Start with high-degree nodes as candidates
        candidates = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:50]
        seed_nodes = [node for node, _ in candidates]
        
        for start_node in seed_nodes:
            subgraph = {start_node}
            frontier = set(G.neighbors(start_node))
            
            while len(subgraph) < max_size and frontier:
                # Add node that contributes most edges to current subgraph
                best_node = None
                best_edge_contribution = -1
                
                for node in frontier:
                    edge_contribution = sum(1 for n in subgraph if G.has_edge(node, n))
                    if edge_contribution > best_edge_contribution:
                        best_node = node
                        best_edge_contribution = edge_contribution
                
                if best_node is None:
                    break
                    
                subgraph.add(best_node)
                frontier.remove(best_node)
                frontier.update(n for n in G.neighbors(best_node) if n not in subgraph and n not in frontier)
                
                if len(subgraph) >= min_size:
                    density = compute_density(G, subgraph)
                    if density > best_density:
                        best_density = density
                        best_subgraph = subgraph.copy()
        
        return best_subgraph
    
    # For larger graphs, use a faster approximate approach
    else:
        return set(sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_size])

def compute_density(G, nodes):
    """
    Computes the density of a subgraph.
    
    Args:
        G: NetworkX graph
        nodes: Set of nodes forming the subgraph
        
    Returns:
        float: Density of the subgraph
    """
    if len(nodes) <= 1:
        return 0
    
    edges = sum(1 for u in nodes for v in nodes if u < v and G.has_edge(u, v))
    max_possible_edges = len(nodes) * (len(nodes) - 1) / 2
    return edges / max_possible_edges if max_possible_edges > 0 else 0

def analyze_feature_distributions(features, labels):
    """
    Analyzes the distribution of node features.
    
    Args:
        features: Node feature matrix
        labels: Node labels
        
    Returns:
        dict: Dictionary with feature distribution metrics
    """
    results = {}
    
    # 1. Feature value distribution
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    
    results['feature_means'] = feature_means
    results['feature_stds'] = feature_stds
    results['feature_ranges'] = feature_maxs - feature_mins
    
    # 2. Feature correlations with labels
    if labels is not None and labels.shape[0] == features.shape[0]:
        flat_labels = labels.flatten() if labels.ndim > 1 else labels
        feature_correlations = []
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], flat_labels)[0, 1]
            feature_correlations.append(corr)
        
        results['feature_label_correlations'] = feature_correlations
        
        # Get top 10 most predictive features
        top_indices = np.argsort(np.abs(feature_correlations))[-10:]
        results['top_predictive_features'] = top_indices.tolist()
        results['top_feature_correlations'] = [feature_correlations[i] for i in top_indices]
    
    # 3. Feature sparsity
    feature_sparsity = np.mean(features == 0, axis=0)
    results['feature_sparsity'] = feature_sparsity
    
    # 4. Check for high-dimensional features
    results['is_high_dimensional'] = features.shape[1] > 100
    
    # 5. Feature value distributions
    # Plot histograms for top features
    if 'top_predictive_features' in results:
        for idx in results['top_predictive_features'][:5]:  # Plot top 5
            plt.figure(figsize=(10, 6))
            sns.histplot(features[:, idx], bins=50, kde=True)
            plt.title(f"Distribution of Feature {idx}")
            plt.savefig(f"feature_dist_{idx}.png")
            plt.close()
    
    return results

def analyze_pairwise_distances(features, sample_size=1000):
    """
    Analyzes pairwise distances between node features to inform kernel width.
    
    Args:
        features: Node feature matrix
        sample_size: Number of nodes to sample for analysis
        
    Returns:
        dict: Dictionary with distance metrics
    """
    results = {}
    
    # Sample nodes if dataset is large
    if features.shape[0] > sample_size:
        indices = np.random.choice(features.shape[0], sample_size, replace=False)
        feature_sample = features[indices]
    else:
        feature_sample = features
    
    # Compute pairwise distances
    distances = pairwise_distances(feature_sample, metric='euclidean')
    
    # Remove self-comparisons
    mask = ~np.eye(distances.shape[0], dtype=bool)
    distances = distances[mask].flatten()
    
    # Calculate distance statistics
    results['mean_distance'] = np.mean(distances)
    results['median_distance'] = np.median(distances)
    results['std_distance'] = np.std(distances)
    results['min_distance'] = np.min(distances)
    results['max_distance'] = np.max(distances)
    
    # Suggested kernel width (heuristic: median distance)
    results['suggested_kernel_width'] = results['median_distance']
    
    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=50, kde=True)
    plt.axvline(results['median_distance'], color='r', linestyle='--', 
                label=f"Median: {results['median_distance']:.4f}")
    plt.title("Distribution of Pairwise Distances")
    plt.xlabel("Euclidean Distance")
    plt.legend()
    plt.savefig("distance_distribution.png")
    plt.close()
    
    return results

def compare_sampling_strategies(neigh_dict, features, sample_size=5, num_nodes=100):
    """
    Compares different sampling strategies for a subset of nodes.
    
    Args:
        neigh_dict: Dictionary of node neighbors
        features: Node feature matrix
        sample_size: Number of neighbors to sample
        num_nodes: Number of nodes to test
        
    Returns:
        dict: Dictionary with comparison metrics
    """
    results = {}
    
    # Choose test nodes
    if len(neigh_dict) > num_nodes:
        test_nodes = np.random.choice(list(neigh_dict.keys()), 
                                      size=num_nodes, 
                                      replace=False)
    else:
        test_nodes = list(neigh_dict.keys())
    
    # Define sampling strategies
    def uniform_sampling(n, ns):
        if len(ns) <= sample_size:
            return ns
        return np.random.choice(ns, sample_size, replace=False)
    
    def distance_based_sampling(n, ns):
        if len(ns) == 0:
            return []
        
        # Calculate distances
        node_feat = features[n]
        neigh_feats = features[ns]
        diffs = node_feat - neigh_feats
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Convert to probabilities (closer = higher probability)
        probs = 1.0 / (distances + 1e-10)
        probs = probs / np.sum(probs)
        
        # Sample
        if len(ns) <= sample_size:
            return ns
        return np.random.choice(ns, sample_size, replace=False, p=probs)
    
    def temperature_sampling(n, ns, temp=0.75):
        if len(ns) == 0:
            return []
        
        # Calculate distances
        node_feat = features[n]
        neigh_feats = features[ns]
        diffs = node_feat - neigh_feats
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Convert to consistency scores
        scores = np.exp(-distances)
        
        # Apply temperature
        probs = scores ** (1.0/temp)
        probs = probs / np.sum(probs)
        
        # Sample
        if len(ns) <= sample_size:
            return ns
        return np.random.choice(ns, sample_size, replace=False, p=probs)
    
    # Test strategies
    sampling_results = {
        'uniform': [],
        'distance': [],
        'temperature': []
    }
    
    for node in test_nodes:
        neighbors = neigh_dict.get(node, [])
        if len(neighbors) > 0:
            uniform_samples = uniform_sampling(node, neighbors)
            distance_samples = distance_based_sampling(node, neighbors)
            temp_samples = temperature_sampling(node, neighbors)
            
            # Store for analysis
            sampling_results['uniform'].append(uniform_samples)
            sampling_results['distance'].append(distance_samples)
            sampling_results['temperature'].append(temp_samples)
    
    # Analyze diversity of samples
    for strategy, samples in sampling_results.items():
        if len(samples) > 0:
            # Calculate average distance between sampled neighbors
            avg_diversity = []
            for node_samples in samples:
                if len(node_samples) > 1:
                    sample_features = features[node_samples]
                    dists = pairwise_distances(sample_features)
                    # Remove self-comparisons
                    mask = ~np.eye(dists.shape[0], dtype=bool)
                    avg_diversity.append(np.mean(dists[mask]))
            
            results[f'{strategy}_diversity'] = np.mean(avg_diversity) if avg_diversity else 0
    
    # Compare how different the samples are between strategies
    strategy_differences = {}
    strategies = list(sampling_results.keys())
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            jaccard_similarities = []
            
            for samples1, samples2 in zip(sampling_results[s1], sampling_results[s2]):
                set1 = set(samples1)
                set2 = set(samples2)
                if set1 or set2:  # Avoid division by zero
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    jaccard_similarities.append(jaccard)
            
            key = f'{s1}_vs_{s2}_similarity'
            strategy_differences[key] = np.mean(jaccard_similarities) if jaccard_similarities else 0
    
    results.update(strategy_differences)
    return results

def evaluate_sampling_impact(adj_list, features, labels, sample_size=5, num_trials=5):
    """
    Evaluates how different sampling strategies affect prediction performance.
    This is a simplified evaluation and requires the full model to be accurate.
    
    Args:
        adj_list: List of adjacency matrices
        features: Node feature matrix
        labels: Node labels
        sample_size: Number of neighbors to sample
        num_trials: Number of trials to run
        
    Returns:
        dict: Dictionary with performance metrics
    """
    # This function would require implementing a simplified version of the full model
    # and is complex for this example. Instead, here's a placeholder that describes
    # what you would implement:
    
    print("Sampling impact evaluation would:")
    print("1. Create a mini-batch of test nodes")
    print("2. Apply different sampling strategies")
    print("3. Run forward pass of a simplified GNN model")
    print("4. Compare prediction accuracy across strategies")
    print("5. Report which strategy works best")
    
    # In practice, you'd implement a simplified version of GraphConsis here
    # and evaluate how different sampling strategies affect performance
    
    return {
        "message": "This is a complex evaluation that requires running the full model",
        "next_steps": "Implement as a separate evaluation script that uses the actual model"
    }

def run_all_analyses(adj_list, features, labels):
    """
    Runs all analyses and returns a comprehensive report.
    
    Args:
        adj_list: List of adjacency matrices
        features: Node feature matrix
        labels: Node labels
        
    Returns:
        dict: Dictionary with all analysis results
    """
    results = {}
    
    print("Analyzing graph structure...")
    graph_results = analyze_yelp_graph_structure(adj_list, features, labels)
    results['graph_structure'] = graph_results
    
    print("Analyzing feature distributions...")
    feature_results = analyze_feature_distributions(features, labels)
    results['feature_distributions'] = feature_results
    
    print("Analyzing pairwise distances...")
    distance_results = analyze_pairwise_distances(features)
    results['pairwise_distances'] = distance_results
    
    # Extract neighbor dictionaries from adjacency matrices
    neigh_dicts = []
    for adj_mat in adj_list:
        neigh_dict = {}
        rows, cols = adj_mat.nonzero()
        for i, j in zip(rows, cols):
            if i not in neigh_dict:
                neigh_dict[i] = []
            neigh_dict[i].append(j)
        neigh_dicts.append(neigh_dict)
    
    print("Comparing sampling strategies...")
    if neigh_dicts:
        sampling_results = compare_sampling_strategies(neigh_dicts[0], features)
        results['sampling_comparison'] = sampling_results
    
    # Suggest optimal parameters based on analysis
    print("Generating parameter recommendations...")
    results['recommendations'] = generate_recommendations(results)
    
    return results

def generate_recommendations(analysis_results):
    """
    Generates recommendations for sampling parameters based on analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        
    Returns:
        dict: Dictionary with recommended parameters
    """
    recommendations = {}
    
    # 1. Kernel width for consistency score
    if 'pairwise_distances' in analysis_results:
        dist_results = analysis_results['pairwise_distances']
        recommendations['kernel_width'] = dist_results.get('suggested_kernel_width', 1.0)
    
    # 2. Sampling temperature based on graph structure
    if 'graph_structure' in analysis_results:
        graph_results = analysis_results['graph_structure']
        
        # Check if graph is scale-free
        is_scale_free = any(graph_results.get(f'graph_{i}_is_scale_free', False) 
                           for i in range(10))  # Check first 10 graphs
        
        if is_scale_free:
            # Higher temperature for scale-free graphs to increase exploration
            recommendations['temperature'] = 0.9
        else:
            recommendations['temperature'] = 0.7
    
    # 3. Feature similarity metric based on dimensionality and sparsity
    if 'feature_distributions' in analysis_results:
        feature_results = analysis_results['feature_distributions']
        is_high_dim = feature_results.get('is_high_dimensional', False)
        avg_sparsity = np.mean(feature_results.get('feature_sparsity', [0]))
        
        if is_high_dim or avg_sparsity > 0.8:
            recommendations['similarity_metric'] = 'cosine'
        else:
            recommendations['similarity_metric'] = 'euclidean'
    
    # 4. Homophily-based recommendations
    if 'graph_structure' in analysis_results:
        graph_results = analysis_results['graph_structure']
        avg_homophily = np.mean([
            graph_results.get(f'graph_{i}_homophily', 0) 
            for i in range(10) 
            if f'graph_{i}_homophily' in graph_results
        ])
        
        if avg_homophily < -0.2:  # Heterophilic graph
            recommendations['invert_similarity'] = True
        else:
            recommendations['invert_similarity'] = False
    
    # 5. Adaptive epsilon threshold
    if 'pairwise_distances' in analysis_results:
        dist_results = analysis_results['pairwise_distances']
        # Set threshold relative to distance distribution
        mean_dist = dist_results.get('mean_distance', 1.0)
        recommendations['epsilon_threshold'] = np.exp(-mean_dist) * 0.5
    
    # 6. Sampling strategy recommendation
    if 'sampling_comparison' in analysis_results:
        sampling_results = analysis_results['sampling_comparison']
        
        # Compare diversity scores
        uniform_div = sampling_results.get('uniform_diversity', 0)
        distance_div = sampling_results.get('distance_diversity', 0)
        temp_div = sampling_results.get('temperature_diversity', 0)
        
        # Select strategy with highest diversity
        best_diversity = max(uniform_div, distance_div, temp_div)
        if best_diversity == uniform_div:
            recommendations['best_strategy'] = 'uniform'
        elif best_diversity == distance_div:
            recommendations['best_strategy'] = 'distance'
        else:
            recommendations['best_strategy'] = 'temperature'
    
    return recommendations

# Example usage:
if __name__ == "__main__":
    # This would be used in your main script after loading the data
    # Example:
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_size', type=float, default=0.8,
                        help='training set percentage')
    
    args = parser.parse_args()

    adj_list, features, split_ids, y = load_data_yelp(train_size=args.train_size)
    label = np.array([y]).T
    features = preprocess_feature(features, to_tuple=False)
    features = np.array(features.todense())
    
    analysis_results = run_all_analyses(adj_list, features, label)
    print(json.dumps(analysis_results['recommendations'], indent=2))
    pass