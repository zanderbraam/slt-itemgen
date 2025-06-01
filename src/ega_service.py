"""
Service functions for Exploratory Graph Analysis (EGA) steps.

This module contains functions related to calculating similarity matrices,
constructing networks (TMFG, EBICglasso), performing community detection,
and calculating fit metrics for the EGA pipeline.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, shrunk_covariance
from sklearn.exceptions import ConvergenceWarning
import warnings
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
import time
import traceback
import re

# Attempt to import optional igraph and set a flag
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    # We won't raise an error here, but functions relying on igraph will.

def label_to_index(label: str) -> int:
    """
    Convert 'Item N' → N-1, handling spaces, underscores, and hyphens.
    Raise ValueError if the pattern is absent.

    Args:
        label: Item label like "Item 17", "Item_17", or "Item-17"

    Returns:
        Zero-based index (e.g., "Item 17" → 16)

    Raises:
        ValueError: If label doesn't match the expected pattern
    """
    m = re.search(r'\bItem[_\s-]*?(\d+)\b', label)
    if not m:
        raise ValueError(f"Cannot parse item label: {label!r}")
    return int(m.group(1)) - 1

def calculate_similarity_matrix(
    embeddings: np.ndarray | sp.spmatrix,
    dense_output: bool = True,
) -> np.ndarray:
    """Calculates the cosine similarity matrix for a given set of embeddings.

    Args:
        embeddings: A matrix where rows represent items and columns represent
            embedding dimensions. Can be a dense NumPy array or a sparse
            SciPy matrix.
        dense_output: If True, returns dense array. If False and input is sparse,
            attempts to return sparse result to save memory.

    Returns:
        A square NumPy array of shape (n_items, n_items) containing the
        pairwise cosine similarities between item embeddings.
        The diagonal elements will be 1.0.

    Raises:
        ValueError: If the input embeddings matrix is not 2-dimensional.
        TypeError: If the input is not a NumPy array or SciPy sparse matrix.

    Example:
        >>> import numpy as np
        >>> dense_embeddings = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        >>> similarity = calculate_similarity_matrix(dense_embeddings)
        >>> print(similarity.round(2))
        [[1.   0.   0.71]
         [0.   1.   0.71]
         [0.71 0.71 1.  ]]

        >>> import scipy.sparse as sp
        >>> sparse_embeddings = sp.csr_matrix([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        >>> similarity_sparse = calculate_similarity_matrix(sparse_embeddings)
        >>> print(similarity_sparse.round(2))
        [[1.   0.   0.71]
         [0.   1.   0.71]
         [0.71 0.71 1.  ]]
    """
    if not isinstance(embeddings, (np.ndarray, sp.spmatrix)):
        raise TypeError(
            "Input embeddings must be a NumPy array or SciPy sparse matrix."
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Input embeddings must be 2-dimensional (shape: n_items x n_features), but got {embeddings.ndim} dimensions."
        )
    if embeddings.shape[0] < 2:
        # Need at least 2 items to calculate pairwise similarity
        return np.array([[1.0]]) if embeddings.shape[0] == 1 else np.array([[]])

    # Calculate cosine similarity with appropriate output format
    if sp.issparse(embeddings) and not dense_output:
        # Try to keep sparse for memory efficiency
        try:
            similarity_matrix = cosine_similarity(embeddings, dense_output=False)
            # Convert to dense for post-processing since we need to modify diagonal
            if sp.issparse(similarity_matrix):
                similarity_matrix = similarity_matrix.toarray()
        except TypeError:
            # Fallback if dense_output parameter not supported
            similarity_matrix = cosine_similarity(embeddings)
    else:
        similarity_matrix = cosine_similarity(embeddings)

    # Ensure diagonal is exactly 1.0 (cosine_similarity might have minor precision issues)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Clip values to [-1.0, 1.0] just in case of floating point errors
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix

def calculate_wto(similarity_matrix: np.ndarray) -> np.ndarray:
    """Calculates the Weighted Topological Overlap (wTO) matrix.

    The wTO measure quantifies the similarity between two nodes based not only
    on their direct connection strength but also on the similarity of their
    connection patterns with other nodes in the network.

    The formula used is:
        wTO_ij = (L_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where:
        - a_ij is the absolute similarity between node i and node j (|similarity_matrix[i, j]|)
        - L_ij = sum_{k != i, j} (a_ik * a_jk) is the shared neighbor strength
        - k_i = sum_{k != i} a_ik is the total connection strength (node degree) of node i

    Args:
        similarity_matrix: A square (n_items, n_items) NumPy array of pairwise
            similarities (e.g., cosine similarity or correlation). Values are
            expected to be between -1 and 1. Diagonal elements are ignored.

    Returns:
        A square NumPy array of shape (n_items, n_items) containing the
        pairwise wTO values. Values range from 0 to 1. Diagonal elements
        are set to 1.0.

    Raises:
        ValueError: If the similarity matrix is not square, not 2D, or has less than 2 nodes.
        TypeError: If similarity_matrix is not a NumPy array.

    References:
        Zhang, B., & Horvath, S. (2005). A general framework for weighted gene
        co-expression network analysis. Statistical applications in genetics
        and molecular biology, 4(1).
    """
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("Input similarity_matrix must be a NumPy array.")
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Input similarity_matrix must be a square 2D array.")

    n_items = similarity_matrix.shape[0]
    if n_items < 2:
        # wTO requires at least 2 nodes.
        return np.array([[1.0]]) if n_items == 1 else np.array([[]])

    # Use absolute similarity for calculations, as wTO assumes weights >= 0
    adj_matrix = np.abs(similarity_matrix.copy())
    np.fill_diagonal(adj_matrix, 0) # Ignore self-similarity in sums

    # Calculate L_ij (shared neighbor strength matrix)
    # This is equivalent to the matrix multiplication A * A
    L = adj_matrix @ adj_matrix

    # Calculate k_i (node strength vector)
    k = np.sum(adj_matrix, axis=1) # Sum across rows

    # Prepare matrices for vectorized calculation
    k_i = k[:, np.newaxis] # Reshape k to column vector for broadcasting
    k_j = k[np.newaxis, :] # Reshape k to row vector for broadcasting

    # Calculate the denominator: min(k_i, k_j) + 1 - a_ij
    min_k = np.minimum(k_i, k_j)
    denominator = min_k + 1 - adj_matrix

    # Calculate the numerator: L_ij + a_ij
    numerator = L + adj_matrix

    # --- Calculate wTO ---
    # Initialize wTO matrix
    wto_matrix = np.zeros_like(adj_matrix)

    # Avoid division by zero where denominator is close to zero
    # This typically happens for pairs of disconnected nodes (a_ij=0, min_k=0)
    # In such cases, wTO should be 0.
    valid_denominator = denominator > 1e-12 # Use a small epsilon
    wto_matrix[valid_denominator] = numerator[valid_denominator] / denominator[valid_denominator]

    # Ensure diagonal is 1.0
    np.fill_diagonal(wto_matrix, 1.0)

    # Clip values to [0, 1] just in case of floating point inaccuracies
    wto_matrix = np.clip(wto_matrix, 0.0, 1.0)

    return wto_matrix

def construct_tmfg_network(
    similarity_matrix: np.ndarray,
    item_labels: list[str] | None = None,
) -> nx.Graph:
    """Constructs a Triangulated Maximally Filtered Graph (TMFG).

    Based on the algorithm described in Massara et al. (2016). The TMFG
    is a planar graph retaining significant connections from the similarity matrix.
    It contains 3*(n-2) edges for n > 2 nodes.

    Args:
        similarity_matrix: A square (n_items, n_items) NumPy array of pairwise
            similarities (e.g., cosine similarity or correlation). Higher values
            indicate stronger relationships. Diagonal elements are ignored.
        item_labels: Optional list of strings representing the names of the items (nodes).
                     If provided, node labels in the graph will be set accordingly.
                     Length must match the dimension of the similarity matrix.

    Returns:
        A networkx.Graph object representing the TMFG.
        Nodes are indexed 0 to n-1, or labeled with item_labels if provided.
        Edges have a 'weight' attribute corresponding to the similarity value.

    Raises:
        ValueError: If the similarity matrix is not square, not 2D, has less than 3 nodes,
                    or if item_labels length doesn't match the matrix dimension.
        TypeError: If similarity_matrix is not a NumPy array.

    References:
        Massara, G. P., Di Matteo, T., & Aste, T. (2016). Network filtering for
        big data: Triangulated Maximally Filtered Graph. Journal of Complex Networks,
        5(2), 161-178. https://doi.org/10.1093/comnet/cnw015
    """
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("Input similarity_matrix must be a NumPy array.")
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Input similarity_matrix must be a square 2D array.")

    n_items = similarity_matrix.shape[0]
    if n_items < 3:
        # TMFG requires at least 3 nodes to form the initial triangle.
        # Return an empty graph or handle as appropriate for n=0, 1, 2.
        # For simplicity, we raise an error here as per typical implementation needs.
        raise ValueError(f"TMFG requires at least 3 nodes, but got {n_items}.")

    if item_labels is not None:
        if len(item_labels) != n_items:
            raise ValueError("Length of item_labels must match the dimension of the similarity matrix.")
        node_ids = item_labels
    else:
        node_ids = list(range(n_items))

    # --- TMFG Algorithm Implementation --- #

    # Use absolute values for edge weights if similarity can be negative (e.g., correlation)
    # For cosine similarity (>=0 usually), this might not be strictly needed but doesn't hurt.
    # The algorithm prioritizes higher similarity/correlation.
    weights = np.abs(similarity_matrix.copy())
    np.fill_diagonal(weights, -np.inf) # Ignore self-loops

    G = nx.Graph()
    nodes_in_graph = set()
    edges_in_graph = set()

    # 1. Find the initial triangle (nodes with highest combined similarity)
    initial_node_indices = np.argsort(np.sum(weights, axis=1))[-3:]
    for i in range(3):
        u = initial_node_indices[i]
        nodes_in_graph.add(u)
        G.add_node(node_ids[u])
        for j in range(i + 1, 3):
            v = initial_node_indices[j]
            weight = similarity_matrix[u, v] # Use original similarity for weight
            G.add_edge(node_ids[u], node_ids[v], weight=weight)
            edges_in_graph.add(tuple(sorted((u, v))))

    # 2. Iteratively add remaining nodes
    nodes_to_add = [i for i in range(n_items) if i not in nodes_in_graph]

    for node_k in nodes_to_add:
        potential_neighbors = list(nodes_in_graph)
        # Find the 3 best neighbors for node_k among nodes already in the graph
        neighbor_similarities = weights[node_k, potential_neighbors]
        best_neighbor_indices_in_list = np.argsort(neighbor_similarities)[-3:]
        best_neighbors = [potential_neighbors[i] for i in best_neighbor_indices_in_list]

        # Add node_k and its 3 edges to the best neighbors
        nodes_in_graph.add(node_k)
        G.add_node(node_ids[node_k])
        for neighbor_node in best_neighbors:
            # Check if edge already exists (shouldn't if logic is correct, but safe)
            edge = tuple(sorted((node_k, neighbor_node)))
            if edge not in edges_in_graph:
                weight = similarity_matrix[node_k, neighbor_node]
                G.add_edge(node_ids[node_k], node_ids[neighbor_node], weight=weight)
                edges_in_graph.add(edge)

    # --- Verification (Optional but recommended) --- #
    expected_edges = 3 * (n_items - 2)
    if G.number_of_edges() != expected_edges:
        # This might indicate an issue with the implementation or input matrix properties
        print(
             f"Warning: TMFG expected {expected_edges} edges, but graph has {G.number_of_edges()}. "
             "This might occur with degenerate similarity matrices."
        )
        # Depending on strictness, could raise an error here.

    # Check planarity using networkx built-in check (can be computationally expensive for large graphs)
    # try:
    #     is_planar, _ = nx.check_planarity(G, counterexample=False)
    #     if not is_planar:
    #         print("Warning: Constructed TMFG graph is not planar according to nx.check_planarity!")
    # except nx.NetworkXException as e:
    #     print(f"Planarity check failed: {e}")

    return G

def construct_ebicglasso_network(
    similarity_matrix: np.ndarray,
    item_labels: list[str] | None = None,
    assume_centered: bool = False,
    cv: int = 5, # Number of cross-validation folds
    max_iter: int = 100,
    force_similarity: bool = False,
) -> nx.Graph:
    """Constructs a network using Graphical LASSO with Cross-Validation (EBICglasso variant).

    This function estimates a sparse precision matrix (inverse covariance) from the
    input similarity matrix using GraphicalLassoCV. The non-zero entries in the
    precision matrix correspond to edges in the resulting Gaussian Graphical Model (GGM).
    Note: This uses standard CV, not strictly EBIC, but serves a similar purpose of
    selecting the sparsity parameter (alpha).

    Args:
        similarity_matrix: A square (n_items, n_items) NumPy array. This is treated
            as if it were a covariance matrix for the purpose of GraphicalLassoCV.
            It should ideally be positive semi-definite.
        item_labels: Optional list of strings for node labels. Length must match matrix dim.
        assume_centered: If True, data is not centered before computation.
                         Set to True if the input matrix is already centered or is
                         a similarity/correlation matrix.
        cv: Number of cross-validation folds used by GraphicalLassoCV.
        max_iter: Maximum number of iterations for GraphicalLassoCV.
        force_similarity: If True, allows using similarity matrices (e.g., cosine similarity)
                         instead of covariance matrices. If False, raises an error if the
                         matrix appears to be a similarity matrix (diagonal ≈ 1).

    Returns:
        A networkx.Graph object representing the GGM.
        Nodes are labeled 0..n-1 or with item_labels.
        Edges correspond to non-zero entries in the estimated precision matrix.
        Edge weights represent the partial correlation derived from the precision matrix.

    Raises:
        ValueError: If inputs are invalid (matrix not square, label mismatch, etc.) or
                   if matrix appears to be similarity-based and force_similarity=False.
        TypeError: If similarity_matrix is not a NumPy array.
        ImportError: If scikit-learn is not installed.

    Notes:
        - GraphicalLassoCV expects a covariance matrix. Using a similarity matrix
          (like cosine similarity) is an approximation when raw scores are unavailable –
          interpret partial correlations cautiously.
        - ConvergenceWarning may occur if max_iter is too low.
    """
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("Input similarity_matrix must be a NumPy array.")
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Input similarity_matrix must be a square 2D array.")

    n_items = similarity_matrix.shape[0]
    if n_items < 2:
        # Graphical Lasso needs at least 2 variables
        raise ValueError(f"Graphical LASSO requires at least 2 nodes, but got {n_items}.")

    if item_labels is not None:
        if len(item_labels) != n_items:
            raise ValueError("Length of item_labels must match the dimension of the similarity matrix.")
        node_ids = item_labels
    else:
        node_ids = list(range(n_items))

    # Check if matrix appears to be similarity-based (diagonal close to 1)
    diagonal_values = np.diag(similarity_matrix)
    is_similarity_like = np.allclose(diagonal_values, 1.0, atol=0.1)
    
    if is_similarity_like and not force_similarity:
        raise ValueError(
            "Input matrix appears to be similarity-based (diagonal ≈ 1) rather than a covariance matrix. "
            "GraphicalLasso expects covariance matrices. If you intend to use similarity matrices, "
            "set force_similarity=True, but interpret partial correlations cautiously."
        )
    
    if is_similarity_like and force_similarity:
        warnings.warn(
            "Using similarity matrix with GraphicalLasso is an approximation when raw scores are "
            "unavailable – interpret partial correlations cautiously. Consider using TMFG if you have "
            "concerns about this approximation.",
            UserWarning
        )

    # 1) Shrink covariance
    try:
        shrunk_sim_matrix = shrunk_covariance(similarity_matrix)
    except Exception as e:
        # fallback to using the raw similarity if shrinkage itself fails
        warnings.warn(f"Covariance shrinkage failed ({e}), using raw similarity matrix for Glasso.")
        shrunk_sim_matrix = similarity_matrix.copy()

    # 2) Calculate jitter amount
    std_dev = np.std(shrunk_sim_matrix[np.triu_indices_from(shrunk_sim_matrix, k=1)]) # Std of off-diagonal elements
    epsilon = 1e-4 * std_dev if std_dev > 1e-8 else 1e-6 # Avoid zero/tiny epsilon

    # wrap fit in a retry-on-FP-error loop, bolstering jitter if needed
    # Handle potential convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            # Apply initial jitter
            jittered_matrix = shrunk_sim_matrix + np.eye(n_items) * epsilon
            model = GraphicalLassoCV(
                cv=cv,
                assume_centered=assume_centered,
                max_iter=max_iter,
                eps=1e-8,  # Keep internal solver regularization
                n_jobs=-1 # Use all available CPU cores
                )
            model.fit(jittered_matrix)
        except FloatingPointError:
            # try again with a larger diagonal penalty, computed from original matrix
            epsilon *= 10
            # Create a *new* matrix with the increased jitter from the original shrunk matrix
            jittered_matrix_boosted = shrunk_sim_matrix + np.eye(n_items) * epsilon
            try:
                model.fit(jittered_matrix_boosted)
            except FloatingPointError as e2:
                raise RuntimeError("Even after shrinkage and jittering, GraphicalLasso failed to converge. " +
                                   "The similarity matrix might be too ill-conditioned. Consider using TMFG, " +
                                   "collecting more varied data, or increasing regularization further.") from e2
        except ImportError:
            raise ImportError("Scikit-learn library not found. Please install it (`pip install scikit-learn`).")
        except Exception as e:
            # Catch other potential errors during fitting
            raise RuntimeError(f"Error fitting GraphicalLassoCV: {e}") from e

    # --- Build graph from the estimated precision matrix --- #
    precision_matrix = model.precision_
    G = nx.Graph()

    for i in range(n_items):
        G.add_node(node_ids[i]) # Add nodes first

    # Calculate partial correlations from precision matrix
    # pcorr(i,j) = -prec(i,j) / sqrt(prec(i,i) * prec(j,j))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if abs(precision_matrix[i, j]) > 1e-8: # Check if non-zero (handle precision)
                denom = np.sqrt(precision_matrix[i, i] * precision_matrix[j, j])
                if denom > 1e-8:
                    partial_corr = -precision_matrix[i, j] / denom
                    # Clip partial correlation to [-1, 1] due to potential float issues
                    partial_corr = np.clip(partial_corr, -1.0, 1.0)
                    G.add_edge(node_ids[i], node_ids[j], weight=partial_corr)
                # else: handle potential division by zero if diagonal precision is near zero

    return G

def detect_communities_walktrap(
    graph: nx.Graph,
    weights: str | None = "weight",
    steps: int = 4,
) -> tuple[dict[str | int, int], ig.VertexClustering | None]:
    """Detects communities using the Walktrap algorithm from igraph.

    Handles potential isolated nodes (zero strength) by running Walktrap
    on the largest connected component with positive strength and assigning
    isolated nodes to community -1.

    Args:
        graph: The networkx Graph to analyze. Assumes nodes are labeled
               with strings or integers as used in networkx.
        weights: The edge attribute key for weights. If None, the graph is
                 treated as unweighted. Defaults to "weight".
        steps: The length of the random walks for the Walktrap algorithm.
               Defaults to 4.

    Returns:
        A tuple containing:
        - membership (dict[str | int, int]): A dictionary mapping original
          node IDs (from networkx graph) to community IDs (integers).
          Isolated nodes are assigned community ID -1.
        - clustering (igraph.VertexClustering | None): The VertexClustering
          object returned by igraph for the component analyzed. Returns None
          if the graph has no nodes with positive strength or if igraph fails.

    Raises:
        ImportError: If the 'igraph' library is not installed.
        ValueError: If the input graph is empty or weights are invalid.
        RuntimeError: If Walktrap fails for unexpected reasons.
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError(
            "Community detection requires the 'python-igraph' library. "
            "Please install it (`pip install python-igraph`)."
        )

    if graph.number_of_nodes() == 0:
        # Return empty results for an empty graph
        return {}, None

    # 1. Convert networkx graph to igraph graph
    # Ensure node names are preserved if they are strings
    try:
        # Create mapping from nx node ID to igraph vertex ID (0 to n-1)
        nx_nodes = list(graph.nodes())
        node_map_nx_to_ig = {node: i for i, node in enumerate(nx_nodes)}
        node_map_ig_to_nx = {i: node for node, i in node_map_nx_to_ig.items()}

        # Create igraph graph structure
        igraph_graph = ig.Graph(
            n=graph.number_of_nodes(), directed=False
        )

        # Add edges with weights if specified
        if weights:
            edge_list = []
            weight_list = []
            for u, v, data in graph.edges(data=True):
                weight_val = data.get(weights)
                if weight_val is None:
                    raise ValueError(f"Edge ({u}, {v}) is missing the specified weight attribute '{weights}'.")
                # Ensure weights are positive for standard Walktrap strength calculation
                # Walktrap itself can handle non-negative, but strength=0 causes issues.
                # We handle strength=0 separately below.
                if weight_val < 0:
                     print(f"Warning: Edge ({u}, {v}) has negative weight {weight_val}. Using absolute value for Walktrap.")
                     weight_val = abs(weight_val) # Or handle as error depending on use case
                edge_list.append((node_map_nx_to_ig[u], node_map_nx_to_ig[v]))
                weight_list.append(weight_val)
            igraph_graph.add_edges(edge_list)
            igraph_graph.es['weight'] = weight_list

        else:
            # Add edges without weights
            igraph_graph.add_edges(
                [(node_map_nx_to_ig[u], node_map_nx_to_ig[v]) for u, v in graph.edges()]
            )
            # Walktrap still needs non-zero strength, implicit weight is 1.

    except Exception as e:
        raise RuntimeError(f"Failed to convert networkx graph to igraph graph: {e}") from e

    # 2. Identify and separate zero-strength nodes
    try:
        strengths = igraph_graph.strength(weights='weight' if weights else None)
        positive_strength_vertices = [i for i, s in enumerate(strengths) if s > 0]
        zero_strength_vertices = [i for i, s in enumerate(strengths) if s <= 0] # Includes isolated nodes

        # Initialize membership dict with isolated nodes assigned to community -1
        membership = {node_map_ig_to_nx[v_idx]: -1 for v_idx in zero_strength_vertices}

        if not positive_strength_vertices:
            # All nodes are isolated or form components with 0-weight edges only
            # Ensure all nodes are marked as isolated
            for nx_node in nx_nodes:
                membership[nx_node] = -1
            return membership, None

        # Create subgraph with only positive-strength vertices
        # Use the more compatible method without implementation parameter
        try:
            subgraph = igraph_graph.induced_subgraph(positive_strength_vertices)
        except TypeError:
            # Fallback for older igraph versions
            subgraph = igraph_graph.subgraph(positive_strength_vertices)

    except Exception as e:
         raise RuntimeError(f"Failed during node strength calculation or subgraph creation: {e}") from e


    # 3. Run Walktrap on the subgraph
    try:
        # Ensure the subgraph still has weights if the original graph did
        subgraph_weights = subgraph.es['weight'] if weights and 'weight' in subgraph.es.attributes() else None

        # Check if the subgraph has edges before running Walktrap
        if subgraph.ecount() == 0:
            # Treat all nodes in the subgraph as a single community
            clustering = None # No meaningful clustering object
            subgraph_membership = {v.index: 0 for v in subgraph.vs} # Assign community 0
        else:
            # Run Walktrap
            clustering = subgraph.community_walktrap(
                weights=subgraph_weights, steps=steps
            ).as_clustering()
            subgraph_membership = {v.index: mem for v, mem in zip(subgraph.vs, clustering.membership)}

        # Map subgraph vertex indices back to original igraph vertex indices
        # Then map original igraph indices back to original networkx node IDs
        original_indices_membership = {
            positive_strength_vertices[sub_idx]: mem
            for sub_idx, mem in subgraph_membership.items()
        }

        # Add results from subgraph to the main membership dictionary
        for ig_idx, comm_id in original_indices_membership.items():
            membership[node_map_ig_to_nx[ig_idx]] = comm_id

        return membership, clustering

    except Exception as e:
        raise RuntimeError(f"Failed during Walktrap community detection: {e}") from e


def calculate_tefi(
    similarity_matrix: np.ndarray, 
    membership: dict[str | int, int],
    *,
    item_order: list[str] | None = None,
) -> float:
    """Calculates the Total Entropy Fit Index (TEFI) variant.

    This variant calculates the standardized difference between the average
    within-community similarity and the average between-community similarity.
    Higher values indicate better fit, meaning communities group items that are
    more similar to each other than to items in other communities.

    Args:
        similarity_matrix: The original (n_items, n_items) similarity matrix.
        membership: A dictionary mapping node IDs to community IDs. 
                    Assumes community IDs are integers, potentially including -1
                    for isolated nodes which are ignored in this calculation.
        item_order: Explicit list defining the order of items corresponding to 
                   matrix rows/columns. If None, uses dict insertion order with warning.

    Returns:
        The TEFI score (float). Returns np.nan if calculation is not possible
        (e.g., no within/between edges found, requires >1 community).

    Raises:
        ValueError: If the dimensions of the matrix and membership seem inconsistent
                    or if node IDs in membership don't correspond to matrix indices.
    """
    n_items = similarity_matrix.shape[0]
    if n_items == 0:
        return np.nan

    # Get node list with explicit ordering or fallback to dict keys
    if item_order is not None:
        node_list = item_order
        # Validate that all items in item_order are in membership
        missing_items = [item for item in item_order if item not in membership]
        if missing_items:
            raise ValueError(f"Items in item_order not found in membership: {missing_items}")
    else:
        node_list = list(membership.keys())
        warnings.warn(
            "TEFI is inferring item order from dict insertion order; "
            "pass `item_order` explicitly to avoid mismatches.",
            RuntimeWarning,
        )

    if len(node_list) != n_items:
        raise ValueError(f"Mismatch between number of items in membership ({len(node_list)}) "
                         f"and similarity matrix dimension ({n_items}).")

    # Get unique, valid community IDs (exclude -1 for isolated nodes)
    valid_communities = {cid for cid in membership.values() if cid != -1}
    if len(valid_communities) < 2:
        # Need at least two communities to compare within vs. between
        # Or if all nodes are isolated (-1)
        print("Warning: TEFI requires at least 2 valid communities. Returning NaN.")
        return np.nan

    within_community_similarities = []
    between_community_similarities = []

    # Iterate through unique pairs of nodes (upper triangle of similarity matrix)
    for i_idx in range(n_items):
        node_i = node_list[i_idx]
        comm_i = membership.get(node_i)
        if comm_i == -1: # Skip isolated nodes
            continue

        for j_idx in range(i_idx + 1, n_items):
            node_j = node_list[j_idx]
            comm_j = membership.get(node_j)
            if comm_j == -1: # Skip isolated nodes
                continue

            similarity = similarity_matrix[i_idx, j_idx]

            if comm_i == comm_j:
                within_community_similarities.append(similarity)
            else:
                between_community_similarities.append(similarity)

    # Calculate averages
    avg_within = np.mean(within_community_similarities) if within_community_similarities else np.nan
    avg_between = np.mean(between_community_similarities) if between_community_similarities else np.nan

    if np.isnan(avg_within) or np.isnan(avg_between):
        print("Warning: Could not calculate average within or between similarities (perhaps only one community?). Returning NaN for TEFI.")
        return np.nan

    # Calculate global standard deviation of off-diagonal similarities used in calculation
    all_relevant_similarities = within_community_similarities + between_community_similarities
    if not all_relevant_similarities or len(all_relevant_similarities) < 2:
         print("Warning: Not enough similarity values to calculate standard deviation. Returning NaN for TEFI.")
         return np.nan

    global_std_dev = np.std(all_relevant_similarities)

    if global_std_dev < 1e-9: # Avoid division by zero or near-zero std dev
        print("Warning: Global standard deviation of similarities is near zero. Returning NaN for TEFI.")
        return np.nan

    tefi_score = (avg_within - avg_between) / global_std_dev

    return tefi_score


# --- Placeholder for NMI --- #
def calculate_nmi(membership1: dict[str | int, int], membership2: dict[str | int, int]) -> float:
    """Calculates the Normalized Mutual Information (NMI) between two clustering solutions.

    Compares how much information is shared between two different community
    assignments of the same set of nodes. NMI values range from 0 (completely
    different) to 1 (identical clustering).

    Args:
        membership1: First clustering solution mapping node IDs to community IDs.
        membership2: Second clustering solution mapping node IDs to community IDs.
                     Must have the same node IDs as membership1.

    Returns:
        NMI score as a float between 0 and 1. Returns np.nan if calculation fails
        or if memberships are incompatible.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If memberships have different node sets.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        raise ImportError("NMI calculation requires scikit-learn. Please install it.")

    # Ensure both memberships have the same nodes
    nodes1 = set(membership1.keys())
    nodes2 = set(membership2.keys())

    if nodes1 != nodes2:
        raise ValueError(f"Memberships have different node sets: {nodes1 - nodes2} vs {nodes2 - nodes1}")

    # Get the common nodes and ensure consistent ordering
    common_nodes = sorted(list(nodes1))  # Use sorted for deterministic ordering

    # Extract labels in the same order
    labels1 = [membership1[node] for node in common_nodes]
    labels2 = [membership2[node] for node in common_nodes]

    if len(labels1) == 0:
        return np.nan

    # Calculate NMI
    try:
        nmi_score = normalized_mutual_info_score(labels1, labels2)
        return float(nmi_score)
    except Exception as e:
        import traceback
        return np.nan


def remove_redundant_items_uva(
    graph: nx.Graph,
    item_labels: list[str | int], # Should match nodes in the graph
    wto_threshold: float = 0.20,
    graph_type: str = "glasso", # 'glasso' or 'tmfg' to determine adjacency matrix logic
) -> tuple[list[str | int], list[tuple[str | int, float]]]:
    """Iteratively removes redundant items based on Weighted Topological Overlap (wTO).

    Calculates wTO based on the structure of the provided graph (TMFG or EBICglasso)
    and iteratively removes the item with the highest wTO above the threshold.
    Tie-breaking is done using the sum of absolute connection strengths in the
    current subgraph.

    Args:
        graph: The networkx.Graph object (TMFG or EBICglasso) containing the
            items as nodes and relationships as weighted edges.
        item_labels: The list of item labels corresponding to the nodes currently
            in the graph. Must match the order/nodes initially.
        wto_threshold: The cutoff value for wTO. Items with max wTO >= this
            threshold are considered for removal.
        graph_type: Specifies how to derive the adjacency matrix for wTO calculation.
            'glasso': Uses absolute partial correlations from edge weights.
            'tmfg': Uses the existing edge weights (assumed non-negative or abs).

    Returns:
        A tuple containing:
        - remaining_items: A list of item labels that were not removed.
        - removed_items_log: A list of tuples, where each tuple contains the
            removed item label and its maximum wTO score at the time of removal.

    Raises:
        ValueError: If graph is empty, item_labels mismatch, threshold is invalid,
                    or graph_type is unknown.
        TypeError: If graph is not a networkx.Graph.
        KeyError: If edge weights are missing when expected.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input 'graph' must be a networkx.Graph object.")
    if not item_labels:
        raise ValueError("Input 'item_labels' cannot be empty.")
    if not 0 <= wto_threshold <= 1:
        raise ValueError("wto_threshold must be between 0 and 1.")
    if graph_type not in ["glasso", "tmfg"]:
        raise ValueError("graph_type must be either 'glasso' or 'tmfg'.")
    if set(item_labels) != set(graph.nodes()):
         # Check if the labels exactly match the current nodes in the graph
        raise ValueError("item_labels must correspond exactly to the nodes in the input graph.")

    # Keep track of the current graph and items
    current_graph = graph.copy() # Work on a copy
    current_items = list(item_labels) # Ensure it's a mutable list copy
    removed_items_log = []

    while len(current_items) > 2: # Need at least 3 items to calculate meaningful wTO in a graph context
        # 1. Get adjacency matrix from the current subgraph
        subgraph_nodes = current_items
        subgraph = current_graph.subgraph(subgraph_nodes)

        # Create adjacency matrix based on graph type
        try:
            if graph_type == 'glasso':
                # Use absolute partial correlation from edge weights
                adj_matrix = nx.to_numpy_array(subgraph, nodelist=subgraph_nodes, weight='weight')
                adj_matrix = np.abs(adj_matrix) # Ensure positive weights for wTO
            elif graph_type == 'tmfg':
                # Use existing edge weights (TMFG weights are typically positive similarity)
                adj_matrix = nx.to_numpy_array(subgraph, nodelist=subgraph_nodes, weight='weight')
                # If TMFG weights could be negative (e.g., raw correlations), take abs
                adj_matrix = np.abs(adj_matrix) # Ensure positive weights for wTO
            else:
                 # This case is already handled by the initial check, but added for safety
                 raise ValueError(f"Unknown graph_type: {graph_type}")
        except KeyError as e:
            raise KeyError(f"Missing 'weight' attribute on edges for graph_type '{graph_type}'. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error converting subgraph to numpy array: {e}")

        # 2. Calculate wTO matrix for the current subgraph
        # Ensure the diagonal is 0 before passing to calculate_wto
        np.fill_diagonal(adj_matrix, 0)
        # Pass the adjacency matrix, not the similarity matrix, to wTO calculation
        wto_matrix_sub = calculate_wto(adj_matrix) # calculate_wto handles its own diag setting

        # 3. Find the item with the maximum wTO score
        # Set diagonal to 0 to ignore self-wTO
        np.fill_diagonal(wto_matrix_sub, 0)
        max_wto_scores = np.max(wto_matrix_sub, axis=1)
        max_wto_overall = np.max(max_wto_scores)
        max_wto_idx = np.argmax(max_wto_scores) # Index within the current subgraph_nodes list


        # 4. Check against threshold
        if max_wto_overall < wto_threshold:
            break # No more items are redundant enough

        # 5. Tie-breaking: If multiple items have the same max wTO, remove the one
        #    with the *lowest* sum of connection strengths in the *current subgraph*.
        #    (Lower strength means less central/important to the current structure).
        candidate_indices = np.where(np.isclose(max_wto_scores, max_wto_overall))[0]

        if len(candidate_indices) > 1:
            # Calculate sum of absolute connection strengths (node degree in weighted graph)
            # Use the adj_matrix derived earlier for the current subgraph
            node_strengths = np.sum(adj_matrix, axis=1) # Sum weights for each node
            candidate_strengths = node_strengths[candidate_indices]
            # Find the index (among candidates) with the minimum strength
            min_strength_candidate_idx = candidate_indices[np.argmin(candidate_strengths)]
            item_to_remove_idx = min_strength_candidate_idx
        else:
            item_to_remove_idx = max_wto_idx # Only one candidate

        # 6. Remove the item
        item_to_remove = subgraph_nodes[item_to_remove_idx]
        removed_items_log.append((item_to_remove, max_wto_overall))

        # Update the list of current items and the graph for the next iteration
        current_items.remove(item_to_remove)
        current_graph.remove_node(item_to_remove) # Modify the copied graph

    return current_items, removed_items_log


# ============================================================
# Top-level helper function for bootEGA parallel processing
# ============================================================
def _run_bootstrap_single(
    args: tuple[int, list[str], np.ndarray | csr_matrix, str, dict, dict, int | None]
) -> tuple[list[str], dict[str, int] | None]:
    """Runs a single bootstrap iteration for item stability analysis.

    This function performs item-level bootstrap resampling (sampling items with replacement)
    rather than traditional case/participant-level resampling. This approach is specifically
    designed for analyzing the stability of item community assignments in psychometric networks.

    Designed to be called by ProcessPoolExecutor.map for parallel execution.

    Args:
        args: A tuple containing:
            iteration_seed (int): Seed for this specific bootstrap iteration.
            original_items (list[str]): The full list of items being analyzed.
            original_embeddings (np.ndarray | csr_matrix): The embedding matrix corresponding to original_items.
            network_method (str): 'tmfg' or 'glasso'.
            network_params (dict): Parameters for network construction.
            walktrap_params (dict): Parameters for Walktrap community detection.
            sample_size (int | None): Size of the bootstrap sample. If None, uses len(original_items).

    Returns:
        tuple[list[str], dict[str, int] | None]: A tuple containing:
            - sampled_item_list (list[str]): The *unique* items included in the bootstrap sample.
            - community_membership (dict[str, int] | None): Dictionary mapping unique item labels
              in the sample to their detected community ID, or None if EGA failed.
              
    Note:
        The resampling draws items with replacement, then reduces to unique items for analysis.
        This ensures that the same item can appear multiple times in the raw sample but is
        analyzed only once in the final network construction.
    """
    (
        iteration_seed,
        original_items,
        original_embeddings,
        network_method,
        network_params,
        walktrap_params,
        sample_size,
    ) = args

    # Set random seed for this process/iteration for reproducibility
    np.random.seed(iteration_seed)
    random.seed(iteration_seed)

    if sample_size is None:
        sample_size = len(original_items)

    # 1. Sample indices with replacement
    sampled_indices = np.random.choice(len(original_items), size=sample_size, replace=True)
    unique_sampled_indices = sorted(list(set(sampled_indices)))

    # If only one unique item is sampled, EGA cannot run
    if len(unique_sampled_indices) < 2:
        # Need at least 2 for similarity, 3 for TMFG maybe?
        return [], None # Return empty list and None communities

    unique_sampled_items = [original_items[i] for i in unique_sampled_indices]
    unique_sampled_embeddings = original_embeddings[unique_sampled_indices, :]


    # 2. Re-run the EGA pipeline on the sampled data
    try:
        # a. Calculate similarity on the *unique* sampled embeddings
        # Need to handle potential errors if matrix is not suitable (e.g., all zeros)
        sampled_similarity = calculate_similarity_matrix(unique_sampled_embeddings)
        if np.all(np.isnan(sampled_similarity)) or sampled_similarity.shape[0] < 2:
             # print(f"[Bootstrap {iteration_seed}] Invalid similarity matrix for sample.")
             return unique_sampled_items, None

        # b. Construct network
        sampled_graph = None
        if network_method.lower() == 'tmfg':
            # TMFG needs at least 3 nodes
            if len(unique_sampled_items) < 3:
                 return unique_sampled_items, None
            sampled_graph = construct_tmfg_network(sampled_similarity, item_labels=unique_sampled_items, **network_params)
        elif network_method.lower() == 'glasso':
            sampled_graph = construct_ebicglasso_network(sampled_similarity, item_labels=unique_sampled_items, **network_params)
        else:
            raise ValueError(f"Unknown network method: {network_method}")

        if sampled_graph is None or sampled_graph.number_of_nodes() == 0:
            # print(f"[Bootstrap {iteration_seed}] Failed to construct network graph for sample.")
            return unique_sampled_items, None

        # c. Detect communities
        # Handle potential errors during community detection
        community_membership, _ = detect_communities_walktrap(
            sampled_graph,
            **walktrap_params
        )

        # Return the list of unique items in the sample and their communities
        # The keys in community_membership should already be the correct item labels
        return unique_sampled_items, community_membership

    except Exception as e:
        # Log the exception for debugging, return None for this bootstrap sample
        # print(f"Error in bootstrap sample {idx}: {e}")
        # traceback.print_exc() # More detailed traceback
        return unique_sampled_items, None # Return item IDs even on failure, but None for communities


# ============================================================
# bootEGA Resampling Function (Modified)
# ============================================================
def run_bootega_resampling(
    items: list[str],
    embeddings: np.ndarray | csr_matrix,
    network_method: str, # 'tmfg' or 'glasso'
    network_params: dict,
    walktrap_params: dict,
    n_bootstrap: int = 100,
    sample_size: int | None = None,
    use_parallel: bool = True,
    max_workers: int | None = None,
    progress_callback: Callable | None = None
) -> list[tuple[list[str], dict[str, int] | None]]:
    """Performs N bootstrap resampling iterations for item stability analysis.
    
    This function implements ITEM-LEVEL bootstrap resampling (not case/participant-level)
    where each bootstrap sample consists of items drawn with replacement from the original
    item pool. This is appropriate for analyzing the stability of item community assignments
    in psychometric networks.

    Note: This differs from traditional EGA bootstrap that resamples participants/cases.
    The current approach is designed specifically for item stability analysis in the
    context of instrument development and item reduction.

    Args:
        items: List of item labels to resample from.
        embeddings: Embedding matrix where each row corresponds to an item in `items`.
        network_method: Either 'tmfg' or 'glasso' for network construction.
        network_params: Dictionary of parameters passed to network construction function.
        walktrap_params: Dictionary of parameters passed to Walktrap community detection.
        n_bootstrap: Number of bootstrap iterations to perform.
        sample_size: Size of each bootstrap sample. If None, uses len(items).
        use_parallel: Whether to use multiprocessing for parallel execution.
        max_workers: Maximum number of worker processes. If None, uses system default.
        progress_callback: Optional function called with (percentage, message) for progress updates.

    Returns:
        List of tuples, each containing:
        - sampled_items (list[str]): Unique items in the bootstrap sample
        - community_membership (dict[str, int] | None): Community assignments or None if EGA failed

    Example:
        >>> # Resample items for stability analysis
        >>> results = run_bootega_resampling(
        ...     items=['Item_1', 'Item_2', 'Item_3'],
        ...     embeddings=item_embeddings,
        ...     network_method='glasso',
        ...     network_params={'force_similarity': True},
        ...     walktrap_params={'steps': 4},
        ...     n_bootstrap=100
        ... )
    """
    bootstrap_results = []

    if progress_callback:
        progress_callback(0, "Starting bootEGA resampling...")

    start_time = time.time()

    # Prepare arguments for mapping
    master_seed = random.randint(0, 2**32 - 1)
    np.random.seed(master_seed)
    iteration_seeds = np.random.randint(0, 2**32 - 1, size=n_bootstrap)

    tasks_args = [
        (
            int(iteration_seeds[i]),
            items,
            embeddings,
            network_method,
            network_params,
            walktrap_params,
            sample_size,
        )
        for i in range(n_bootstrap)
    ]

    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results_iterator = executor.map(_run_bootstrap_single, tasks_args)
                for i, result in enumerate(results_iterator):
                    bootstrap_results.append(result)
                    if progress_callback and (i + 1) % max(1, n_bootstrap // 20) == 0:
                        percentage = (i + 1) / n_bootstrap * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / (i + 1)) * (n_bootstrap - (i + 1)) if (i + 1) > 0 else 0
                        progress_callback(percentage, f"Resampling {i+1}/{n_bootstrap}... (ETA: {eta:.0f}s)")

        except Exception as e:
            print(f"Error during parallel execution: {e}")
            # print(traceback.format_exc()) # Optional: Print full traceback for debugging
            print("Falling back to sequential execution.")
            use_parallel = False
            bootstrap_results = [] # Reset results for sequential run

    if not use_parallel:
        print("Running bootEGA resampling sequentially...") # Add indication
        for i, args in enumerate(tasks_args):
            result = _run_bootstrap_single(args)
            bootstrap_results.append(result)
            if progress_callback and (i + 1) % max(1, n_bootstrap // 10) == 0:
                percentage = (i + 1) / n_bootstrap * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (n_bootstrap - (i + 1)) if (i + 1) > 0 else 0
                progress_callback(percentage, f"Resampling {i+1}/{n_bootstrap}... (ETA: {eta:.0f}s)")

    if progress_callback:
        progress_callback(100, f"Resampling completed ({time.time() - start_time:.1f}s).")

    return bootstrap_results


def perform_bootega_stability_analysis(
    initial_items: list[str],
    original_embeddings: np.ndarray | sp.spmatrix,
    original_community_membership: dict[str, int],
    network_method: str,
    network_params: dict,
    walktrap_params: dict,
    stability_threshold: float = 0.75,
    n_bootstrap: int = 100,
    sample_size: int | None = None,
    use_parallel: bool = True,
    max_workers: int | None = None,
    progress_callback: Callable | None = None,
    verbose: bool = False # Added for potential logging
) -> tuple[list[str], dict[str, float], list[tuple[str, float, int]]]:
    """Performs iterative bootEGA stability analysis.

    Removes items that consistently fail to cluster with their original community
    across bootstrap samples.

    Args:
        initial_items: List of item labels (e.g., after UVA) to start with.
        original_embeddings: Embeddings matrix for the *initial_items*.
        original_community_membership: Community assignments for *initial_items*
                                        (from the initial EGA run before bootEGA).
        network_method: 'tmfg' or 'glasso'.
        network_params: Parameters for network construction.
        walktrap_params: Parameters for Walktrap.
        stability_threshold: Proportion of times an item must be in its original
                             community to be considered stable (0.0 to 1.0).
        n_bootstrap: Number of bootstrap samples per iteration.
        sample_size: Size of each bootstrap sample. Defaults to len(current_items).
        use_parallel: Whether to use multiprocessing for resampling.
        max_workers: Max workers for parallel execution.
        progress_callback: Function to report progress (percentage, message).
        verbose: If True, print more detailed logs.

    Returns:
        tuple containing:
        - stable_items (list[str]): The final list of items meeting the stability threshold.
        - final_stability_scores (dict[str, float]): Dictionary mapping stable items
          to their final stability scores.
        - removed_log (list[tuple[str, float, int]]): Log of removed items,
          their stability score at removal, and the iteration number.
    """
    if not initial_items:
        return [], {}, []

    current_items = list(initial_items)
    removed_log = []
    iteration = 0

    # --- Initialize variables before the loop --- #
    unstable_items = [] # Ensure this is always defined
    final_stability_scores = {} # Ensure this is always defined

    while True:
        iteration += 1
        if verbose:
            print(f"\n--- bootEGA Iteration {iteration} ---")
            print(f"Items remaining: {len(current_items)}")

        if not current_items:
            if verbose:
                print("No items left to analyze.")
            break

        # 1. Run Bootstrap Resampling on Current Items
        # Need embeddings corresponding to *current_items*
        try:
            current_indices = [label_to_index(lbl) for lbl in current_items]
            if not current_indices:
                raise ValueError("Could not map any current items to indices.")
            current_embeddings = original_embeddings[current_indices, :]
        except ValueError as e:
             print(f"Error mapping current item labels in iteration {iteration}: {e}")
             # Decide how to handle: maybe break or return current state?
             # For now, let's break, assuming something is wrong.
             break

        bootstrap_results = run_bootega_resampling(
            items=current_items,
            embeddings=current_embeddings,
            network_method=network_method,
            network_params=network_params,
            walktrap_params=walktrap_params,
            n_bootstrap=n_bootstrap,
            sample_size=sample_size, # Allow specific sample size if needed
            use_parallel=use_parallel,
            max_workers=max_workers,
            progress_callback=progress_callback # Pass callback down
        )

        # 2. Calculate Item Stability
        item_stability = defaultdict(float)
        item_appearances = defaultdict(int)
        valid_bootstrap_runs = 0

        for sampled_items_unique, communities in bootstrap_results:
            if communities is not None:
                valid_bootstrap_runs += 1
                for item in sampled_items_unique:
                    if item in current_items: # Ensure item is still relevant
                        item_appearances[item] += 1
                        original_comm = original_community_membership.get(item)
                        current_comm = communities.get(item)

                        if original_comm != -99 and current_comm == original_comm:
                            item_stability[item] += 1

        # --- Handle case where all bootstrap runs failed --- #
        if valid_bootstrap_runs == 0:
            print(f"Warning: All {n_bootstrap} bootstrap runs failed in iteration {iteration}. Cannot calculate stability.")
            # Decide outcome: Treat all items as unstable? Return error? Stop?
            # Let's stop and return the current state, logging the issue.
            final_stability_scores = {item: np.nan for item in current_items} # Indicate NaN stability
            unstable_items = [] # Ensure it's defined, though loop will break
            # removed_log will contain items removed in previous iterations
            break # Exit the while loop

        # Normalize stability scores
        current_stability_scores = {}
        for item in current_items:
            if item_appearances[item] > 0:
                current_stability_scores[item] = item_stability[item] / item_appearances[item]
            else:
                # Item never appeared in a successful bootstrap sample (should be rare if n_bootstrap is large)
                current_stability_scores[item] = 0.0 # Assign 0 stability

        # Store these scores as the potential final scores if loop terminates here
        final_stability_scores = current_stability_scores.copy()

        # 3. Identify Unstable Items
        unstable_items = [
            item for item in current_items
            if current_stability_scores.get(item, 0.0) < stability_threshold
        ]

        if not unstable_items:
            # All remaining items are stable
            if verbose:
                print(f"All {len(current_items)} items stable at threshold {stability_threshold}. bootEGA finished.")
            break # Exit the while loop
        else:
            # 4. Remove the *least* stable item below the threshold
            unstable_items.sort(key=lambda item: current_stability_scores.get(item, 0.0))
            item_to_remove = unstable_items[0]
            stability_at_removal = current_stability_scores.get(item_to_remove, 0.0)

            if verbose:
                print(f"Removing item '{item_to_remove}' (Stability: {stability_at_removal:.3f} < {stability_threshold}) in iteration {iteration}.")

            removed_log.append((item_to_remove, stability_at_removal, iteration))
            current_items.remove(item_to_remove)

            # Clear the progress callback state if it exists
            if progress_callback:
                 progress_callback(0, f"Iteration {iteration} finished. Starting next...")

    # Return the final list of stable items and the log
    return current_items, final_stability_scores, removed_log


# --- Add main block for testing ---
if __name__ == '__main__':
    # Example usage/testing area (add tests for new functions)
    print("Running basic tests for ega_service...")

    # Test data
    test_items = [f"Item_{i}" for i in range(10)]
    np.random.seed(42)
    test_dense_embeddings = np.random.rand(10, 50) # 10 items, 50 dims
    test_sparse_embeddings = sp.random(10, 50, density=0.1, format='csr', random_state=42)


    # Test calculate_similarity_matrix
    print("\nTesting calculate_similarity_matrix...")
    sim_dense = calculate_similarity_matrix(test_dense_embeddings)
    print(f"Dense similarity matrix shape: {sim_dense.shape}")
    assert sim_dense.shape == (10, 10)
    sim_sparse = calculate_similarity_matrix(test_sparse_embeddings)
    print(f"Sparse similarity matrix shape: {sim_sparse.shape}")
    assert sim_sparse.shape == (10, 10)
    print("Similarity calculation basic check passed.")


    # Test calculate_wto
    print("\nTesting calculate_wto...")
    wto_dense = calculate_wto(sim_dense)
    print(f"Dense wTO matrix shape: {wto_dense.shape}")
    assert wto_dense.shape == (10, 10)
    assert np.all(wto_dense >= 0) and np.all(wto_dense <= 1)
    print("wTO calculation basic check passed.")


    # Test construct_tmfg_network
    print("\nTesting construct_tmfg_network...")
    try:
        tmfg_graph = construct_tmfg_network(sim_dense, item_labels=test_items)
        print(f"TMFG graph nodes: {tmfg_graph.number_of_nodes()}, edges: {tmfg_graph.number_of_edges()}")
        assert tmfg_graph.number_of_nodes() == 10
        # Expected edges for TMFG = 3 * (n - 2) = 3 * (10 - 2) = 24
        assert tmfg_graph.number_of_edges() == 24
        print("TMFG construction basic check passed.")
    except Exception as e:
        print(f"TMFG construction failed: {e}")

    # Test construct_ebicglasso_network
    print("\nTesting construct_ebicglasso_network...")
    try:
        # Using assume_centered=True for similarity matrix input typically
        glasso_graph = construct_ebicglasso_network(sim_dense, item_labels=test_items, assume_centered=True, force_similarity=True)
        print(f"EBICglasso graph nodes: {glasso_graph.number_of_nodes()}, edges: {glasso_graph.number_of_edges()}")
        assert glasso_graph.number_of_nodes() == 10
        # Number of edges varies based on data and lasso alpha
        print("EBICglasso construction basic check passed.")
    except Exception as e:
        print(f"EBICglasso construction failed: {e}")


    # Test detect_communities_walktrap (requires a graph)
    print("\nTesting detect_communities_walktrap...")
    if 'glasso_graph' in locals() and IGRAPH_AVAILABLE:
        try:
            membership, _ = detect_communities_walktrap(glasso_graph)
            print(f"Walktrap detected {len(set(membership.values()))} communities.")
            assert len(membership) == glasso_graph.number_of_nodes()
            print("Walktrap detection basic check passed.")
        except Exception as e:
            print(f"Walktrap detection failed: {e}")
    elif not IGRAPH_AVAILABLE:
         print("Skipping Walktrap test: igraph not available.")
    else:
        print("Skipping Walktrap test: EBICglasso graph not available.")


    # Test calculate_tefi (requires similarity matrix and membership)
    print("\nTesting calculate_tefi...")
    if 'membership' in locals():
        try:
            # Pass explicit item order to avoid deprecation warning
            item_order = list(membership.keys())
            tefi_score = calculate_tefi(sim_dense, membership, item_order=item_order)
            print(f"TEFI score: {tefi_score:.4f}")
            assert isinstance(tefi_score, float)
            print("TEFI calculation basic check passed.")
        except Exception as e:
            print(f"TEFI calculation failed: {e}")
    else:
        print("Skipping TEFI test: Membership not available.")

    # Test calculate_nmi (placeholder test)
    print("\nTesting calculate_nmi...")
    if 'membership' in locals():
         # Create dummy membership for testing structure
         dummy_membership2 = {item: (comm_id + 1) % 3 for item, comm_id in membership.items()}
         try:
             nmi_score = calculate_nmi(membership, dummy_membership2)
             print(f"NMI score (vs dummy): {nmi_score:.4f}")
             assert isinstance(nmi_score, float) or np.isnan(nmi_score)
             print("NMI calculation basic check passed.")
         except Exception as e:
             print(f"NMI calculation failed: {e}")
    else:
        print("Skipping NMI test: Membership not available.")


    # Test remove_redundant_items_uva (requires a graph)
    print("\nTesting remove_redundant_items_uva...")
    if 'glasso_graph' in locals():
        try:
            remaining, removed = remove_redundant_items_uva(glasso_graph.copy(), test_items, wto_threshold=0.15, graph_type='glasso')
            print(f"UVA remaining items: {len(remaining)}, removed: {len(removed)}")
            print(f"Removed log: {removed}")
            assert len(remaining) + len(removed) == len(test_items)
            print("UVA basic check passed.")
        except Exception as e:
             print(f"UVA failed: {e}")
    else:
         print("Skipping UVA test: EBICglasso graph not available.")


    # Test run_bootega_resampling (new function)
    print("\nTesting run_bootega_resampling...")
    if 'glasso_graph' in locals() and IGRAPH_AVAILABLE: # Requires graph for params
        try:
            # Use items remaining after UVA if available, else all items
            items_for_bootega = locals().get('remaining', test_items)
            if not items_for_bootega: items_for_bootega = test_items # Fallback if UVA removed all

            print(f"Running bootEGA resampling on {len(items_for_bootega)} items...")
            # Example parameters (adjust as needed)
            boot_network_params = {'assume_centered': True, 'cv': 3, 'force_similarity': True} # Fewer CV folds for testing
            boot_walktrap_params = {'steps': 4}

            # Run sequentially first for easier debugging if needed
            print("Testing sequential bootEGA...")
            results_seq = run_bootega_resampling(
                items=items_for_bootega,
                embeddings=test_dense_embeddings, # Assuming using dense embeddings here
                network_method='glasso',
                network_params=boot_network_params,
                walktrap_params=boot_walktrap_params,
                n_bootstrap=10, # Small number for testing
                use_parallel=False
            )
            print(f"Sequential run produced {len(results_seq)} results.")
            assert len(results_seq) == 10
            # Check structure of results (list of (sampled_list, community_dict or None))
            assert isinstance(results_seq[0], tuple)
            assert isinstance(results_seq[0][0], list) # sampled_item_list
            assert isinstance(results_seq[0][1], dict) or results_seq[0][1] is None # community_membership_dict
            print("Sequential bootEGA basic check passed.")


            # Test parallel execution
            print("\nTesting parallel bootEGA...")
            results_par = run_bootega_resampling(
                items=items_for_bootega,
                embeddings=test_dense_embeddings,
                network_method='glasso',
                network_params=boot_network_params,
                walktrap_params=boot_walktrap_params,
                n_bootstrap=10,
                use_parallel=True,
                max_workers=2 # Limit workers for testing stability
            )
            print(f"Parallel run produced {len(results_par)} results.")
            assert len(results_par) == 10
            assert isinstance(results_par[0], tuple)
            print("Parallel bootEGA basic check passed.")

        except ImportError as ie:
             print(f"Skipping bootEGA test: {ie}")
        except Exception as e:
            import traceback
            print(f"bootEGA resampling failed: {e}")
            traceback.print_exc()
    elif not IGRAPH_AVAILABLE:
        print("Skipping bootEGA test: igraph not available.")
    else:
        print("Skipping bootEGA test: EBICglasso graph not available for parameter context.")


    # Test perform_bootega_stability_analysis (new function)
    print("\nTesting perform_bootega_stability_analysis...")
    if 'glasso_graph' in locals() and IGRAPH_AVAILABLE and 'membership' in locals():
        try:
            # Use items remaining after UVA if available, else all items
            items_after_uva = locals().get('remaining', test_items)
            if not items_after_uva: items_after_uva = test_items

            # Need original community membership for these items
            # Get the original membership for the items that went into UVA
            original_membership_subset = {item: membership[item] for item in items_after_uva if item in membership}

            if not original_membership_subset:
                 print("Skipping bootEGA stability test: No original membership found for items after UVA.")
            else:
                print(f"Running bootEGA stability analysis on {len(items_after_uva)} items...")
                # Example parameters
                boot_network_params = {'assume_centered': True, 'cv': 3, 'force_similarity': True}
                boot_walktrap_params = {'steps': 4}

                stable_items, final_scores, removed_log = perform_bootega_stability_analysis(
                    initial_items=items_after_uva,
                    original_embeddings=test_dense_embeddings,
                    original_community_membership=original_membership_subset, # Use the relevant subset
                    network_method='glasso',
                    network_params=boot_network_params,
                    walktrap_params=boot_walktrap_params,
                    stability_threshold=0.7, # Lower threshold for testing removal
                    n_bootstrap=20, # Fewer bootstraps for faster testing
                    use_parallel=False, # Easier to debug sequentially first
                    progress_callback=None, # No progress callback for testing
                    verbose=True # Added for potential logging
                )

                print("\nbootEGA Stability Results:")
                print(f"  Stable Items ({len(stable_items)}): {stable_items}")
                print(f"  Final Stability Scores: { {k: round(v, 3) for k, v in final_scores.items()} }")
                print(f"  Removed Items ({len(removed_log)}): { [(i, round(s, 3)) for i, s in removed_log] }")
                assert len(stable_items) + len(removed_log) == len(items_after_uva)
                assert all(item in final_scores for item in stable_items)
                print("bootEGA stability analysis basic check passed.")

        except ImportError as ie:
            print(f"Skipping bootEGA stability test: {ie}")
        except Exception as e:
            import traceback
            print(f"bootEGA stability analysis failed: {e}")
            traceback.print_exc()
    elif not IGRAPH_AVAILABLE:
        print("Skipping bootEGA stability test: igraph not available.")
    else:
        print("Skipping bootEGA stability test: Prerequisite data (graph, membership) not available.")


    print("\nAll basic tests completed.")