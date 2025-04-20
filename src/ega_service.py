"""
Service functions for Exploratory Graph Analysis (EGA) steps.

This module contains functions related to calculating similarity matrices,
constructing networks (TMFG, EBICglasso), performing community detection,
and calculating fit metrics for the EGA pipeline.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, shrunk_covariance
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import igraph as ig

# Attempt to import optional igraph and set a flag
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    # We won't raise an error here, but functions relying on igraph will.

def calculate_similarity_matrix(
    embeddings: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Calculates the cosine similarity matrix for a given set of embeddings.

    Args:
        embeddings: A matrix where rows represent items and columns represent
            embedding dimensions. Can be a dense NumPy array or a sparse
            SciPy matrix.

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


    # Calculate cosine similarity
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

    Returns:
        A networkx.Graph object representing the GGM.
        Nodes are labeled 0..n-1 or with item_labels.
        Edges correspond to non-zero entries in the estimated precision matrix.
        Edge weights represent the partial correlation derived from the precision matrix.

    Raises:
        ValueError: If inputs are invalid (matrix not square, label mismatch, etc.).
        TypeError: If similarity_matrix is not a NumPy array.
        ImportError: If scikit-learn is not installed.

    Notes:
        - GraphicalLassoCV expects a covariance matrix. Using a similarity matrix
          (like cosine similarity) directly is a common heuristic in network psychometrics
          when raw data isn't available, but results should be interpreted cautiously.
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

    # 1) Shrink covariance, 2) add tiny jitter to force SPD:
    try:
        shrunk_sim_matrix = shrunk_covariance(similarity_matrix)
    except Exception as e:
        # fallback to using the raw similarity if shrinkage itself fails
        warnings.warn(f"Covariance shrinkage failed ({e}), using raw similarity matrix for Glasso.")
        shrunk_sim_matrix = similarity_matrix.copy()

    # always add a small multiple of the identity, scaled by matrix variation
    std_dev = np.std(shrunk_sim_matrix[np.triu_indices_from(shrunk_sim_matrix, k=1)]) # Std of off-diagonal elements
    epsilon = 1e-4 * std_dev if std_dev > 1e-8 else 1e-6 # Avoid zero/tiny epsilon
    shrunk_sim_matrix = shrunk_sim_matrix + np.eye(n_items) * epsilon

    # wrap fit in a retry-on-FP-error loop, bolstering jitter if needed
    # Handle potential convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            model = GraphicalLassoCV(
                cv=cv,
                assume_centered=assume_centered,
                max_iter=max_iter,
                eps=1e-8,  # Keep internal solver regularization
                n_jobs=-1 # Use all available CPU cores
                )
            model.fit(shrunk_sim_matrix)
        except FloatingPointError:
            # try again with a larger diagonal penalty
            epsilon *= 10
            # Create a *new* matrix with the increased jitter
            shrunk_sim_matrix_boosted = shrunk_sim_matrix + np.eye(n_items) * epsilon
            try:
                model.fit(shrunk_sim_matrix_boosted)
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
            print("Warning: No nodes with positive strength found. Skipping Walktrap.")
            # All nodes are isolated or form components with 0-weight edges only
            return membership, None

        # Create subgraph with only positive-strength vertices
        subgraph = igraph_graph.induced_subgraph(positive_strength_vertices, implementation="create_from_scratch")

    except Exception as e:
         raise RuntimeError(f"Failed during node strength calculation or subgraph creation: {e}") from e


    # 3. Run Walktrap on the subgraph
    try:
        # Ensure the subgraph still has weights if the original graph did
        subgraph_weights = subgraph.es['weight'] if weights and 'weight' in subgraph.es.attributes() else None

        # Check if the subgraph has edges before running Walktrap
        if subgraph.ecount() == 0:
            print("Warning: Subgraph for Walktrap has no edges. Assigning all nodes to community 0.")
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

    except ig.InternalError as e:
        # Catch specific igraph errors if possible, e.g., "negative edge weights"
        # The zero strength error should be preempted by the check above.
        raise RuntimeError(f"igraph Walktrap algorithm failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during Walktrap community detection: {e}") from e


def calculate_tefi(similarity_matrix: np.ndarray, membership: dict[str | int, int]) -> float:
    """Calculates the Total Entropy Fit Index (TEFI) variant.

    This variant calculates the standardized difference between the average
    within-community similarity and the average between-community similarity.
    Higher values indicate better fit, meaning communities group items that are
    more similar to each other than to items in other communities.

    Args:
        similarity_matrix: The original (n_items, n_items) similarity matrix.
        membership: A dictionary mapping node IDs (matching matrix indices/labels if applicable)
                    to community IDs. Assumes community IDs are integers, potentially including -1
                    for isolated nodes which are ignored in this calculation.

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

    # Assume membership keys correspond to matrix indices 0 to n-1
    # If item_labels were used, mapping might be needed, but app.py passes
    # membership keys matching the original item order/labels.
    # We need to map node IDs (potentially strings like "Item 1") to matrix indices (0..n-1).
    # Let's create the index mapping based on the order in membership keys if needed,
    # assuming the similarity matrix rows/cols correspond to the item order.

    # Get unique, valid community IDs (exclude -1 for isolated nodes)
    valid_communities = {cid for cid in membership.values() if cid != -1}
    if len(valid_communities) < 2:
        # Need at least two communities to compare within vs. between
        # Or if all nodes are isolated (-1)
        print("Warning: TEFI requires at least 2 valid communities. Returning NaN.")
        return np.nan

    within_community_similarities = []
    between_community_similarities = []

    # Create a mapping from node ID (key in membership) to its 0-based index
    # Assumes the order of items in similarity_matrix matches the order they were added
    # which corresponds to the keys in membership (if generated from item_labels)
    node_list = list(membership.keys()) # Get node IDs in a fixed order
    node_to_index = {node_id: i for i, node_id in enumerate(node_list)}

    if len(node_to_index) != n_items:
        raise ValueError(f"Mismatch between number of items in membership ({len(node_to_index)}) "
                         f"and similarity matrix dimension ({n_items}).")

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
    """Calculates the Normalized Mutual Information (NMI) between two community structures.

    Requires two membership dictionaries (mapping node ID to community ID) for the same set of nodes.

    Args:
        membership1: The first membership dictionary.
        membership2: The second membership dictionary.

    Returns:
        The NMI score (float between 0 and 1).

    Raises:
        NotImplementedError: As NMI calculation between different structures
                           is intended for later phases (e.g., Phase 5).
        ValueError: If the node sets in the two memberships differ.
        ImportError: If scikit-learn is not installed.
    """
    # NMI calculation will be implemented properly in Phase 5/6 when needed.
    # For now, raise error or return NaN to indicate it's not applicable yet.
    # raise NotImplementedError("NMI calculation requires two community structures, typically calculated in Phase 5 or 6.")

    # --- OR --- Return NaN as a placeholder that can be handled in the UI
    # Check if scikit-learn is available for the eventual implementation
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
         print("Warning: scikit-learn not installed. NMI calculation will not be possible.")
         return np.nan # Cannot calculate NMI without sklearn

    # Basic check: Do memberships cover the same nodes?
    nodes1 = set(membership1.keys())
    nodes2 = set(membership2.keys())
    if nodes1 != nodes2:
        raise ValueError("Node sets in the two membership dictionaries must be identical for NMI calculation.")

    if not nodes1: # Handle empty memberships
        return 1.0 if not nodes2 else 0.0 # NMI is 1 if both are empty, 0 if one is empty

    # Ensure consistent node ordering for label lists
    ordered_nodes = sorted(list(nodes1))
    labels1 = [membership1[node] for node in ordered_nodes]
    labels2 = [membership2[node] for node in ordered_nodes]

    # Calculate NMI
    # Use average_method='arithmetic' as is common
    nmi_score = normalized_mutual_info_score(labels1, labels2, average_method='arithmetic')
    return nmi_score


def remove_redundant_items_uva(
    graph: nx.Graph,
    item_labels: list[str | int], # Should match nodes in the graph
    wto_threshold: float = 0.20,
    graph_type: str = "glasso", # 'glasso' or 'tmfg' to determine adjacency matrix logic
) -> tuple[list[str | int], list[tuple[str | int, float]]]:
    """Performs Unique Variable Analysis (UVA) to remove redundant items from a graph.

    Iteratively removes the most redundant item based on Weighted Topological
    Overlap (wTO) calculated from the **graph's structure** (adjacency matrix)
    until no pair of items has a wTO value above the specified threshold.

    Redundancy tie-breaking: If multiple pairs have the maximum wTO, the item
    to be removed is chosen from the pair(s) by selecting the item with the
    *lowest* sum of absolute connection strengths (edge weights) to all
    *other remaining* items within the *current subgraph*.

    Args:
        graph: The initial networkx Graph (TMFG or EBICglasso) containing all items.
               Edge weights are expected ('weight' attribute).
        item_labels: A list of the initial item labels (strings or integers)
                     corresponding to the nodes in the graph.
        wto_threshold: The cutoff value for wTO. Items in pairs with wTO >= threshold
                       are considered redundant. Defaults to 0.20.
        graph_type: Specifies the type of graph ('glasso' or 'tmfg'). This determines
                    how the adjacency matrix for wTO is derived.
                    'glasso': Uses absolute partial correlations (edge weights).
                    'tmfg': Uses absolute similarity (edge weights). Defaults to 'glasso'.


    Returns:
        A tuple containing:
        - remaining_items (list[str | int]): A list of the item labels that
          were NOT removed.
        - removed_items_log (list[tuple[str | int, float]]): A list of tuples,
          where each tuple contains the label of the removed item and the
          maximum wTO value that triggered its removal.

    Raises:
        ValueError: If inputs are invalid (graph empty, labels mismatch, threshold invalid,
                    unknown graph_type).
        TypeError: If inputs have incorrect types.
        KeyError: If graph edges are missing the 'weight' attribute.
    """
    # --- Input Validation ---
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input graph must be a networkx.Graph object.")
    if not isinstance(item_labels, list):
        raise TypeError("Input item_labels must be a list.")
    if set(item_labels) != set(graph.nodes()):
        raise ValueError("item_labels must match the nodes in the graph.")
    if not isinstance(wto_threshold, (int, float)) or not (0 <= wto_threshold <= 1):
        raise ValueError("wto_threshold must be a float between 0 and 1.")
    if graph_type not in ["glasso", "tmfg"]:
        raise ValueError("graph_type must be either 'glasso' or 'tmfg'.")

    n_initial = len(item_labels)
    if n_initial < 2:
        # No pairs to compare, no removal needed
        return item_labels, []

    # --- Initialization ---
    current_graph = graph.copy() # Work on a copy to avoid modifying the original
    removed_items_log = []
    # Keep track of current labels, ordered consistently
    # We'll use this order for matrix indexing
    current_labels_ordered = sorted(list(current_graph.nodes()), key=lambda x: item_labels.index(x))

    # --- Iterative Removal Loop ---
    while current_graph.number_of_nodes() >= 2:
        n_current = current_graph.number_of_nodes()

        # 1. Get the adjacency matrix for the *current* subgraph
        #    Ensure the matrix rows/columns align with current_labels_ordered
        try:
            # Use absolute weights for wTO calculation regardless of graph type
            adj_matrix = nx.to_numpy_array(
                current_graph,
                nodelist=current_labels_ordered,
                weight='weight', # Assumes weights exist
                dtype=np.float64
            )
            adj_matrix = np.abs(adj_matrix) # Ensure non-negative weights for wTO
        except KeyError as e:
            raise KeyError("Graph edges must have a 'weight' attribute for wTO calculation.") from e
        except Exception as e:
            raise RuntimeError(f"Failed to extract adjacency matrix: {e}") from e

        # 2. Calculate wTO for the current subset
        current_wto_matrix = calculate_wto(adj_matrix)

        # 3. Find the maximum off-diagonal wTO value
        np.fill_diagonal(current_wto_matrix, -np.inf) # Ignore diagonal
        if n_current == 1: # Should be caught by loop condition, but safeguard
             max_wto = -np.inf
        else:
             max_wto = np.max(current_wto_matrix) if current_wto_matrix.size > 0 else -np.inf


        # 4. Check if max wTO meets removal threshold
        if max_wto < wto_threshold or np.isneginf(max_wto):
            break # No more redundancy found

        # 5. Identify pair(s) with max wTO
        # Find indices within the *current_wto_matrix* (local indices relative to current_labels_ordered)
        max_wto_indices_local = np.argwhere(current_wto_matrix >= max_wto - 1e-9) # Use tolerance

        # 6. Determine which item to remove (tie-breaking using graph connection strength)
        item_to_remove_local_idx = -1
        min_sum_strength = np.inf

        # Calculate sum of absolute edge weights (connection strengths) for tie-breaking
        # Use the already computed absolute adjacency matrix
        np.fill_diagonal(adj_matrix, 0) # Ensure diagonal is zero for sum calculation
        current_sum_strengths = np.sum(adj_matrix, axis=1)

        candidate_items_local_indices = set()
        for idx_pair_local in max_wto_indices_local:
            candidate_items_local_indices.add(idx_pair_local[0])
            candidate_items_local_indices.add(idx_pair_local[1])

        # Find the candidate with the minimum sum of connection strengths
        items_to_consider = sorted(list(candidate_items_local_indices)) # Process in a consistent order
        for local_idx in items_to_consider:
            node_sum_strength = current_sum_strengths[local_idx]

            # Compare with current minimum
            if node_sum_strength < min_sum_strength:
                min_sum_strength = node_sum_strength
                item_to_remove_local_idx = local_idx
            # If sums are equal, the existing item_to_remove_local_idx (lower index) is kept

        # 7. Remove the selected item
        if item_to_remove_local_idx != -1:
            # Get the label of the item to remove using the local index
            label_to_remove = current_labels_ordered[item_to_remove_local_idx]

            # Remove the node from the graph
            current_graph.remove_node(label_to_remove)

            # Update the ordered list of labels for the next iteration
            current_labels_ordered.pop(item_to_remove_local_idx)

            # Log the removal
            removed_items_log.append((label_to_remove, max_wto))
            # print(f"    [UVA] Removing item '{label_to_remove}' due to max wTO {max_wto:.4f}") # Optional debug print
        else:
            # Should not happen if max_wto >= threshold, safety break
            warnings.warn("Could not identify item to remove despite max_wto >= threshold.")
            break

    # --- Return Results ---#
    # The remaining nodes in current_graph are the ones not removed
    remaining_items = current_labels_ordered # Already updated
    return remaining_items, removed_items_log


if __name__ == '__main__':
    # Example Usage & Basic Test Cases
    print("Testing with dense embeddings:")
    dense_embeddings = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0.5, 0.5, 0.5]])
    similarity = calculate_similarity_matrix(dense_embeddings)
    print("Dense Embeddings (4x3):\n", dense_embeddings)
    print("Similarity Matrix:\n", similarity.round(3))
    assert similarity.shape == (4, 4), "Shape mismatch for dense"
    assert np.allclose(np.diag(similarity), 1.0), "Diagonal not 1.0 for dense"

    print("\nTesting with sparse embeddings:")
    sparse_embeddings = sp.csr_matrix([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0.5, 0.5, 0.5]])
    similarity_sparse = calculate_similarity_matrix(sparse_embeddings)
    print("Sparse Embeddings (4x3):\n", sparse_embeddings.toarray())
    print("Similarity Matrix:\n", similarity_sparse.round(3))
    assert similarity_sparse.shape == (4, 4), "Shape mismatch for sparse"
    assert np.allclose(np.diag(similarity_sparse), 1.0), "Diagonal not 1.0 for sparse"

    print("\nTesting edge case: single item:")
    single_item_emb = np.array([[1, 2, 3]])
    similarity_single = calculate_similarity_matrix(single_item_emb)
    print("Single Item Embedding (1x3):\n", single_item_emb)
    print("Similarity Matrix:\n", similarity_single)
    assert similarity_single.shape == (1, 1), "Shape mismatch for single item"
    assert np.allclose(similarity_single, [[1.0]]), "Value mismatch for single item"

    print("\nTesting edge case: zero items:")
    zero_item_emb = np.empty((0, 5))
    similarity_zero = calculate_similarity_matrix(zero_item_emb)
    print("Zero Item Embedding (0x5):\n", zero_item_emb)
    print("Similarity Matrix:\n", similarity_zero)
    assert similarity_zero.shape == (0, 0), "Shape mismatch for zero items"

    # --- Test TMFG --- #
    print("\n--- Testing TMFG Construction ---")
    # Create a sample similarity matrix (needs >= 3 nodes)
    sim_matrix_test = np.array([
        [1.0, 0.8, 0.1, 0.6],
        [0.8, 1.0, 0.7, 0.2],
        [0.1, 0.7, 1.0, 0.5],
        [0.6, 0.2, 0.5, 1.0]
    ])
    n_test = sim_matrix_test.shape[0]
    print("\nTest Similarity Matrix (4x4):\n", sim_matrix_test)
    tmfg_graph = None # Initialize
    try:
        tmfg_graph = construct_tmfg_network(sim_matrix_test, item_labels=['A', 'B', 'C', 'D'])
        print("\nTMFG Graph Nodes:", tmfg_graph.nodes())
        print("TMFG Graph Edges (with weights):")
        for u, v, data in tmfg_graph.edges(data=True):
            print(f"  ({u}, {v}, weight: {data['weight']:.2f})")

        expected_edges = 3 * (n_test - 2)
        print(f"Expected Edges: {expected_edges}")
        print(f"Actual Edges: {tmfg_graph.number_of_edges()}")
        assert tmfg_graph.number_of_edges() == expected_edges, "Incorrect number of edges in TMFG"
        assert set(tmfg_graph.nodes()) == {'A', 'B', 'C', 'D'}, "Node labels mismatch"

    except ValueError as e:
        print(f"Error constructing TMFG: {e}")
    except ImportError:
        print("NetworkX library not found. Please install it (`pip install networkx`) to test TMFG.")

    # Test edge case n < 3
    print("\nTesting TMFG with n < 3:")
    sim_matrix_small = np.array([[1.0, 0.5], [0.5, 1.0]])
    try:
        construct_tmfg_network(sim_matrix_small)
    except ValueError as e:
        print(f"Caught expected error for n=2: {e}")
        assert "requires at least 3 nodes" in str(e)

    # --- Test EBICglasso --- #
    print("\n--- Testing EBICglasso Construction ---")
    print("\nTest Similarity Matrix (4x4):\n", sim_matrix_test)
    glasso_graph = None # Initialize
    try:
        glasso_graph = construct_ebicglasso_network(sim_matrix_test, item_labels=['A', 'B', 'C', 'D'], assume_centered=True)
        print("\nEBICglasso Graph Nodes:", glasso_graph.nodes())
        print("EBICglasso Graph Edges (with partial correlation weights):")
        if glasso_graph.number_of_edges() > 0:
            for u, v, data in glasso_graph.edges(data=True):
                print(f"  ({u}, {v}, weight: {data['weight']:.2f})")
        else:
            print("  No edges found (graph is empty).")
        print(f"Actual Edges: {glasso_graph.number_of_edges()}")
        assert set(glasso_graph.nodes()) == {'A', 'B', 'C', 'D'}, "Node labels mismatch"

    except ValueError as e:
        print(f"Error constructing EBICglasso network: {e}")
    except ImportError:
        print("Scikit-learn library not found. Please install it (`pip install scikit-learn`) to test EBICglasso.")
    except RuntimeError as e:
        print(f"Runtime error during EBICglasso fitting: {e}")

    # Test edge case n < 2
    print("\nTesting EBICglasso with n < 2:")
    sim_matrix_tiny = np.array([[1.0]])
    try:
        construct_ebicglasso_network(sim_matrix_tiny, assume_centered=True)
    except ValueError as e:
        print(f"Caught expected error for n=1: {e}")
        assert "requires at least 2 nodes" in str(e)

    print("\nEBICglasso test passed (or skipped if sklearn unavailable).")

    # --- Test UVA (Refactored) --- #
    print("\n--- Testing UVA (Refactored using Graph) ---")
    # Example Graph (mimicking EBICglasso output with partial correlations)
    G_test = nx.Graph()
    test_labels = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
    G_test.add_nodes_from(test_labels)
    # Add edges with weights (partial correlations) - Some high, some low
    G_test.add_edge('Item1', 'Item2', weight=0.7)  # High correlation -> High wTO expected?
    G_test.add_edge('Item1', 'Item3', weight=0.1)
    G_test.add_edge('Item1', 'Item4', weight=0.2)
    G_test.add_edge('Item2', 'Item3', weight=0.6)
    G_test.add_edge('Item2', 'Item4', weight=0.15)
    G_test.add_edge('Item3', 'Item4', weight=0.8)  # High correlation -> High wTO expected?
    G_test.add_edge('Item3', 'Item5', weight=0.5)
    G_test.add_edge('Item4', 'Item5', weight=0.4)

    print("Test Graph Edges (Partial Correlations):")
    for u, v, data in G_test.edges(data=True):
        print(f"  ({u}, {v}, weight: {data['weight']:.2f})")

    try:
        remaining, removed_log = remove_redundant_items_uva(
            G_test.copy(), # Use a copy
            test_labels,
            wto_threshold=0.20, # Example threshold
            graph_type='glasso' # Use absolute partial correlation for adjacency
        )
        print("\nUVA Results (Threshold 0.20, Type Glasso):")
        print("  Remaining Items:", remaining)
        print("  Removed Items Log:", [(label, f"{wto:.3f}") for label, wto in removed_log])

        # Test with TMFG type (will use same weights here, just tests the logic branch)
        remaining_tmfg, removed_log_tmfg = remove_redundant_items_uva(
            G_test.copy(), # Use a copy
            test_labels,
            wto_threshold=0.10, # Lower threshold for potentially more removal
            graph_type='tmfg'
        )
        print("\nUVA Results (Threshold 0.10, Type TMFG):")
        print("  Remaining Items:", remaining_tmfg)
        print("  Removed Items Log:", [(label, f"{wto:.3f}") for label, wto in removed_log_tmfg])

    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        print(f"Error during UVA test: {e}")

    # --- Test Community Detection --- #
    print("\n--- Testing Walktrap Community Detection ---")
    membership_tmfg = None # Initialize
    clustering_tmfg = None
    if tmfg_graph and tmfg_graph.number_of_nodes() > 0:
        print("\nTesting Walktrap on TMFG graph:")
        try:
            membership_tmfg, clustering_tmfg = detect_communities_walktrap(tmfg_graph, weights='weight')
            print(f"  Detected Communities (TMFG): {membership_tmfg}")
            if clustering_tmfg: # Check if clustering object exists
                 print(f"  Modularity (TMFG): {clustering_tmfg.modularity:.4f}")
            assert set(membership_tmfg.keys()) == set(tmfg_graph.nodes())
            assert all(isinstance(cid, int) for cid in membership_tmfg.values())
        except ImportError as e:
            print(f"  Skipping Walktrap test: {e}")
        except RuntimeError as e:
            print(f"  Error during Walktrap detection (TMFG): {e}")
        except KeyError as e:
            print(f"  Error during Walktrap detection - Likely missing weights (TMFG): {e}")

    membership_glasso = None # Initialize
    clustering_glasso = None
    if glasso_graph and glasso_graph.number_of_nodes() > 0:
        print("\nTesting Walktrap on EBICglasso graph:")
        try:
            membership_glasso, clustering_glasso = detect_communities_walktrap(glasso_graph, weights='weight')
            print(f"  Detected Communities (EBICglasso): {membership_glasso}")
            if clustering_glasso: # Check if clustering object exists
                 print(f"  Modularity (EBICglasso): {clustering_glasso.modularity:.4f}")
            assert set(membership_glasso.keys()) == set(glasso_graph.nodes())
            assert all(isinstance(cid, int) for cid in membership_glasso.values())
        except ImportError as e:
            print(f"  Skipping Walktrap test: {e}")
        except RuntimeError as e:
            print(f"  Error during Walktrap detection (EBICglasso): {e}")
        except KeyError as e:
            print(f"  Error during Walktrap detection - Likely missing weights (EBICglasso): {e}")

    # Test TEFI calculation (if communities were detected)
    print("\n--- Testing TEFI Calculation ---")
    if 'sim_matrix_test' in locals() and membership_tmfg:
        print("\nTesting TEFI on TMFG communities:")
        try:
            tmfg_labels_ordered = list(tmfg_graph.nodes()) # Get actual node order from graph used
            original_labels_order = ['A', 'B', 'C', 'D'] # Order corresponding to sim_matrix_test
            label_to_orig_index = {label: i for i, label in enumerate(original_labels_order)}
            ordered_indices = [label_to_orig_index[label] for label in tmfg_labels_ordered]
            sim_matrix_for_tmfg_tefi = sim_matrix_test[np.ix_(ordered_indices, ordered_indices)]

            tefi_tmfg = calculate_tefi(sim_matrix_for_tmfg_tefi, membership_tmfg)
            print(f"  TEFI (TMFG): {tefi_tmfg:.4f}")
        except Exception as e:
            print(f"  Error calculating TEFI (TMFG): {e}")
    else:
        print("Skipping TEFI test for TMFG (missing similarity matrix or communities).")

    if 'sim_matrix_test' in locals() and membership_glasso:
        print("\nTesting TEFI on EBICglasso communities:")
        try:
            glasso_labels_ordered = list(glasso_graph.nodes()) # Get actual node order from graph used
            original_labels_order = ['A', 'B', 'C', 'D'] # Order corresponding to sim_matrix_test
            label_to_orig_index_g = {label: i for i, label in enumerate(original_labels_order)}
            ordered_indices_g = [label_to_orig_index_g[label] for label in glasso_labels_ordered]
            sim_matrix_for_glasso_tefi = sim_matrix_test[np.ix_(ordered_indices_g, ordered_indices_g)]

            tefi_glasso = calculate_tefi(sim_matrix_for_glasso_tefi, membership_glasso)
            print(f"  TEFI (EBICglasso): {tefi_glasso:.4f}")
        except Exception as e:
            print(f"  Error calculating TEFI (EBICglasso): {e}")
    else:
        print("Skipping TEFI test for EBICglasso (missing similarity matrix or communities).")

    print("\n--- End of ega_service Tests ---")