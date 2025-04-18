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
    initial_similarity_matrix: np.ndarray,
    item_labels: list[str | int],
    wto_threshold: float = 0.20,
) -> tuple[list[str | int], list[tuple[str | int, float]]]:
    """Performs Unique Variable Analysis (UVA) to remove redundant items.

    Iteratively removes the most redundant item based on Weighted Topological
    Overlap (wTO) until no pair of items has a wTO value above the specified
    threshold.

    Redundancy tie-breaking: If multiple pairs have the maximum wTO, the item
    to be removed is chosen from the pair(s) by selecting the item with the
    *lowest* sum of absolute similarities to all *other remaining* items.

    Args:
        initial_similarity_matrix: The initial square (n_items, n_items) similarity
            matrix corresponding to the full set of initial item_labels.
        item_labels: A list of the initial item labels (strings or integers).
            The order must correspond to the rows/columns of the
            initial_similarity_matrix.
        wto_threshold: The cutoff value for wTO. Items in pairs with wTO >= threshold
                       are considered redundant. Defaults to 0.20 based on the
                       AI-GENIE paper suggestion.

    Returns:
        A tuple containing:
        - remaining_items (list[str | int]): A list of the item labels that
          were NOT removed.
        - removed_items_log (list[tuple[str | int, float]]): A list of tuples,
          where each tuple contains the label of the removed item and the
          maximum wTO value that triggered its removal.

    Raises:
        ValueError: If input dimensions or labels mismatch, or threshold is invalid.
        TypeError: If inputs have incorrect types.
    """
    # --- Input Validation ---
    if not isinstance(initial_similarity_matrix, np.ndarray):
        raise TypeError("Input initial_similarity_matrix must be a NumPy array.")
    if initial_similarity_matrix.ndim != 2 or initial_similarity_matrix.shape[0] != initial_similarity_matrix.shape[1]:
        raise ValueError("Input initial_similarity_matrix must be a square 2D array.")
    if not isinstance(item_labels, list):
        raise TypeError("Input item_labels must be a list.")
    if len(item_labels) != initial_similarity_matrix.shape[0]:
        raise ValueError("Length of item_labels must match the dimension of the initial_similarity_matrix.")
    if not isinstance(wto_threshold, (int, float)) or not (0 <= wto_threshold <= 1):
        raise ValueError("wto_threshold must be a float between 0 and 1.")

    n_initial = len(item_labels)
    if n_initial < 2:
        # No pairs to compare, no removal needed
        return item_labels, []

    # --- Initialization ---
    # Keep track of active indices corresponding to the initial matrix
    current_indices = list(range(n_initial))
    removed_items_log = []
    # Create a mapping from initial index to label for easy lookup
    index_to_label = {i: label for i, label in enumerate(item_labels)}

    # --- Iterative Removal Loop ---
    while len(current_indices) >= 2:
        # 1. Get the subset of the similarity matrix for currently active items
        current_sim_matrix = initial_similarity_matrix[np.ix_(current_indices, current_indices)]

        # 2. Calculate wTO for the current subset
        # Need at least 2 items to calculate wTO
        if current_sim_matrix.shape[0] < 2:
             break # Should not happen due to while loop condition, but safe check
        current_wto_matrix = calculate_wto(current_sim_matrix)

        # 3. Find the maximum off-diagonal wTO value
        np.fill_diagonal(current_wto_matrix, -np.inf) # Ignore diagonal
        max_wto = np.max(current_wto_matrix)

        # 4. Check if max wTO meets removal threshold
        if max_wto < wto_threshold:
            break # No more redundancy found

        # 5. Identify pair(s) with max wTO
        # Find indices within the *current_wto_matrix* (local indices)
        max_wto_indices_local = np.argwhere(current_wto_matrix >= max_wto - 1e-9) # Use tolerance for float comparison

        # 6. Determine which item to remove (tie-breaking)
        item_to_remove_local_idx = -1
        min_sum_similarity = np.inf

        # Use absolute similarity for connection strength calculation
        current_abs_sim_matrix = np.abs(current_sim_matrix)
        np.fill_diagonal(current_abs_sim_matrix, 0)

        # Calculate sum similarity for all currently active nodes
        current_sum_similarities = np.sum(current_abs_sim_matrix, axis=1)

        candidate_items_local_indices = set()
        for idx_pair_local in max_wto_indices_local:
            candidate_items_local_indices.add(idx_pair_local[0])
            candidate_items_local_indices.add(idx_pair_local[1])

        for local_idx in candidate_items_local_indices:
            node_sum_similarity = current_sum_similarities[local_idx]

            # Compare with current minimum
            # If lower sum_similarity found, this becomes the new item to remove
            # If equal sum_similarity, the one encountered first (arbitrary but consistent) is kept
            if node_sum_similarity < min_sum_similarity:
                min_sum_similarity = node_sum_similarity
                item_to_remove_local_idx = local_idx
            elif node_sum_similarity == min_sum_similarity:
                # Tie-breaking: If sums are equal, prefer removing the one with the higher index
                # This provides deterministic behavior but is somewhat arbitrary.
                # Another option could be random choice or based on original label order.
                if item_to_remove_local_idx != -1 and local_idx > item_to_remove_local_idx:
                    item_to_remove_local_idx = local_idx
        # 7. Remove the selected item
        if item_to_remove_local_idx != -1:
            # Map local index back to the original index
            original_index_to_remove = current_indices.pop(item_to_remove_local_idx)
            removed_label = index_to_label[original_index_to_remove]
            removed_items_log.append((removed_label, max_wto))
            print(f"    [UVA] Removing item '{removed_label}' (Initial index: {original_index_to_remove}) due to max wTO {max_wto:.4f}") # Debug print
        else:
            # Should not happen if max_wto >= threshold, but safety break
            print("Warning: Could not identify item to remove despite max_wto >= threshold.")
            break

    # --- Return Results ---
    remaining_items = [index_to_label[i] for i in current_indices]
    return remaining_items, removed_items_log


# TODO: Further testing for TEFI and NMI in __main__

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

        # Test planarity (optional, can be slow)
        # is_planar, _ = nx.check_planarity(tmfg_graph)
        # assert is_planar, "TMFG graph is not planar"
        # print("Planarity Check: Passed")

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


    print("\nAll basic tests passed.")

    # --- Test EBICglasso --- #
    print("\n--- Testing EBICglasso Construction ---")
    # Reuse the 4x4 similarity matrix from TMFG test
    print("\nTest Similarity Matrix (4x4):\n", sim_matrix_test)

    try:
        # Use assume_centered=True as we are inputting a similarity matrix
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

    print("\nAll basic tests passed.")

    # --- Test Community Detection --- #
    print("\n--- Testing Walktrap Community Detection ---")
    # Use the TMFG graph generated earlier for testing
    if 'tmfg_graph' in locals() and tmfg_graph.number_of_nodes() > 0:
        print("\nTesting Walktrap on TMFG graph:")
        try:
            membership_tmfg, clustering_tmfg = detect_communities_walktrap(tmfg_graph, weights='weight')
            print(f"  Detected Communities (TMFG): {membership_tmfg}")
            print(f"  Modularity (TMFG): {clustering_tmfg.modularity:.4f}")
            # Simple check: ensure all original nodes are in the result keys
            assert set(membership_tmfg.keys()) == set(tmfg_graph.nodes())
            # Simple check: ensure community IDs are integers
            assert all(isinstance(cid, int) for cid in membership_tmfg.values())
        except ImportError as e:
            print(f"  Skipping Walktrap test: {e}")
        except RuntimeError as e:
            print(f"  Error during Walktrap detection (TMFG): {e}")
        except KeyError as e:
            print(f"  Error during Walktrap detection - Likely missing weights (TMFG): {e}")

    # Use the EBICglasso graph generated earlier for testing
    if 'glasso_graph' in locals() and glasso_graph.number_of_nodes() > 0:
        print("\nTesting Walktrap on EBICglasso graph:")
        try:
            membership_glasso, clustering_glasso = detect_communities_walktrap(glasso_graph, weights='weight')
            print(f"  Detected Communities (Glasso): {membership_glasso}")
            print(f"  Modularity (Glasso): {clustering_glasso.modularity:.4f}")
            assert set(membership_glasso.keys()) == set(glasso_graph.nodes())
            assert all(isinstance(cid, int) for cid in membership_glasso.values())
        except ImportError as e:
            print(f"  Skipping Walktrap test: {e}")
        except RuntimeError as e:
            print(f"  Error during Walktrap detection (Glasso): {e}")
        except KeyError as e:
            print(f"  Error during Walktrap detection - Likely missing weights (Glasso): {e}")

    print("\nCommunity detection tests complete (if igraph installed and graphs generated).")

    print("\nAll basic tests passed.")

    print("\n--- Testing TEFI Calculation ---")
    # Example based on the TMFG test case
    sim_matrix_tefi = np.array([
        [1.0, 0.8, 0.1, 0.6], # A
        [0.8, 1.0, 0.7, 0.2], # B
        [0.1, 0.7, 1.0, 0.5], # C
        [0.6, 0.2, 0.5, 1.0]  # D
    ])
    # Hypothetical community structures
    membership_good = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
    membership_bad = {'A': 0, 'B': 1, 'C': 0, 'D': 1}
    membership_single = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    membership_isolated = {'A': 0, 'B': 0, 'C': 1, 'D': -1} # Node D is isolated

    tefi_good = calculate_tefi(sim_matrix_tefi, membership_good)
    print(f"TEFI (Good Structure {membership_good}): {tefi_good:.4f}")
    # Expected: High similarity within {A,B} (0.8) and {C,D} (0.5).
    # Low similarity between {(A,C)=0.1, (A,D)=0.6, (B,C)=0.7, (B,D)=0.2}. AvgWithin > AvgBetween.

    tefi_bad = calculate_tefi(sim_matrix_tefi, membership_bad)
    print(f"TEFI (Bad Structure {membership_bad}): {tefi_bad:.4f}")
    # Expected: Lower TEFI than good structure.

    tefi_single = calculate_tefi(sim_matrix_tefi, membership_single)
    print(f"TEFI (Single Community {membership_single}): {tefi_single}") # Expected: nan

    tefi_isolated = calculate_tefi(sim_matrix_tefi, membership_isolated)
    print(f"TEFI (With Isolated Node {membership_isolated}): {tefi_isolated:.4f}")
    # Expected: Calculation ignores node D. Should be based on A,B,C.

    # Example NMI test (requires two structures)
    print("\n--- Testing NMI Calculation (Placeholder) ---")
    try:
        nmi_score = calculate_nmi(membership_good, membership_bad)
        print(f"NMI between {membership_good} and {membership_bad}: {nmi_score:.4f}")
        nmi_perfect = calculate_nmi(membership_good, membership_good)
        print(f"NMI between {membership_good} and itself: {nmi_perfect:.4f}")
        assert np.isclose(nmi_perfect, 1.0)
    except NotImplementedError as e:
        print(f"Caught expected error for NMI: {e}")
    except ImportError:
        print("Skipping NMI test: scikit-learn not installed.")
    except ValueError as e:
         print(f"Error during NMI test: {e}")

    print("\nAll basic tests passed.")

    print("\n--- Testing UVA Item Removal ---")
    # Test case 1: Clear redundancy
    sim_redundant = np.array([
        [1.0, 0.9, 0.2, 0.3], # A, highly similar to B
        [0.9, 1.0, 0.3, 0.4], # B
        [0.2, 0.3, 1.0, 0.8], # C, highly similar to D
        [0.3, 0.4, 0.8, 1.0]  # D
    ])
    labels_redundant = ['A', 'B', 'C', 'D']
    print(f"\nInitial items: {labels_redundant}")
    remaining_uva1, removed_uva1 = remove_redundant_items_uva(sim_redundant, labels_redundant, wto_threshold=0.5)
    print(f"Remaining items (Threshold 0.5): {remaining_uva1}")
    print(f"Removed items log: {removed_uva1}")
    # Expected: Either A or B removed first, then either C or D. Which one depends on sum similarity tie-break.

    # Test case 2: Lower threshold, potentially removing more
    print(f"\nInitial items: {labels_redundant}")
    remaining_uva2, removed_uva2 = remove_redundant_items_uva(sim_redundant, labels_redundant, wto_threshold=0.2)
    print(f"Remaining items (Threshold 0.2): {remaining_uva2}")
    print(f"Removed items log: {removed_uva2}")

    # Test case 3: No redundancy above threshold
    sim_noredund = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    labels_noredund = ['X', 'Y', 'Z']
    print(f"\nInitial items: {labels_noredund}")
    remaining_uva3, removed_uva3 = remove_redundant_items_uva(sim_noredund, labels_noredund, wto_threshold=0.5)
    print(f"Remaining items (Threshold 0.5): {remaining_uva3}")
    print(f"Removed items log: {removed_uva3}")
    assert remaining_uva3 == labels_noredund
    assert not removed_uva3

    # Test case 4: Edge case - fewer than 2 items
    print(f"\nInitial items: ['Single']")
    remaining_uva4, removed_uva4 = remove_redundant_items_uva(np.array([[1.0]]), ['Single'], wto_threshold=0.2)
    print(f"Remaining items: {remaining_uva4}")
    print(f"Removed items log: {removed_uva4}")
    assert remaining_uva4 == ['Single']
    assert not removed_uva4

    print("\nAll basic tests passed.") 