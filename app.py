import os
import toml
import re
import traceback

import streamlit as st
from openai import OpenAI, APIError
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from src.prompting import (
    DEFAULT_NEGATIVE_EXAMPLES,
    DEFAULT_POSITIVE_EXAMPLES,
    create_system_prompt,
    create_user_prompt,
    parse_generated_items,
    filter_unique_items
)
from src.embedding_service import get_dense_embeddings, get_sparse_embeddings_tfidf, memory
from src.ega_service import calculate_similarity_matrix, construct_tmfg_network, construct_ebicglasso_network, detect_communities_walktrap

# Attempt to import optional igraph for community detection
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    # No need to raise error here, handle gracefully later

# Setup joblib memory cache
CACHE_DIR = "./.joblib_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# memory is already imported from embedding_service, assuming it's initialized there

def load_local_secrets():
    """Load secrets from secrets.toml for local testing."""
    try:
        secrets = toml.load("secrets.toml")["default"]
        for key, value in secrets.items():
            os.environ[key] = value  # Set environment variables
    except FileNotFoundError:
        print("secrets.toml file not found. Ensure it exists for local testing.")
    except KeyError:
        print("Invalid format in secrets.toml. Ensure [default] section is present.")


# Try loading local secrets first
try:
    load_local_secrets()
    api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print(f"Local secrets loading failed: {e}")
    api_key = None

# If no local API key, fall back to Streamlit secrets
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        raise ValueError("No API key found. Add it to secrets.toml for local use or Streamlit secrets for deployment.")

# Ensure the API key is set
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your secrets.toml or Streamlit secrets.")

# Set OpenAI API key
client = OpenAI(
    api_key=api_key,
)

# Options and specific prompts
options = [
    "Engagement in a communicative activity",
    "Motivation during a communicative activity",
    "Persistence in a communicative activity",
    "Social connection in a communicative activity",
    "Sense of belonging in a communicative activity",
    "Affect during a communicative activity",
    "---", # Separator
    "TEST: Big Five - Extraversion",
    "TEST: Big Five - Agreeableness",
    "TEST: Big Five - Conscientiousness",
    "TEST: Big Five - Neuroticism",
    "TEST: Big Five - Openness",
]

# Define default examples here so they can be reused or overridden
# MOVED to src/prompting.py

specific_prompts = {
    "Engagement in a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' engagement in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Motivation during a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' motivation during "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Persistence in a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' persistence in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Social connection in a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' social connection in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Sense of belonging in a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' sense of belonging in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Affect during a communicative activity": (
        "Generate {n} items to measure a child with communication difficulties' affect during "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    # Add Big Five test prompts (Placeholders - refine if needed)
    "TEST: Big Five - Extraversion": (
        "Generate {n} items measuring the personality trait Extraversion (e.g., friendly, positive, assertive, energetic). "
        "Items should be suitable for general adult self-report."
    ),
    "TEST: Big Five - Agreeableness": (
        "Generate {n} items measuring the personality trait Agreeableness (e.g., cooperative, compassionate, trustworthy, humble). "
        "Items should be suitable for general adult self-report."
    ),
    "TEST: Big Five - Conscientiousness": (
        "Generate {n} items measuring the personality trait Conscientiousness (e.g., organized, responsible, disciplined, prudent). "
        "Items should be suitable for general adult self-report."
    ),
    "TEST: Big Five - Neuroticism": (
        "Generate {n} items measuring the personality trait Neuroticism (e.g., anxious, depressed, insecure, emotional). "
        "Items should be suitable for general adult self-report."
    ),
    "TEST: Big Five - Openness": (
        "Generate {n} items measuring the personality trait Openness (e.g., creative, perceptual, curious, philosophical). "
        "Items should be suitable for general adult self-report."
    ),
}


def plot_network_with_communities(
    graph: nx.Graph,
    membership: dict[str | int, int],
    pos: dict | None = None,
    title: str = "Network with Communities",
    ax: plt.Axes | None = None,
    show_labels: bool = False
) -> None:
    """Plots the network with nodes colored by community membership.

    Args:
        graph: The networkx graph.
        membership: Dictionary mapping node ID to community ID. Assumes -1 for isolated nodes.
        pos: Optional layout for nodes (e.g., from nx.spring_layout). If None, calculates spring layout.
        title: Title for the plot.
        ax: Matplotlib axes object to plot on. If None, uses current axes.
        show_labels: If True, display node labels (item numbers).
    """
    if ax is None:
        ax = plt.gca()

    if pos is None:
        pos = nx.spring_layout(graph, seed=42)  # Use a fixed seed for reproducibility

    # Prepare node colors based on community membership
    node_colors = []
    unique_communities = sorted(list(set(membership.values())))
    # Define a color map (e.g., viridis, tab10, etc.)
    # Exclude -1 from the communities used for colormap generation
    valid_communities = [c for c in unique_communities if c != -1]
    cmap = plt.get_cmap('viridis', max(1, len(valid_communities)))  # Ensure at least 1 color

    # Create a mapping from community ID to color
    color_map = {comm_id: cmap(i) for i, comm_id in enumerate(valid_communities)}
    color_map[-1] = 'grey'  # Assign grey to isolated nodes (community -1)

    for node in graph.nodes():
        community_id = membership.get(node, -1) # Default to -1 if node somehow missing
        node_colors.append(color_map[community_id])

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='grey', ax=ax)

    # Conditionally draw labels (extract number from 'Item X' format)
    if show_labels:
        labels = {}
        for node in graph.nodes():
            # Attempt to extract number if label is like 'Item X'
            match = re.search(r'\d+$', str(node))
            if match:
                labels[node] = match.group(0)
            else:
                labels[node] = str(node) # Fallback to full node name
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')

    # Add legend
    legend_handles = []
    # Add isolated node entry first if present
    if -1 in unique_communities:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Isolated',
                                         markerfacecolor='grey', markersize=10))
    # Add other communities
    for comm_id in valid_communities:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Community {comm_id}',
                                         markerfacecolor=color_map[comm_id], markersize=10))

    if legend_handles:
        ax.legend(handles=legend_handles, title="Communities", loc='best')


def main():
    st.title("SLTItemGen: AI-Assisted Psychometric Item Generator")
    st.write("""
        Generate custom items for communicative participation or test Big Five generation.
        Follows principles from the AI-GENIE paper (van Lissa et al., 2024).
        """)

    # --- Section 1: Select Focus and Parameters ---
    st.header("1. Select Focus and Parameters")
    selected_option = st.selectbox(
        "Select a focus area for the items:",
        options,
    )

    n_items = st.slider(
        "Number of items to generate (n):",
        min_value=5,
        max_value=60, # Increased max based on AI-GENIE paper needs
        value=10,
        step=1,
        help="Target number of unique items to request from the AI in this batch.",
    )

    temperature = st.slider(
        "Model creativity (temperature):",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help=(
            "Controls randomness. Lower values (0.0) = more deterministic, higher values (1.5) = more creative/random."
        ),
    )

    # --- Section 2: (Optional) Customize Prompting ---
    st.header("2. (Optional) Customize Prompting")
    with st.expander("Advanced Prompting Options"):
        user_positive_examples = st.text_area(
            "Positive Examples (Good Items - one per line)",
            placeholder="(Optional) Provide examples of the types of items you WANT.\nIf left blank, defaults for the selected focus area will be used (if applicable).",
            height=150,
        )
        user_negative_examples = st.text_area(
            "Negative Examples (Bad Items - one per line)",
            placeholder="(Optional) Provide examples of the types of items you want to AVOID.\nIf left blank, defaults for the selected focus area will be used (if applicable).",
            height=150,
        )
        user_forbidden_words = st.text_area(
            "Forbidden Words (one per line)",
            placeholder="(Optional) List specific words the AI should NOT use in the generated items.",
            height=100,
        )

    # --- Section 3: Generate and View Items ---
    st.header("3. Generate and View Items")

    # Initialize session state
    if 'previous_items' not in st.session_state:
        st.session_state.previous_items = []
    if "dense_embeddings" not in st.session_state:
        st.session_state.dense_embeddings = None
    # Renamed sparse embeddings session state variable
    if "sparse_embeddings_tfidf" not in st.session_state:
        st.session_state.sparse_embeddings_tfidf = None
    # Add session state for similarity matrices
    if "similarity_matrix_dense" not in st.session_state:
        st.session_state.similarity_matrix_dense = None
    if "similarity_matrix_sparse" not in st.session_state:
        st.session_state.similarity_matrix_sparse = None
    # Add session state for TMFG graphs
    if "tmfg_graph_dense" not in st.session_state:
        st.session_state.tmfg_graph_dense = None
    if "tmfg_graph_sparse" not in st.session_state:
        st.session_state.tmfg_graph_sparse = None
    # Add session state for EBICglasso graphs
    if "glasso_graph_dense" not in st.session_state:
        st.session_state.glasso_graph_dense = None
    if "glasso_graph_sparse" not in st.session_state:
        st.session_state.glasso_graph_sparse = None

    # Add session state for community detection results
    if "communities_tmfg_dense" not in st.session_state:
        st.session_state.communities_tmfg_dense = None
    if "modularity_tmfg_dense" not in st.session_state:
        st.session_state.modularity_tmfg_dense = None
    if "communities_tmfg_sparse" not in st.session_state:
        st.session_state.communities_tmfg_sparse = None
    if "modularity_tmfg_sparse" not in st.session_state:
        st.session_state.modularity_tmfg_sparse = None
    if "communities_glasso_dense" not in st.session_state:
        st.session_state.communities_glasso_dense = None
    if "modularity_glasso_dense" not in st.session_state:
        st.session_state.modularity_glasso_dense = None
    if "communities_glasso_sparse" not in st.session_state:
        st.session_state.communities_glasso_sparse = None
    if "modularity_glasso_sparse" not in st.session_state:
        st.session_state.modularity_glasso_sparse = None
    # Add session state for label visibility toggle
    if "show_network_labels" not in st.session_state:
        st.session_state.show_network_labels = False # Default to False (Hide)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate New Items", type="primary"):
            if selected_option == "---":
                st.warning("Please select a valid focus area.")
            else:
                with st.spinner('Generating items...'):
                    new_items = generate_items(
                        prompt_focus=selected_option,
                        n=n_items,
                        temperature=temperature,
                        previous_items=st.session_state.previous_items,
                        # Pass optional user inputs
                        positive_examples=user_positive_examples or None,
                        negative_examples=user_negative_examples or None,
                        forbidden_words_str=user_forbidden_words or None,
                    )
                    if new_items:
                        st.session_state.previous_items.extend(new_items)
                        st.success(f"{len(new_items)} new unique items generated.")
                        # Rerun to update the display immediately after generation
                        st.rerun()
                    else:
                        st.error("Failed to generate new items. Check OpenAI status or API key.")

    with col2:
        if st.button("Clear Item History"):
            st.session_state.previous_items = []
            st.session_state.dense_embeddings = None
            st.session_state.sparse_embeddings_tfidf = None # Clear TF-IDF embeddings too
            # Clear similarity matrices as well
            st.session_state.similarity_matrix_dense = None
            st.session_state.similarity_matrix_sparse = None
            # Clear network graphs
            st.session_state.tmfg_graph_dense = None
            st.session_state.tmfg_graph_sparse = None
            st.session_state.glasso_graph_dense = None
            st.session_state.glasso_graph_sparse = None
            # Clear community results
            st.session_state.communities_tmfg_dense = None
            st.session_state.modularity_tmfg_dense = None
            st.session_state.communities_tmfg_sparse = None
            st.session_state.modularity_tmfg_sparse = None
            st.session_state.communities_glasso_dense = None
            st.session_state.modularity_glasso_dense = None
            st.session_state.communities_glasso_sparse = None
            st.session_state.modularity_glasso_sparse = None
            # Clear label visibility toggle
            st.session_state.show_network_labels = False

            st.success("Item generation history cleared.")
            st.rerun()

    # Display generated items
    st.subheader("Generated Item Pool (Cumulative)")
    if st.session_state.previous_items:
        unique_items = list(dict.fromkeys(st.session_state.previous_items))
        st.write(f"Total unique items: {len(unique_items)}")
        st.text_area("Items:", "\n".join(f"{i+1}. {item}" for i, item in enumerate(unique_items)), height=300)
    else:
        st.info("No items generated yet in this session.")

    # --- Section 4: Generate Embeddings --- #
    st.divider()
    st.header("4. Generate Embeddings")

    if not st.session_state.previous_items:
        st.info("Generate some items first (Section 3) before creating embeddings.")
    else:
        st.subheader("4.1 Dense Embeddings (OpenAI)")
        st.info(f"Ready to generate dense embeddings for {len(st.session_state.previous_items)} unique items.")
        if st.button("Generate Dense Embeddings (OpenAI)", key="generate_dense_embeddings_button"):
            with st.spinner("Generating dense embeddings via OpenAI API (might take a moment)..."):
                try:
                    # Use get_dense_embeddings
                    dense_emb = get_dense_embeddings(
                        st.session_state.previous_items
                    )
                    st.session_state.dense_embeddings = dense_emb

                    if dense_emb is not None:
                        st.success("Successfully generated dense embeddings!")
                    else:
                        st.error("Failed to generate dense embeddings. Check API key and OpenAI status.")
                except Exception as e:
                    st.error(f"An error occurred during dense embedding generation: {e}")
                    if "api key" in str(e).lower():
                        st.error("Please ensure your OPENAI_API_KEY is correctly set.")
                    else:
                        st.code(traceback.format_exc())

        # Display Dense Embedding Status
        if st.session_state.dense_embeddings is not None:
            st.metric("Dense Embeddings Status", "Generated", f"Shape: {st.session_state.dense_embeddings.shape}")
        else:
            st.metric("Dense Embeddings Status", "Not Generated", "-")

        st.subheader("4.2 Sparse Embeddings (TF-IDF)")
        st.info(f"Ready to generate sparse TF-IDF embeddings for {len(st.session_state.previous_items)} unique items.")
        if st.button("Generate Sparse Embeddings (TF-IDF)", key="generate_sparse_embeddings_button"):
            with st.spinner("Generating sparse TF-IDF embeddings locally..."):
                try:
                    # Use get_sparse_embeddings_tfidf
                    sparse_emb_tfidf = get_sparse_embeddings_tfidf(
                        st.session_state.previous_items
                    )
                    st.session_state.sparse_embeddings_tfidf = sparse_emb_tfidf

                    if sparse_emb_tfidf is not None:
                        st.success("Successfully generated sparse TF-IDF embeddings!")
                    else:
                        st.error("Failed to generate sparse TF-IDF embeddings.")
                except Exception as e:
                    st.error(f"An error occurred during sparse TF-IDF embedding generation: {e}")
                    st.code(traceback.format_exc())

        # Display Sparse Embedding Status
        if st.session_state.sparse_embeddings_tfidf is not None:
            st.metric("Sparse TF-IDF Status", "Generated", f"Shape: {st.session_state.sparse_embeddings_tfidf.shape}")
        else:
            st.metric("Sparse TF-IDF Status", "Not Generated", "-")

    # --- Section 5: Calculate Similarity Matrix --- #
    st.divider()
    st.header("5. Calculate Similarity Matrix")
    st.write("Calculate pairwise cosine similarity between items based on their embeddings.")

    if not st.session_state.previous_items:
        st.info("Generate items (Section 3) and embeddings (Section 4) first.")
    else:
        col_sim1, col_sim2 = st.columns(2)

        with col_sim1:
            st.subheader("5.1 From Dense Embeddings")
            if st.session_state.dense_embeddings is not None:
                st.info(f"Dense embeddings available (Shape: {st.session_state.dense_embeddings.shape}).")
                if st.button("Calculate Dense Similarity", key="calc_dense_sim"):
                    with st.spinner("Calculating dense similarity matrix..."):
                        try:
                            sim_matrix = calculate_similarity_matrix(st.session_state.dense_embeddings)
                            st.session_state.similarity_matrix_dense = sim_matrix
                            st.success("Dense similarity matrix calculated!")
                            st.rerun() # Rerun to update display
                        except Exception as e:
                            st.error(f"Error calculating dense similarity: {e}")
                            st.code(traceback.format_exc())
            else:
                st.warning("Generate Dense Embeddings (Section 4.1) first.")

            # Display Dense Similarity Matrix Status
            if st.session_state.similarity_matrix_dense is not None:
                st.metric("Dense Similarity Matrix", "Calculated", f"Shape: {st.session_state.similarity_matrix_dense.shape}")
                # Optional: Display a small preview (can be slow for large matrices)
                # if st.checkbox("Show Dense Similarity Matrix Preview"):
                #     st.dataframe(st.session_state.similarity_matrix_dense)
            else:
                st.metric("Dense Similarity Matrix", "Not Calculated", "-")

        with col_sim2:
            st.subheader("5.2 From Sparse Embeddings")
            if st.session_state.sparse_embeddings_tfidf is not None:
                st.info(f"Sparse TF-IDF embeddings available (Shape: {st.session_state.sparse_embeddings_tfidf.shape}).")
                if st.button("Calculate Sparse Similarity", key="calc_sparse_sim"):
                    with st.spinner("Calculating sparse TF-IDF similarity matrix..."):
                        try:
                            sim_matrix = calculate_similarity_matrix(st.session_state.sparse_embeddings_tfidf)
                            st.session_state.similarity_matrix_sparse = sim_matrix
                            st.success("Sparse TF-IDF similarity matrix calculated!")
                            st.rerun() # Rerun to update display
                        except Exception as e:
                            st.error(f"Error calculating sparse similarity: {e}")
                            st.code(traceback.format_exc())
            else:
                st.warning("Generate Sparse Embeddings (Section 4.2) first.")

            # Display Sparse Similarity Matrix Status
            if st.session_state.similarity_matrix_sparse is not None:
                st.metric("Sparse Similarity Matrix", "Calculated", f"Shape: {st.session_state.similarity_matrix_sparse.shape}")
                # Optional: Display a small preview
                # if st.checkbox("Show Sparse Similarity Matrix Preview"):
                #     st.dataframe(st.session_state.similarity_matrix_sparse)
            else:
                st.metric("Sparse Similarity Matrix", "Not Calculated", "-")

    # --- Section 6: Construct Network & Detect Communities --- #
    st.divider()
    st.header("6. Construct Network & Detect Communities")
    st.write("Build a network graph from the similarity matrix and detect communities using Walktrap.")

    if not st.session_state.previous_items or (st.session_state.similarity_matrix_dense is None and st.session_state.similarity_matrix_sparse is None):
        st.info("Generate items, embeddings, and calculate a similarity matrix first (Sections 3-5).")
    else:
        # --- Network Construction Selection ---
        col_net1, col_net2 = st.columns(2)
        with col_net1:
            network_method = st.radio(
                "Select Network Construction Method:",
                ("TMFG", "EBICglasso"),
                key="network_method_select",
                # horizontal=True, # Keep vertical for clarity
            )
        with col_net2:
            input_matrix_type = st.radio(
                "Select Input Similarity Matrix:",
                ("Dense Embeddings", "Sparse Embeddings (TF-IDF)"),
                key="input_matrix_select",
                # horizontal=True, # Keep vertical for clarity
            )

        # --- Determine state keys based on selections ---
        matrix_suffix = "dense" if input_matrix_type == "Dense Embeddings" else "sparse"
        method_prefix = "tmfg" if network_method == "TMFG" else "glasso"

        selected_sim_matrix = st.session_state.get(f"similarity_matrix_{matrix_suffix}")
        graph_state_key = f"{method_prefix}_graph_{matrix_suffix}"
        community_membership_key = f"communities_{method_prefix}_{matrix_suffix}"
        modularity_key = f"modularity_{method_prefix}_{matrix_suffix}"

        # --- Construct Network Button and Logic ---
        disable_construct_button = selected_sim_matrix is None
        construct_button_label = f"Construct {network_method} Network"
        if selected_sim_matrix is not None:
            st.info(f"Ready to build {network_method} network from {input_matrix_type} similarity matrix (Shape: {selected_sim_matrix.shape}).")
        else:
            st.warning(f"Please calculate the similarity matrix for {input_matrix_type} first (Section 5).")


        if st.button(construct_button_label, key="construct_network_button", disabled=disable_construct_button):
            # Clear previous community results for this graph type if rebuilding
            st.session_state[community_membership_key] = None
            st.session_state[modularity_key] = None

            item_labels = [f"Item {i+1}" for i in range(len(st.session_state.previous_items))]
            if len(item_labels) != selected_sim_matrix.shape[0]:
                    st.error("Mismatch between number of items and similarity matrix dimension during construction.")
                    st.stop() # Prevent further execution in this run

            if network_method == "TMFG":
                with st.spinner(f"Constructing TMFG network from {input_matrix_type} similarity..."):
                    try:
                        graph = construct_tmfg_network(
                            selected_sim_matrix,
                            item_labels=item_labels
                        )
                        st.session_state[graph_state_key] = graph
                        st.success(f"TMFG network constructed successfully from {input_matrix_type} data!")
                        st.rerun()
                    except ValueError as ve:
                         st.error(f"Error during TMFG construction: {ve}")
                         st.info("TMFG typically requires at least 3 items.")
                    except ImportError:
                         st.error("NetworkX library not found. Please install it (`pip install networkx`).")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during TMFG construction: {e}")
                        st.code(traceback.format_exc())
            elif network_method == "EBICglasso":
                 with st.spinner(f"Constructing EBICglasso network from {input_matrix_type} similarity..."):
                    try:
                        graph = construct_ebicglasso_network(
                            selected_sim_matrix,
                            item_labels=item_labels,
                            assume_centered=True # Important when using similarity matrix
                        )
                        st.session_state[graph_state_key] = graph
                        st.success(f"EBICglasso network constructed successfully from {input_matrix_type} data!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Error during EBICglasso construction: {ve}")
                        st.info("EBICglasso typically requires at least 2 items.")
                    except ImportError:
                         st.error("Scikit-learn or NetworkX library not found. Please install them.")
                    except RuntimeError as rt:
                        st.error(f"Runtime error during EBICglasso fitting: {rt}")
                        # st.info("This might be due to issues with the input matrix or convergence problems.") # Redundant with error message
                        st.code(traceback.format_exc())
                    except Exception as e:
                        st.error(f"An unexpected error occurred during EBICglasso construction: {e}")
                        st.code(traceback.format_exc())
            else:
                st.error(f"Network construction method '{network_method}' not implemented yet.")

        # --- Display Status & Community Detection --- #
        st.subheader("Network Status & Community Detection")
        current_graph = st.session_state.get(graph_state_key)
        current_communities = st.session_state.get(community_membership_key)
        current_modularity = st.session_state.get(modularity_key)

        if current_graph is not None:
            col_stat1, col_stat2, col_stat3 = st.columns(3) # Add column for modularity
            with col_stat1:
                st.metric(f"{network_method} Graph", "Constructed",
                          f"{current_graph.number_of_nodes()} Nodes, {current_graph.number_of_edges()} Edges")
            with col_stat2:
                if current_communities is not None:
                    num_communities = len(set(current_communities.values()))
                    st.metric("Communities (Walktrap)", f"{num_communities} Found", delta=None) # Removed delta
                else:
                    st.metric("Communities (Walktrap)", "Not Detected", "-")
            with col_stat3:
                if current_modularity is not None:
                     st.metric("Modularity", f"{current_modularity:.4f}", delta=None) # Removed delta
                else:
                     st.metric("Modularity", "-", "-")


            # --- Detect Communities Button ---
            st.caption("Use Walktrap algorithm to find communities within the constructed network.")
            if st.button("Detect Communities", key="detect_communities_button"):
                with st.spinner("Running Walktrap community detection..."):
                    try:
                        membership_dict, clustering = detect_communities_walktrap(
                            current_graph,
                            weights='weight' # Assuming edges have 'weight' attribute
                        )
                        st.session_state[community_membership_key] = membership_dict
                        st.session_state[modularity_key] = clustering.modularity # Store modularity
                        st.success(f"Walktrap detected {len(set(membership_dict.values()))} communities.")
                        st.rerun() # Rerun to update metrics and plot color
                    except ImportError:
                        st.error("`python-igraph` library not found. Please install it (`pip install python-igraph`).")
                    except KeyError as ke:
                        st.error(f"Error during community detection: {ke}. Ensure graph edges have 'weight' attribute.")
                    except RuntimeError as rt:
                        st.error(f"Runtime error during community detection: {rt}")
                        st.code(traceback.format_exc())
                    except Exception as e:
                         st.error(f"An unexpected error occurred during community detection: {e}")
                         st.code(traceback.format_exc())


            # --- Network Visualization --- #
            # Add radio button for label visibility *before* the expander
            label_visibility_choice = st.radio(
                "Node Labels:",
                ("Hide Item Numbers", "Show Item Numbers"),
                index=1 if st.session_state.show_network_labels else 0, # Set index based on state
                key="label_visibility_radio",
                horizontal=True,
            )
            # Update session state based on radio button choice
            st.session_state.show_network_labels = (label_visibility_choice == "Show Item Numbers")

            with st.expander("Show Network Visualization", expanded=True): # Expand by default
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pos = nx.spring_layout(current_graph, seed=42) # Calculate layout once

                    if current_communities:
                        plot_network_with_communities(
                            graph=current_graph,
                            membership=current_communities,
                            pos=pos, # Use pre-calculated layout
                            title=f"{network_method} Network ({input_matrix_type.capitalize()}) - Walktrap Communities",
                            ax=ax,
                            show_labels=st.session_state.show_network_labels # Pass label visibility flag
                        )
                    else:
                        # Plot without community colors if detection hasn't run or failed
                        # Still respect the show_labels flag
                        nx.draw_networkx_nodes(current_graph, pos, node_size=300, node_color='skyblue', ax=ax)
                        nx.draw_networkx_edges(current_graph, pos, edge_color='gray', ax=ax)
                        if st.session_state.show_network_labels:
                             labels = {}
                             for node in current_graph.nodes():
                                 match = re.search(r'\d+$', str(node))
                                 if match:
                                     labels[node] = match.group(0)
                                 else:
                                     labels[node] = str(node)
                             nx.draw_networkx_labels(current_graph, pos, labels=labels, font_size=8, ax=ax)

                        ax.set_title(f"{network_method} Network ({input_matrix_type.capitalize()})")
                    st.pyplot(fig)
                except ImportError:
                    st.error("Matplotlib or Igraph required for visualization/community detection. Please install them (`pip install matplotlib python-igraph`).")
                except Exception as e:
                    st.error(f"An error occurred during network visualization: {e}")
                    st.code(traceback.format_exc())

        else:
            st.metric(f"{network_method} Network ({input_matrix_type})", "Not Constructed", "-")


def generate_items(
    prompt_focus: str,
    n: int,
    temperature: float,
    previous_items: list[str],
    positive_examples: str | None = None,
    negative_examples: str | None = None,
    forbidden_words_str: str | None = None,
) -> list[str]:
    """Generates psychometric items using OpenAI's GPT model based on a focus area.

    Applies prompt engineering techniques including persona priming, few-shot
    examples (default or user-provided), adaptive prompting (duplicate avoidance),
    and forbidden words, based on the AI-GENIE methodology.

    Args:
        prompt_focus: The specific construct to generate items for (from specific_prompts keys).
        n: The target number of unique items to generate in this batch.
        temperature: The creativity/randomness setting for the LLM (0.0 to 1.5).
        previous_items: A list of already generated items in this session to avoid duplicates.
        positive_examples: Optional user-provided string of positive examples (one per line).
                         If None, defaults are used for communicative participation focus.
        negative_examples: Optional user-provided string of negative examples (one per line).
                         If None, defaults are used for communicative participation focus.
        forbidden_words_str: Optional user-provided string of forbidden words (one per line).

    Returns:
        A list of newly generated unique items, or an empty list if an error occurs.

    Raises:
        ValueError: If the prompt_focus is not found in the predefined prompts.
    """
    try:
        # 1. Create User Prompt
        user_prompt_instruction = create_user_prompt(prompt_focus, n, specific_prompts)

        # 2. Determine Persona, Background, and Examples based on focus
        is_big_five_test = prompt_focus.startswith("TEST: Big Five")
        if is_big_five_test:
            persona = "You are an expert psychometrician and test developer specializing in personality assessment."
            background_info = "TARGET DOMAIN: Big Five Personality Traits (Adult Self-Report)"
            positive_examples_to_use = positive_examples or "" # No defaults for Big Five yet
            negative_examples_to_use = negative_examples or "" # No defaults for Big Five yet
        else:
            persona = "You are an expert psychometrician and test developer specializing in pediatric speech-language therapy and communicative participation assessment."
            background_info = (
                "TARGET DOMAIN: Communicative participation outcomes of children aged 6-11 with communication difficulties.\n"
                "DEFINITION: Communicative participation involves participating in life situations where knowledge, information, ideas, "
                "and feelings are exchanged. It means understanding and being understood in social contexts using verbal or non-verbal communication skills.\n"
                "EXAMPLES: Participating in group activities, playing with friends, initiating conversations, school/community involvement, engaging in play.\n"
                "INTENDED USE: Measurement instrument for speech-language therapists (SLTs) and parents."
            )
            positive_examples_to_use = positive_examples or DEFAULT_POSITIVE_EXAMPLES
            negative_examples_to_use = negative_examples or DEFAULT_NEGATIVE_EXAMPLES

        # 3. Construct System Prompt & Get Forbidden Words List
        system_prompt, forbidden_words_list = create_system_prompt(
            persona=persona,
            background_info=background_info,
            positive_examples=positive_examples_to_use,
            negative_examples=negative_examples_to_use,
            previous_items=previous_items,
            forbidden_words_str=forbidden_words_str,
            is_big_five_test=is_big_five_test
        )

        # 4. Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_instruction}
        ]

        # Log the prompt for debugging if needed (optional)
        # print("--- SYSTEM PROMPT ---")
        # print(system_prompt)
        # print("--- USER PROMPT ---")
        # print(user_prompt_instruction)

        # 5. Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max(200, 20 * n), # Adjusted token estimation
            n=1,
            temperature=temperature,
            stop=None
        )

        # 6. Extract, Parse, and Filter Forbidden Words
        generated_text = response.choices[0].message.content.strip()
        parsed_items = parse_generated_items(generated_text, forbidden_words_list)

        # 7. Filter out duplicates relative to previous items
        new_unique_items = filter_unique_items(parsed_items, previous_items)

        return new_unique_items

    except Exception as e:
        st.error(f"An error occurred during item generation: {e}")
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return []


if __name__ == "__main__":
    main()
