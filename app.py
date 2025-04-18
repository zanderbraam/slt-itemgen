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
from src.ega_service import (
    calculate_similarity_matrix,
    construct_tmfg_network,
    construct_ebicglasso_network,
    detect_communities_walktrap,
    calculate_tefi,
    remove_redundant_items_uva
)

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

    # --- Initialize Session State --- #
    if "item_pool_text" not in st.session_state:
        st.session_state.item_pool_text = ""
    if "previous_items" not in st.session_state:
        st.session_state.previous_items = [] # Holds the list of *confirmed* items
    if "items_confirmed" not in st.session_state:
        st.session_state.items_confirmed = False

    # Embedding State
    if "dense_embeddings" not in st.session_state:
        st.session_state.dense_embeddings = None
    if "sparse_embeddings_tfidf" not in st.session_state:
        st.session_state.sparse_embeddings_tfidf = None
    if "embedding_error" not in st.session_state:
        st.session_state.embedding_error = None

    # Similarity Matrix State
    if "similarity_matrix_dense" not in st.session_state:
        st.session_state.similarity_matrix_dense = None
    if "similarity_matrix_sparse" not in st.session_state:
        st.session_state.similarity_matrix_sparse = None
    if "similarity_error" not in st.session_state:
        st.session_state.similarity_error = None

    # EGA State
    if "ega_method" not in st.session_state:
        st.session_state.ega_method = "TMFG"
    if "ega_input" not in st.session_state:
        st.session_state.ega_input = "Dense Embeddings"
    if "tmfg_graph_dense" not in st.session_state:
        st.session_state.tmfg_graph_dense = None
    if "tmfg_graph_sparse" not in st.session_state:
        st.session_state.tmfg_graph_sparse = None
    if "glasso_graph_dense" not in st.session_state:
        st.session_state.glasso_graph_dense = None
    if "glasso_graph_sparse" not in st.session_state:
        st.session_state.glasso_graph_sparse = None
    if "ega_graph_error" not in st.session_state:
        st.session_state.ega_graph_error = None
    if "community_membership_dense_tmfg" not in st.session_state:
        st.session_state.community_membership_dense_tmfg = None
    if "community_membership_sparse_tmfg" not in st.session_state:
        st.session_state.community_membership_sparse_tmfg = None
    if "community_membership_dense_glasso" not in st.session_state:
        st.session_state.community_membership_dense_glasso = None
    if "community_membership_sparse_glasso" not in st.session_state:
        st.session_state.community_membership_sparse_glasso = None
    if "clustering_object_dense_tmfg" not in st.session_state:
        st.session_state.clustering_object_dense_tmfg = None
    if "clustering_object_sparse_tmfg" not in st.session_state:
        st.session_state.clustering_object_sparse_tmfg = None
    if "clustering_object_dense_glasso" not in st.session_state:
        st.session_state.clustering_object_dense_glasso = None
    if "clustering_object_sparse_glasso" not in st.session_state:
        st.session_state.clustering_object_sparse_glasso = None
    if "community_error" not in st.session_state:
        st.session_state.community_error = None
    if "show_network_labels" not in st.session_state:
        st.session_state.show_network_labels = False # Default to hiding labels
    if "tefi_dense_tmfg" not in st.session_state:
        st.session_state.tefi_dense_tmfg = np.nan
    if "tefi_sparse_tmfg" not in st.session_state:
        st.session_state.tefi_sparse_tmfg = np.nan
    if "tefi_dense_glasso" not in st.session_state:
        st.session_state.tefi_dense_glasso = np.nan
    if "tefi_sparse_glasso" not in st.session_state:
        st.session_state.tefi_sparse_glasso = np.nan
    # NMI will be added later

    # UVA State
    if "uva_remaining_items" not in st.session_state:
        st.session_state.uva_remaining_items = None
    if "uva_removed_items_log" not in st.session_state:
        st.session_state.uva_removed_items_log = None
    if "uva_threshold_used" not in st.session_state:
        st.session_state.uva_threshold_used = 0.20 # Default threshold
    if "uva_run_complete" not in st.session_state:
        st.session_state.uva_run_complete = False
    if "uva_error" not in st.session_state:
        st.session_state.uva_error = None

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

    # --- Section 3: Generate, Edit, and Confirm Items ---
    st.header("3. Generate, Edit, and Confirm Items")

    # Display Generation Controls
    col_gen1, col_gen2 = st.columns([1, 1])
    with col_gen1:
        if st.button("Generate New Items", type="primary", disabled=st.session_state.items_confirmed):
            if selected_option == "---":
                st.warning("Please select a valid focus area.")
            else:
                with st.spinner('Generating items...'):
                    # Use current items from text area for duplicate checking
                    current_items_in_text = parse_items_from_text(st.session_state.item_pool_text)
                    new_items = generate_items(
                        prompt_focus=selected_option,
                        n=n_items,
                        temperature=temperature,
                        previous_items=current_items_in_text, # Pass current text items
                        positive_examples=user_positive_examples or None,
                        negative_examples=user_negative_examples or None,
                        forbidden_words_str=user_forbidden_words or None,
                    )
                    if new_items:
                        # Append to existing text area content
                        existing_text = st.session_state.item_pool_text
                        new_text = "\n".join(new_items)
                        st.session_state.item_pool_text = f"{existing_text}\n{new_text}".strip()
                        st.success(f"{len(new_items)} new unique items generated and appended below.")
                        # Don't rerun here, let user confirm
                    else:
                        st.error("Failed to generate new items. Check OpenAI status or API key.")

    with col_gen2:
        if st.button("Clear Item History & Start Over", disabled=st.session_state.items_confirmed):
            # This button resets everything, including the text area
            st.session_state.item_pool_text = ""
            st.session_state.previous_items = []
            st.session_state.items_confirmed = False
            # Clear downstream states
            st.session_state.dense_embeddings = None
            st.session_state.sparse_embeddings_tfidf = None
            st.session_state.embedding_error = None
            st.session_state.similarity_matrix_dense = None
            st.session_state.similarity_matrix_sparse = None
            st.session_state.similarity_error = None
            st.session_state.tmfg_graph_dense = None
            st.session_state.tmfg_graph_sparse = None
            st.session_state.glasso_graph_dense = None
            st.session_state.glasso_graph_sparse = None
            st.session_state.ega_graph_error = None
            st.session_state.community_membership_dense_tmfg = None
            st.session_state.community_membership_sparse_tmfg = None
            st.session_state.community_membership_dense_glasso = None
            st.session_state.community_membership_sparse_glasso = None
            st.session_state.clustering_object_dense_tmfg = None
            st.session_state.clustering_object_sparse_tmfg = None
            st.session_state.clustering_object_dense_glasso = None
            st.session_state.clustering_object_sparse_glasso = None
            st.session_state.community_error = None
            st.session_state.tefi_dense_tmfg = np.nan
            st.session_state.tefi_sparse_tmfg = np.nan
            st.session_state.tefi_dense_glasso = np.nan
            st.session_state.tefi_sparse_glasso = np.nan
            # Clear UVA state
            st.session_state.uva_remaining_items = None
            st.session_state.uva_removed_items_log = None
            st.session_state.uva_threshold_used = 0.20 # Reset to default
            st.session_state.uva_run_complete = False
            st.session_state.uva_error = None
            # Clear potentially other future states here
            st.rerun()

    # Editable Item Pool Text Area
    st.subheader("Current Item Pool (Edit/Paste/Generate Here)")
    st.session_state.item_pool_text = st.text_area(
        "Items (one per line):",
        value=st.session_state.item_pool_text, # Use the dedicated state variable
        height=300,
        key="item_pool_editor",
        disabled=st.session_state.items_confirmed,
        help="Enter your items here, one per line. You can generate items using the button above, paste a list, or edit manually. Click 'Confirm & Use This Item Pool' below to proceed."
    )

    # Display current item count from text area
    items_in_textarea = parse_items_from_text(st.session_state.item_pool_text)
    st.write(f"Items currently in editor: {len(items_in_textarea)}")

    # --- Confirmation Button --- #
    if st.button("Confirm & Use This Item Pool", type="primary", disabled=st.session_state.items_confirmed):
        final_items = parse_items_from_text(st.session_state.item_pool_text)
        if not final_items:
            st.warning("Item pool is empty. Please add or generate items before confirming.")
        else:
            # Lock in the items
            st.session_state.previous_items = final_items
            # Clear downstream state
            st.session_state.dense_embeddings = None
            st.session_state.sparse_embeddings_tfidf = None
            st.session_state.similarity_matrix_dense = None
            st.session_state.similarity_matrix_sparse = None
            st.session_state.tmfg_graph_dense = None
            st.session_state.tmfg_graph_sparse = None
            st.session_state.glasso_graph_dense = None
            st.session_state.glasso_graph_sparse = None
            st.session_state.communities_tmfg_dense = None
            st.session_state.modularity_tmfg_dense = None
            st.session_state.communities_tmfg_sparse = None
            st.session_state.modularity_tmfg_sparse = None
            st.session_state.communities_glasso_dense = None
            st.session_state.modularity_glasso_dense = None
            st.session_state.communities_glasso_sparse = None
            st.session_state.modularity_glasso_sparse = None
            st.session_state.tefi_tmfg_dense = None
            st.session_state.tefi_tmfg_sparse = None
            st.session_state.tefi_glasso_dense = None
            st.session_state.tefi_glasso_sparse = None
            # Mark as confirmed
            st.session_state.items_confirmed = True
            st.success(f"Item pool confirmed with {len(final_items)} items. Proceed to Section 4.")
            st.rerun()

    if st.session_state.items_confirmed:
        st.success(f"Item pool is confirmed with {len(st.session_state.previous_items)} items. Generation and editing are disabled. Use 'Clear Item History' to start over.")


    # --- Section 4: Generate Embeddings --- #
    st.divider()
    st.header("4. Generate Embeddings")

    if not st.session_state.items_confirmed:
        st.info("Confirm an item pool in Section 3 before generating embeddings.")
    else:
        st.subheader("4.1 Dense Embeddings (OpenAI)")
        st.info(f"Ready to generate dense embeddings for {len(st.session_state.previous_items)} confirmed items.")
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

    if not st.session_state.items_confirmed:
        st.info("Confirm an item pool (Section 3) and generate embeddings (Section 4) first.")
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

    if not st.session_state.items_confirmed or (st.session_state.similarity_matrix_dense is None and st.session_state.similarity_matrix_sparse is None):
        st.info("Confirm items (Section 3), generate embeddings (Section 4), and calculate a similarity matrix (Section 5) first.")
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
        tefi_key = f"tefi_{method_prefix}_{matrix_suffix}" # Key for TEFI results

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
        current_tefi = st.session_state.get(tefi_key) # Get current TEFI score

        if current_graph is not None:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4) # Add column for TEFI/NMI
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
            with col_stat4:
                if current_tefi is not None:
                     st.metric("TEFI", f"{current_tefi:.4f}")
                else:
                    # NMI cannot be calculated yet
                    st.metric("TEFI / NMI", "N/A", "Requires communities")


            # --- Detect Communities Button ---
            st.caption("Use Walktrap algorithm to find communities within the constructed network.")
            if st.button("Detect Communities", key="detect_communities_button"):
                with st.spinner("Running Walktrap community detection..."):
                    try:
                        membership_dict, clustering = detect_communities_walktrap(
                            current_graph,
                            weights='weight' # Assuming edges have 'weight' attribute
                        )
                        # Check if communities were successfully detected
                        if membership_dict is not None and clustering is not None:
                            st.session_state[community_membership_key] = membership_dict
                            st.session_state[modularity_key] = clustering.modularity # Store modularity

                            # Now calculate TEFI using the result and the original similarity matrix
                            try:
                                tefi_score = calculate_tefi(selected_sim_matrix, membership_dict)
                                st.session_state[tefi_key] = tefi_score
                                st.success(f"Walktrap detected {len(set(m for m in membership_dict.values() if m != -1))} communities. TEFI calculated: {tefi_score:.4f}")
                            except ValueError as ve:
                                st.error(f"Error calculating TEFI: {ve}")
                                st.session_state[tefi_key] = None # Reset TEFI state on error
                            except Exception as e_tefi:
                                st.error(f"An unexpected error occurred during TEFI calculation: {e_tefi}")
                                st.session_state[tefi_key] = None # Reset TEFI state on error
                                st.code(traceback.format_exc())
                        else:
                            # Handle case where walktrap returns None (e.g., no positive strength nodes)
                            st.session_state[community_membership_key] = membership_dict # Store potentially partial results (only -1s)
                            st.session_state[modularity_key] = None
                            st.session_state[tefi_key] = None
                            st.warning("Walktrap completed, but no valid communities found or modularity could not be calculated. TEFI cannot be calculated.")

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


    # --- Section 7: Unique Variable Analysis (UVA) - Remove Redundancy --- #
    st.header("7. Unique Variable Analysis (UVA)")

    # Determine if prerequisite similarity matrix for UVA is available
    # UVA needs the *initial* similarity matrix corresponding to the *confirmed* items
    # We'll use the similarity matrix chosen implicitly by the EGA input selection for now.
    # A more robust approach might explicitly store the similarity matrix used for UVA.
    prereq_for_uva = (
        st.session_state.items_confirmed and
        (st.session_state.similarity_matrix_dense is not None or st.session_state.similarity_matrix_sparse is not None)
    )

    if prereq_for_uva:
        st.subheader("7.1 Configure & Run UVA")

        # Select which similarity matrix to use for UVA (based on EGA selection or allow choice)
        # For simplicity, let's base it on the EGA input type selected in Section 6
        uva_input_matrix = None
        uva_input_type_label = ""
        if st.session_state.ega_input == "Dense Embeddings" and st.session_state.similarity_matrix_dense is not None:
            uva_input_matrix = st.session_state.similarity_matrix_dense
            uva_input_type_label = "Dense Similarity Matrix"
        elif st.session_state.ega_input == "Sparse Embeddings (TF-IDF)" and st.session_state.similarity_matrix_sparse is not None:
            uva_input_matrix = st.session_state.similarity_matrix_sparse
            uva_input_type_label = "Sparse Similarity Matrix"
        else:
            # Fallback if EGA input doesn't match available matrix (shouldn't happen if logic is sound)
            if st.session_state.similarity_matrix_dense is not None:
                 uva_input_matrix = st.session_state.similarity_matrix_dense
                 uva_input_type_label = "Dense Similarity Matrix (Fallback)"
            elif st.session_state.similarity_matrix_sparse is not None:
                 uva_input_matrix = st.session_state.similarity_matrix_sparse
                 uva_input_type_label = "Sparse Similarity Matrix (Fallback)"

        if uva_input_matrix is not None:
            st.write(f"Using: **{uva_input_type_label}** (based on Section 6 input selection)")

            # Slider for wTO threshold
            wto_threshold = st.slider(
                "wTO Redundancy Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("uva_threshold_used", 0.20), # Use last used or default
                step=0.01,
                help="Items in pairs with wTO >= this value will be considered for removal. Lower values remove more items. (AI-GENIE paper suggests 0.20)",
                key="uva_threshold_slider"
            )

            # Button to run UVA
            if st.button("Run UVA to Remove Redundant Items", key="run_uva"):
                confirmed_items = st.session_state.previous_items
                if uva_input_matrix is not None and confirmed_items:
                    try:
                        st.session_state.uva_error = None # Clear previous errors
                        with st.spinner("Running Unique Variable Analysis... This may take a moment."):
                            remaining, removed_log = remove_redundant_items_uva(
                                initial_similarity_matrix=uva_input_matrix,
                                item_labels=confirmed_items,
                                wto_threshold=wto_threshold,
                            )
                        st.session_state.uva_remaining_items = remaining
                        st.session_state.uva_removed_items_log = removed_log
                        st.session_state.uva_threshold_used = wto_threshold # Store threshold used
                        st.session_state.uva_run_complete = True
                        st.success("UVA process completed.")
                        st.rerun() # Rerun to update display immediately

                    except (ValueError, TypeError) as e:
                        st.session_state.uva_error = f"UVA Input Error: {e}"
                        st.session_state.uva_run_complete = False
                        st.error(st.session_state.uva_error)
                    except Exception as e:
                        st.session_state.uva_error = f"An unexpected error occurred during UVA: {traceback.format_exc()}"
                        st.session_state.uva_run_complete = False
                        st.error(st.session_state.uva_error)
                else:
                    st.warning("Cannot run UVA. Ensure items are confirmed and a similarity matrix is available.")

            # Display UVA error if it occurred
            if st.session_state.get("uva_error"):
                st.error(f"UVA Error: {st.session_state.uva_error}")

        else:
             st.warning("Selected input similarity matrix type not available. Please calculate it in Section 5.")

        # --- Display UVA Results --- #
        st.subheader("7.2 UVA Results")
        if st.session_state.get("uva_run_complete"):
            removed_log = st.session_state.uva_removed_items_log
            remaining_items = st.session_state.uva_remaining_items
            threshold = st.session_state.uva_threshold_used
            initial_count = len(st.session_state.previous_items)
            removed_count = len(removed_log)
            remaining_count = len(remaining_items)

            st.metric(label=f"Items Removed (wTO >= {threshold:.2f})", value=removed_count, delta=f"{removed_count - initial_count} items removed")
            st.metric(label="Items Remaining", value=remaining_count)

            if removed_log:
                st.write("Items removed due to redundancy:")
                # Convert log to DataFrame for better display
                removed_df = pd.DataFrame(removed_log, columns=['Removed Item', 'Triggering wTO'])
                removed_df['Triggering wTO'] = removed_df['Triggering wTO'].round(4)
                st.dataframe(removed_df, use_container_width=True)
            else:
                st.info("No items were removed based on the current threshold.")

            st.write("Final items after UVA:")
            st.text_area(
                "Remaining Items",
                value="\n".join(remaining_items) if remaining_items else "",
                height=200,
                disabled=True,
                key="uva_remaining_items_display"
            )
        else:
            st.info("Run UVA in section 7.1 to see results.")

    else:
        st.info("Calculate at least one similarity matrix (Section 5) to proceed to UVA.")

    # --- Section 8: bootEGA (Placeholder) --- #
    st.header("8. bootEGA Stability Analysis (Phase 5)")
    if st.session_state.get("uva_run_complete"):
         st.info("TODO: Implement bootEGA based on the remaining items from UVA.")
         # Future logic will depend on st.session_state.uva_remaining_items
    else:
         st.info("Complete Unique Variable Analysis (Section 7) to proceed to bootEGA.")

    # --- Section 9: Export (Placeholder) --- #
    st.header("9. Export Results (Phase 6)")
    if st.session_state.get("uva_run_complete"): # Should depend on bootEGA completion eventually
         st.info("TODO: Implement PDF report and CSV export.")
    else:
         st.info("Complete the analysis pipeline (including UVA and bootEGA) to enable export.")


def parse_items_from_text(text_content: str) -> list[str]:
    """Parses items from a multi-line text block.

    - Splits by newline.
    - Strips leading/trailing whitespace from each line.
    - Removes common list markers (e.g., "1. ", "- ", "* ").
    - Filters out empty lines.

    Args:
        text_content: The string content from the text area.

    Returns:
        A list of parsed item strings.
    """
    items = []
    if not text_content:
        return items

    lines = text_content.strip().split('\n')
    for line in lines:
        cleaned_line = line.strip()
        # Remove potential list markers using regex
        cleaned_line = re.sub(r'^\\s*\\d+\\.\\s*', '', cleaned_line) # "1. "
        cleaned_line = re.sub(r'^\\s*[-*]\\s*', '', cleaned_line)    # "- " or "* "
        if cleaned_line:
            items.append(cleaned_line)
    return items


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
