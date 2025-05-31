import os
import toml
import re
import traceback
import logging

import streamlit as st
from openai import OpenAI, APIError
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.prompting import (
    DEFAULT_NEGATIVE_EXAMPLES,
    DEFAULT_POSITIVE_EXAMPLES,
    create_system_prompt,
    create_user_prompt,
    parse_generated_items,
    filter_unique_items,
)
from src.embedding_service import get_dense_embeddings, get_sparse_embeddings_tfidf, memory
from src.ega_service import (
    calculate_similarity_matrix,
    construct_tmfg_network,
    construct_ebicglasso_network,
    detect_communities_walktrap,
    calculate_tefi,
    calculate_nmi,
    remove_redundant_items_uva,
    perform_bootega_stability_analysis,
    IGRAPH_AVAILABLE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    # EGA State (Graphs, Communities, Metrics) - Ensure keys are distinct
    graph_types = ['tmfg', 'glasso']
    input_types = ['dense', 'sparse']
    for g_type in graph_types:
        for i_type in input_types:
            key_prefix = f"{g_type}_{i_type}"
            if f"graph_{key_prefix}" not in st.session_state:
                st.session_state[f"graph_{key_prefix}"] = None
            if f"community_membership_{key_prefix}" not in st.session_state:
                st.session_state[f"community_membership_{key_prefix}"] = None
            if f"community_clustering_{key_prefix}" not in st.session_state:
                st.session_state[f"community_clustering_{key_prefix}"] = None
            if f"tefi_{key_prefix}" not in st.session_state:
                st.session_state[f"tefi_{key_prefix}"] = None
            if f"nmi_{key_prefix}" not in st.session_state: # Placeholder NMI
                st.session_state[f"nmi_{key_prefix}"] = None # Initialize NMI state
            if f"ega_error_{key_prefix}" not in st.session_state:
                st.session_state[f"ega_error_{key_prefix}"] = None

    if "show_network_labels" not in st.session_state:
         st.session_state.show_network_labels = False # Default to hide labels

    # UVA State
    if "uva_final_items" not in st.session_state:
        st.session_state.uva_final_items = None
    if "uva_removed_log" not in st.session_state:
        st.session_state.uva_removed_log = None
    if "uva_error" not in st.session_state:
        st.session_state.uva_error = None
    if "uva_status" not in st.session_state:
        st.session_state.uva_status = "Not Run"

    # bootEGA State (New)
    if "bootega_stable_items" not in st.session_state:
        st.session_state.bootega_stable_items = None
    if "bootega_final_stability_scores" not in st.session_state:
        st.session_state.bootega_final_stability_scores = None
    if "bootega_removed_log" not in st.session_state:
        st.session_state.bootega_removed_log = None
    if "bootega_final_nmi" not in st.session_state:
        st.session_state.bootega_final_nmi = None
    if "bootega_initial_nmi_compared" not in st.session_state: # Store the initial NMI for comparison
        st.session_state.bootega_initial_nmi_compared = None
    if "bootega_error" not in st.session_state:
        st.session_state.bootega_error = None
    if "bootega_status" not in st.session_state:
        st.session_state.bootega_status = "Not Run"
    if "bootega_final_community_membership" not in st.session_state:
         st.session_state.bootega_final_community_membership = None # Store final communities after bootEGA

    # --- Helper Function for Clearing State ---
    def clear_downstream_state(from_section: int):
        keys_to_clear = []
        # Based on the section where clearing is triggered, clear subsequent sections' state
        if from_section <= 4: # Clear Embeddings onwards
             keys_to_clear.extend(['dense_embeddings', 'sparse_embeddings_tfidf', 'embedding_error'])
        if from_section <= 5: # Clear Similarity onwards
            keys_to_clear.extend(['similarity_matrix_dense', 'similarity_matrix_sparse', 'similarity_error'])
        if from_section <= 6: # Clear EGA onwards
            for g_type in graph_types:
                for i_type in input_types:
                    key_prefix = f"{g_type}_{i_type}"
                    keys_to_clear.extend([
                        f"graph_{key_prefix}",
                        f"community_membership_{key_prefix}",
                        f"community_clustering_{key_prefix}",
                        f"tefi_{key_prefix}",
                        f"nmi_{key_prefix}",
                        f"ega_error_{key_prefix}"
                    ])
        if from_section <= 7: # Clear UVA onwards
             keys_to_clear.extend(['uva_final_items', 'uva_removed_log', 'uva_error', 'uva_status'])
        if from_section <= 8: # Clear bootEGA onwards (New)
             keys_to_clear.extend([
                 'bootega_stable_items', 'bootega_final_stability_scores',
                 'bootega_removed_log', 'bootega_final_nmi',
                 'bootega_initial_nmi_compared', 'bootega_error', 'bootega_status',
                 'bootega_final_community_membership'
             ])
        # Add future sections here (e.g., export)

        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None
            # Reset status flags too
            if key.endswith("_status"):
                 st.session_state[key] = "Not Run"
            elif key.endswith("_error"):
                 st.session_state[key] = None

        # Special reset for confirmation flag if clearing from Section 3 or earlier
        if from_section <= 3:
            st.session_state.items_confirmed = False
            st.session_state.item_pool_text = "" # Reset editor text too
            st.session_state.previous_items = []

    # --- Section 1: Select Focus and Parameters ---
    st.header("1. Select Focus and Parameters")
    selected_option = st.selectbox(
        "Select a focus area for the items:",
        options,
        key="focus_area_selectbox"
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

    top_p = st.slider(
        "Top P (nucleus sampling):",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help=(
            "Controls diversity via nucleus sampling. Lower values (0.1) = focus on most likely tokens, higher values (1.0) = consider all tokens."
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
                        top_p=top_p,
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
            clear_downstream_state(3)
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
            clear_downstream_state(4)
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
        graph_state_key = f"graph_{method_prefix}_{matrix_suffix}"
        community_membership_key = f"community_membership_{method_prefix}_{matrix_suffix}"
        community_clustering_key = f"community_clustering_{method_prefix}_{matrix_suffix}"
        tefi_key = f"tefi_{method_prefix}_{matrix_suffix}" # Key for TEFI results
        nmi_key = f"nmi_{method_prefix}_{matrix_suffix}"   # Key for NMI results
        ega_error_key = f"ega_error_{method_prefix}_{matrix_suffix}"

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
            st.session_state[community_clustering_key] = None
            st.session_state[tefi_key] = None # Clear TEFI too
            st.session_state[nmi_key] = None # Clear NMI
            st.session_state[ega_error_key] = None # Clear error
            clear_downstream_state(7) # Clear UVA and bootEGA if graph is reconstructed

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
        current_clustering = st.session_state.get(community_clustering_key) # Need clustering for modularity
        current_tefi = st.session_state.get(tefi_key) # Get current TEFI score
        current_nmi = st.session_state.get(nmi_key) # Get current NMI score

        if current_graph is not None:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4) # Add column for TEFI/NMI
            with col_stat1:
                st.metric(f"{network_method} Graph", "Constructed",
                          f"{current_graph.number_of_nodes()} Nodes, {current_graph.number_of_edges()} Edges")
            with col_stat2:
                if current_communities is not None:
                    # Calculate num communities excluding isolated (-1)
                    num_communities = len(set(c for c in current_communities.values() if c != -1))
                    st.metric("Communities (Walktrap)", f"{num_communities} Found", delta=None)
                else:
                    st.metric("Communities (Walktrap)", "Not Detected", "-")
            with col_stat3:
                # Modularity comes from the clustering object
                modularity_val = current_clustering.modularity if current_clustering else None
                if modularity_val is not None:
                     st.metric("Modularity", f"{modularity_val:.4f}", delta=None)
                else:
                     st.metric("Modularity", "-", "-")
            with col_stat4:
                # Display TEFI and NMI (NMI is often NaN initially)
                tefi_display = f"{current_tefi:.4f}" if current_tefi is not None and not pd.isna(current_tefi) else "N/A"
                nmi_display = f"{current_nmi:.4f}" if current_nmi is not None and not pd.isna(current_nmi) else "N/A"
                st.metric("Fit Metrics", tefi_display, f"NMI: {nmi_display}")

            # --- Detect Communities Button ---
            st.caption("Use Walktrap algorithm to find communities within the constructed network.")
            if st.button("Detect Communities", key="detect_communities_button"):
                 with st.spinner("Running Walktrap community detection..."):
                    try:
                        membership_dict, clustering_obj = detect_communities_walktrap(
                            current_graph,
                            weights='weight' # Assuming edges have 'weight' attribute
                        )
                        # Store results using consistent keys
                        st.session_state[community_membership_key] = membership_dict
                        st.session_state[community_clustering_key] = clustering_obj # Store clustering object

                        # Check if communities were successfully detected
                        if membership_dict is not None:
                            # Now calculate TEFI using the result and the original similarity matrix
                            try:
                                tefi_score = calculate_tefi(selected_sim_matrix, membership_dict)
                                st.session_state[tefi_key] = tefi_score # Use consistent key
                                st.success(f"Walktrap detected {len(set(m for m in membership_dict.values() if m != -1))} communities. TEFI calculated: {tefi_score:.4f}")
                            except ValueError as ve:
                                st.error(f"Error calculating TEFI: {ve}")
                                st.session_state[tefi_key] = None # Reset TEFI state on error
                            except Exception as e_tefi:
                                st.error(f"An unexpected error occurred during TEFI calculation: {e_tefi}")
                                st.session_state[tefi_key] = None # Reset TEFI state on error
                                st.code(traceback.format_exc())
                        else:
                            # Handle case where walktrap returns None or empty dict
                            st.session_state[community_membership_key] = membership_dict # Store potentially partial results
                            st.session_state[community_clustering_key] = None
                            st.session_state[tefi_key] = None
                            st.warning("Walktrap completed, but no valid communities found or modularity could not be calculated. TEFI cannot be calculated.")

                        # NMI calculation - Calculate initial NMI comparing detected vs random baseline
                        # For initial NMI, we'll compare against a baseline where each item is in its own community
                        if membership_dict:
                            try:
                                # Create baseline: each item in its own community (maximum modularity baseline)
                                baseline_membership = {item: i for i, item in enumerate(membership_dict.keys())}
                                initial_nmi = calculate_nmi(baseline_membership, membership_dict)
                                st.session_state[nmi_key] = initial_nmi
                            except Exception as e_nmi:
                                # If NMI calculation fails, fall back to NaN
                                st.session_state[nmi_key] = np.nan
                        else:
                            st.session_state[nmi_key] = np.nan

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
                         st.session_state[community_membership_key] = None # Clear state on unexpected error
                         st.session_state[community_clustering_key] = None
                         st.session_state[tefi_key] = None
                         st.session_state[nmi_key] = None


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


    # ==============================================
    # Section 7: Unique Variable Analysis (UVA)
    # ==============================================
    st.header("7. Unique Variable Analysis (UVA)")

    # --- Check if *any* graph has been constructed --- #
    any_graph_constructed = False
    graph_types = ['tmfg', 'glasso']
    input_types = ['dense', 'sparse']
    for g_type in graph_types:
        for i_type in input_types:
            if st.session_state.get(f"graph_{g_type}_{i_type}") is not None:
                any_graph_constructed = True
                break
        if any_graph_constructed:
            break

    # This section is only active if at least one network has been constructed
    if any_graph_constructed:
        st.write("Apply Unique Variable Analysis (UVA) to remove redundant items based on Weighted Topological Overlap (wTO) calculated from the selected network structure.")

        # Read the state directly from the radio button keys used in Section 6
        selected_network_method_uva = st.session_state.get('network_method_select', 'TMFG')
        selected_embedding_type_uva = st.session_state.get('input_matrix_select', 'Dense Embeddings')
        method_prefix_check = "tmfg" if selected_network_method_uva == "TMFG" else "glasso"
        matrix_suffix_check = "dense" if selected_embedding_type_uva == "Dense Embeddings" else "sparse"
        dynamic_graph_key_check = f"graph_{method_prefix_check}_{matrix_suffix_check}"

        # Check if the *specific* graph for the current selection exists
        graph_to_use = st.session_state.get(dynamic_graph_key_check)
        run_uva_disabled = False
        sim_matrix_source_desc = ""

        if graph_to_use is None:
            st.warning(f"The currently selected graph configuration ({selected_network_method_uva} from {selected_embedding_type_uva}) has not been constructed yet in Section 6. Please construct it or select a configuration that has been constructed.")
            run_uva_disabled = True # Disable the UVA button
            sim_matrix_source_desc = "Selected graph not available"
        else:
            graph_type_str = method_prefix_check
            sim_matrix_source_desc = f"{selected_network_method_uva} network (from {selected_embedding_type_uva})"
            st.info(f"UVA will run using the: **{sim_matrix_source_desc}**")

        # Display controls (slider always active, button potentially disabled)
        uva_threshold = st.slider(
            "wTO Redundancy Threshold",
            min_value=0.0, max_value=1.0, value=0.20, step=0.01,
            help="Items in pairs with wTO >= this threshold (calculated on the selected network) are considered redundant. The item with lower total connection strength in the pair is removed iteratively."
        )

        if st.button("Run Unique Variable Analysis (UVA)", key="run_uva", disabled=run_uva_disabled):
            # Ensure graph_to_use is valid before proceeding (should be, due to disabled state, but double-check)
            if not graph_to_use:
                 st.error(f"Cannot run UVA: The required graph ({sim_matrix_source_desc}) is not available. Please construct it in Section 6.")
            elif not st.session_state.previous_items:
                st.error("Cannot run UVA: No confirmed items found.")
            else:
                try:
                    # Get the item labels corresponding to the nodes in the graph
                    current_item_labels = list(graph_to_use.nodes())

                    with st.spinner(f"Running UVA with threshold {uva_threshold:.2f} on {sim_matrix_source_desc}..."):
                        remaining_items, removed_log = remove_redundant_items_uva(
                            graph=graph_to_use,
                            item_labels=current_item_labels,
                            wto_threshold=uva_threshold,
                            graph_type=graph_type_str # Use the determined graph type
                        )
                    st.session_state.uva_final_items = remaining_items
                    st.session_state.uva_removed_log = removed_log
                    st.session_state.uva_threshold = uva_threshold # Store the threshold used
                    st.session_state.uva_input_source = sim_matrix_source_desc # Store which graph was used
                    st.session_state.uva_status = "Completed"
                    st.session_state.uva_error = None # Clear previous error
                    clear_downstream_state(8) # Clear bootEGA results if UVA is re-run
                    st.success(f"UVA completed using {sim_matrix_source_desc}.")
                    st.rerun()
                except (ValueError, TypeError, KeyError, RuntimeError) as e:
                    st.error(f"Error during UVA calculation: {e}")
                    st.exception(e)
                    st.session_state.uva_final_items = None
                    st.session_state.uva_removed_log = None
                    st.session_state.uva_error = f"Error during UVA: {traceback.format_exc()}"
                    st.session_state.uva_status = "Error"
                except Exception as e:
                    st.error(f"An unexpected error occurred during UVA: {e}")
                    st.exception(e)
                    st.session_state.uva_final_items = None
                    st.session_state.uva_removed_log = None
                    st.session_state.uva_error = f"Unexpected error during UVA: {traceback.format_exc()}"
                    st.session_state.uva_status = "Error"

        # --- Display UVA Results --- #
        if st.session_state.uva_status == "Completed" and st.session_state.uva_final_items is not None:
            st.subheader("UVA Results")
            col1, col2, col3 = st.columns(3)
            removed_count = len(st.session_state.uva_removed_log)
            remaining_count = len(st.session_state.uva_final_items)
            # Calculate initial count based on items passed *into* this UVA run
            # We need to know how many items were in graph_to_use
            initial_count = graph_to_use.number_of_nodes() if graph_to_use else (remaining_count + removed_count)
            col1.metric("Items Before UVA", initial_count)
            col2.metric("Items Removed", removed_count)
            col3.metric("Items After UVA", remaining_count)
            st.caption(f"Based on threshold wTO >= {st.session_state.get('uva_threshold', 'N/A'):.2f} using {st.session_state.get('uva_input_source', 'N/A')}")

            if st.session_state.uva_removed_log:
                st.write("**Items Removed by UVA:**")
                removed_data = []
                for item_label, wto in st.session_state.uva_removed_log:
                    actual_text = get_item_text_from_label(item_label, st.session_state.previous_items)
                    removed_data.append({
                        "Removed Item": actual_text,
                        f"Max wTO Trigger ({st.session_state.get('uva_threshold', 'N/A'):.2f})": f"{wto:.4f}"
                    })
                st.dataframe(pd.DataFrame(removed_data), use_container_width=True)

            st.write("**Final Item Pool after UVA:**")
            final_items_text = format_items_with_text(st.session_state.uva_final_items, st.session_state.previous_items)
            st.text_area("Final Items", value=final_items_text, height=200, key="uva_final_items_display", disabled=True)
        elif st.session_state.uva_status == "Error":
            st.error("UVA could not be completed due to an error. Check logs or adjust parameters.")
            if st.session_state.uva_error:
                 st.code(st.session_state.uva_error)

    else:
        st.info("Construct a network (TMFG or EBICglasso using selected embeddings) in Section 6 to enable Unique Variable Analysis.")


    # ==============================================
    # Section 8: bootEGA Stability Analysis
    # ==============================================
    st.header("8. bootEGA Stability Analysis")

    # --- Configuration --- # Display config info but get items directly
    st.markdown("**Configuration Info (based on last UVA run):**")
    network_method_info = st.session_state.get("network_method_select", "N/A") # Method selected when UVA was likely run
    input_matrix_type_info = st.session_state.get("input_matrix_select", "N/A") # Embedding type selected
    uva_threshold_info = st.session_state.get("uva_threshold", 0.20) # Threshold used for UVA
    uva_input_source_info = st.session_state.get('uva_input_source', 'N/A') # Which graph was used for UVA

    # --- Get actual items remaining after the last successful UVA run --- #
    items_after_uva = st.session_state.get("uva_final_items") or []

    st.write(f"- **UVA ran on:** {uva_input_source_info}") # Display which graph UVA used
    st.write(f"- **UVA Threshold Used:** {uva_threshold_info:.2f}")
    st.write(f"- **Items Remaining After UVA:** {len(items_after_uva)}")

    col1_boot, col2_boot = st.columns(2)
    with col1_boot:
        n_bootstraps = st.number_input(
            "Number of Bootstrap Samples (N)", min_value=10, max_value=1000, value=100, step=10,
            key="bootega_n_bootstraps",
            help="Number of bootstrap resamples to perform (e.g., 100-500)."
        )
    with col2_boot:
        stability_threshold = st.slider(
            "Item Stability Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
            key="bootega_stability_threshold",
            help="Items with stability below this value will be removed (proportion, 0.0-1.0)."
        )

    # Simplified parallel toggle - always allow toggle, default based on common env var
    default_parallel = not os.getenv("STREAMLIT_SERVER_RUNNING_ON", "").startswith("streamlit.app")
    use_parallel = st.toggle("Use Parallel Processing", value=default_parallel, key="bootega_use_parallel",
                            help="Use multiple CPU cores for faster resampling (may be unstable on Streamlit Cloud).")

    # --- Execution --- #
    if st.button("Run bootEGA Stability Analysis", key="run_bootega_button", disabled=not items_after_uva):

        # --- Input Validation & Preparation ---
        if not items_after_uva:
            st.error("No items remaining after UVA (Section 7). Cannot run bootEGA.")
            st.stop()

        # Determine embedding type and fetch from state
        matrix_suffix = "dense" if input_matrix_type_info == "Dense Embeddings" else "sparse"
        embedding_key = f"{matrix_suffix}_embeddings" if matrix_suffix == "dense" else f"{matrix_suffix}_embeddings_tfidf"
        selected_embedding_for_bootega = st.session_state.get(embedding_key)
        if selected_embedding_for_bootega is None:
            st.error(f"Required embeddings ('{embedding_key}') not found in session state. Please run Section 4.")
            st.stop()

        # Determine method prefix and fetch original community membership using CORRECT key construction
        method_prefix = "tmfg" if network_method_info == "TMFG" else "glasso"
        original_comm_key = f"community_membership_{method_prefix}_{matrix_suffix}"
        original_community_membership = st.session_state.get(original_comm_key)
        if original_community_membership is None:
            st.error(f"Original community membership ('{original_comm_key}') not found. Please run Section 6.")
            st.stop()

        # Subset original community membership to only include items remaining after UVA
        original_community_membership_subset = {
            item: comm_id for item, comm_id in original_community_membership.items()
            if item in items_after_uva
        }
        if not original_community_membership_subset:
            st.error("Filtered original community membership is empty after UVA. Cannot run bootEGA.")
            st.stop()

        # Fetch initial NMI from EGA results (Section 6) CORRECTLY
        initial_nmi_key = f"nmi_{method_prefix}_{matrix_suffix}"
        initial_nmi = st.session_state.get(initial_nmi_key, np.nan) # Get the NMI from section 6

        # Define Network/Walktrap Params
        network_params = {"assume_centered": True} if method_prefix == "glasso" else {}
        walktrap_params = {"steps": 4}

        # Define Progress Callback for bootEGA - ENSURE this is defined
        progress_bar = st.progress(0.0, text="Starting bootEGA...")
        def update_progress(progress_fraction: float, message: str):
            # Ensure fraction is between 0.0 and 1.0
            clamped_fraction = max(0.0, min(1.0, progress_fraction / 100.0))
            progress_bar.progress(clamped_fraction, text=message)

        # Clear previous bootEGA results and downstream state
        clear_downstream_state(8) # Clear bootEGA and Export state

        st.session_state.bootega_status = "Running bootEGA... (This may take a while)"
        st.session_state.bootega_error = None
        # DO NOT rerun here - let the analysis run first
        # st.rerun()

        try:
            # Run the bootEGA analysis
            stable_items, item_stabilities, removed_log = perform_bootega_stability_analysis(
                initial_items=items_after_uva,
                original_embeddings=selected_embedding_for_bootega, # Pass the correct embeddings
                original_community_membership=original_community_membership_subset, # Subsetted original communities
                network_method=method_prefix, # Pass 'tmfg' or 'glasso'
                network_params=network_params,
                walktrap_params=walktrap_params,
                stability_threshold=stability_threshold,
                n_bootstrap=n_bootstraps,
                sample_size=len(items_after_uva), # Sample size = number of items post-UVA
                use_parallel=use_parallel,
                max_workers=os.cpu_count() if use_parallel else 1,
                progress_callback=update_progress, # Pass the callback
                verbose=True # Enable verbose output for debugging
            )

            # --- Store Results in Session State ---
            st.session_state.bootega_stable_items = stable_items
            st.session_state.bootega_removed_log = removed_log
            st.session_state.bootega_final_stability_scores = item_stabilities

            # --- Perform Final EGA on Stable Items & Calculate Final NMI ---
            st.session_state.bootega_final_community_membership = None
            st.session_state.bootega_final_nmi = None
            st.session_state.bootega_initial_nmi_compared = initial_nmi # Store the initial NMI fetched earlier

            if stable_items:
                # 1. Recalculate similarity matrix for stable items
                # Ensure previous_items exists and is a list before finding indices
                if isinstance(st.session_state.get('previous_items'), list):
                    try:
                        stable_indices = [st.session_state.previous_items.index(item) for item in stable_items if item in st.session_state.previous_items]
                        if not stable_indices:
                             raise ValueError("Could not find indices for any stable items.")
                    except ValueError as e:
                        st.warning(f"Error finding indices for stable items: {e}. Skipping final EGA/NMI.")
                        stable_items = [] # Prevent further processing if indices failed
                else:
                    st.warning("Original item list ('previous_items') not found or invalid. Skipping final EGA/NMI.")
                    stable_items = []

                # Proceed only if we have stable items and valid indices
                if stable_items:
                    stable_embeddings = selected_embedding_for_bootega[stable_indices]
                    final_similarity_matrix = calculate_similarity_matrix(stable_embeddings)

                    # 2. Reconstruct the network
                    if method_prefix == "tmfg":
                        final_graph = construct_tmfg_network(final_similarity_matrix, item_labels=stable_items)
                    else: # ebicglasso
                        final_graph = construct_ebicglasso_network(final_similarity_matrix, item_labels=stable_items, **network_params)

                    # 3. Re-detect communities
                    final_community_membership, _ = detect_communities_walktrap(final_graph, **walktrap_params)
                    st.session_state.bootega_final_community_membership = final_community_membership

                    # 4. Calculate final NMI
                    if final_community_membership:
                        # Subset the *original* community membership (already subsetted for UVA items)
                        # to include only the finally stable items
                        original_membership_stable_subset = {
                            item: original_community_membership_subset[item]
                            for item in stable_items
                            if item in original_community_membership_subset # Should always be true here
                        }
                        # Ensure both dictionaries have the same keys in the same order for NMI
                        items_for_nmi = sorted(list(stable_items))
                        labels_true = [original_membership_stable_subset.get(item, -99) for item in items_for_nmi]
                        labels_pred = [final_community_membership.get(item, -99) for item in items_for_nmi]

                        # Use try-except for NMI calculation as sklearn might not be installed
                        try:
                            final_nmi = calculate_nmi(labels_true, labels_pred)
                            st.session_state.bootega_final_nmi = final_nmi
                        except ImportError:
                             st.warning("scikit-learn not installed. Cannot calculate final NMI.")
                             st.session_state.bootega_final_nmi = np.nan
                        except ValueError as ve:
                             st.warning(f"Error calculating final NMI: {ve}")
                             st.session_state.bootega_final_nmi = np.nan

            st.session_state.bootega_status = "Completed"
            progress_bar.progress(1.0, text="bootEGA Completed!") # Ensure progress bar finishes

        except ImportError as e:
            st.session_state.bootega_status = "Error"
            st.session_state.bootega_error = f"ImportError during bootEGA: {e}. Ensure required libraries (e.g., igraph, scikit-learn) are installed."
            logger.error(f"bootEGA ImportError: {traceback.format_exc()}")
            progress_bar.progress(1.0, text="bootEGA Error")
        except Exception as e:
            st.session_state.bootega_status = "Error"
            st.session_state.bootega_error = f"An unexpected error occurred during bootEGA: {e}"
            logger.error(f"bootEGA Error: {traceback.format_exc()}")
            progress_bar.progress(1.0, text="bootEGA Error")
        finally:
            # Move the rerun here, AFTER all state updates and try/except/finally block
            st.rerun()


    # --- Display bootEGA Results --- #
    st.subheader("bootEGA Results")
    bootega_status = st.session_state.get("bootega_status", "Not Run")

    if bootega_status == "Completed":
        stable_items = st.session_state.get("bootega_stable_items", [])
        removed_log = st.session_state.get("bootega_removed_log", [])
        stability_scores = st.session_state.get("bootega_final_stability_scores", {})
        initial_nmi = st.session_state.get("bootega_initial_nmi_compared", np.nan)
        final_nmi = st.session_state.get("bootega_final_nmi", np.nan)
        items_before_bootega = len(st.session_state.get("uva_final_items", [])) # Items after UVA

        # Format NMI scores for display using pd.isna for robustness
        initial_nmi_display = f"{initial_nmi:.4f}" if not pd.isna(initial_nmi) else "N/A"
        final_nmi_display = f"{final_nmi:.4f}" if not pd.isna(final_nmi) else "N/A"

        st.success("bootEGA analysis completed successfully.")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Items Before bootEGA", items_before_bootega)
        with res_col2:
            st.metric("Items Removed", len(removed_log))
        with res_col3:
            st.metric("Items Stable (Final)", len(stable_items))
        with res_col4:
            st.metric("NMI (Initial→Final)", final_nmi_display, delta=f"Initial: {initial_nmi_display}", delta_color="off")

        # Display Removed Items Log
        if removed_log:
            st.write("**Items Removed During bootEGA Iterations:**")
            removed_data = []
            for item_label, stability_score, iteration in removed_log:
                actual_text = get_item_text_from_label(item_label, st.session_state.previous_items)
                removed_data.append({
                    "Removed Item": actual_text,
                    "Stability Score": f"{stability_score:.4f}",
                    "Iteration": iteration
                })
            removed_df = pd.DataFrame(removed_data)
            st.dataframe(removed_df, use_container_width=True)
        else:
            st.write("No items were removed during bootEGA.")

        # Display Final Stability Scores
        if stability_scores and stable_items:
            st.write("**Final Stability Scores for Stable Items:**")
            scores_data = []
            for item_label in stable_items: # Iterate through stable items to maintain order
                actual_text = get_item_text_from_label(item_label, st.session_state.previous_items)
                score = stability_scores.get(item_label, np.nan)
                scores_data.append({
                    "Stable Item": actual_text,
                    "Final Stability Score": f"{score:.4f}"
                })

            scores_df = pd.DataFrame(scores_data)
            st.dataframe(scores_df, use_container_width=True)

            st.write("**Final Stable Item Pool:**")
            final_stable_items_text = format_items_with_text(stable_items, st.session_state.previous_items)
            st.text_area("Stable Items", value=final_stable_items_text, height=max(150, len(stable_items)*25), key="bootega_stable_items_display", disabled=True)

        elif not stable_items:
             st.warning("No items remained after bootEGA stability analysis.")

    elif bootega_status == "Error":
        st.error("bootEGA analysis failed.")
        error_message = st.session_state.get("bootega_error", "No specific error message recorded.")
        st.code(error_message)
    elif bootega_status == "Running bootEGA... (This may take a while)":
         st.info("bootEGA analysis is currently running. Please wait for completion.")
         # Optionally keep the progress bar visible here if needed, but it's defined inside the button logic
    else: # Not Run or other status
        st.info("Run the bootEGA stability analysis using the button above to see results.")


    # ==============================================
    # Section 9: Export Results
    # ==============================================
    st.header("9. Export Results")

    # Placeholder - Requires Phase 6 implementation
    if st.session_state.get("uva_final_items") is not None: # Check if UVA produced results (bootEGA might modify this later)
        st.info("Export functionality (CSV/PDF report) is planned for Phase 6.")
        # Add download buttons here later
    else:
        st.info("Complete the analysis pipeline (at least through UVA) to enable export.")


# ==============================================
# Helper Functions (moved below main for clarity)
# ==============================================

def parse_items_from_text(text_content: str) -> list[str]:
    """Parses items from a potentially numbered or bulleted list in a text block.

    Args:
        text_content: The string content from the text area.

    Returns:
        A list of cleaned item strings.
    """
    items = []
    lines = text_content.strip().split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # Remove common list markers (digits, dots, hyphens, asterisks followed by space)
        match = re.match(r"^\s*\d+[\.\)]\s*|^\s*[-*+]\s+", cleaned_line)
        if match:
            item_text = cleaned_line[match.end():].strip()
        else:
            item_text = cleaned_line

        if item_text: # Ensure we don't add empty strings
            items.append(item_text)
    return items


def get_item_text_from_label(item_label: str, confirmed_items: list[str]) -> str:
    """Maps an 'Item X' label back to the actual item text.
    
    Args:
        item_label: Label in format 'Item X' where X is a number.
        confirmed_items: List of actual item texts from st.session_state.previous_items.
        
    Returns:
        The actual item text, or the original label if mapping fails.
    """
    try:
        # Extract number from "Item X" format
        match = re.search(r'Item (\d+)', item_label)
        if match:
            item_num = int(match.group(1))
            # Convert to 0-based index
            index = item_num - 1
            if 0 <= index < len(confirmed_items):
                return confirmed_items[index]
    except (ValueError, IndexError):
        pass
    
    # Return original label if mapping fails
    return item_label


def format_items_with_text(items: list[str], confirmed_items: list[str]) -> str:
    """Formats a list of 'Item X' labels with actual item text for display.
    
    Args:
        items: List of item labels in 'Item X' format.
        confirmed_items: List of actual item texts.
        
    Returns:
        Formatted string with numbered list showing actual item text.
    """
    formatted_lines = []
    for i, item_label in enumerate(items):
        actual_text = get_item_text_from_label(item_label, confirmed_items)
        formatted_lines.append(f"{i+1}. {actual_text}")
    return "\n".join(formatted_lines)


def generate_items(
    prompt_focus: str,
    n: int,
    temperature: float,
    top_p: float,
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
        top_p: The nucleus sampling parameter for the LLM (0.0 to 1.0).
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
            top_p=top_p,
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


# ==============================================
# Main Execution Guard
# ==============================================
if __name__ == "__main__":
    main()
