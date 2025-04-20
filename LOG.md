# Development Log

## [Date TBD] - Phase 1: Item Generation Foundation

*   Created empty `LOG.md`.
*   Refactored initial `app.py` script:
    *   Renamed `generate_questions` to `generate_items`.
    *   Updated function signature to `(prompt_focus: str, n: int, temperature: float, previous_items: list[str]) -> list[str]` with type hints.
    *   Added Google-style docstring.
    *   Implemented enhanced system prompt using AI-GENIE principles:
        *   AI Persona Priming (Expert psychometrician).
        *   Clear context (target domain, definition, examples, intended use).
        *   Few-shot examples (positive/negative).
        *   Adaptive prompting (duplicate avoidance using `previous_items`).
        *   Strict formatting instructions (output only numbered list).
    *   Updated OpenAI API call (`gpt-4o`) to use refined prompts.
    *   Improved item parsing logic.
    *   Added post-generation duplicate filtering.
    *   Modified Streamlit UI (`main`):
        *   Added slider for `n` items.
        *   Adjusted temperature slider range (0.0-1.5).
        *   Used `st.session_state` to track `previous_items` across runs.
        *   Displayed cumulative unique items generated in the session.
        *   Added "Clear History" button.
*   Updated `PROJECTPLAN.md` to mark Phase 0 (`LOG.md`) and Phase 1 tasks as complete.

## [Date TBD] - Phase 1 Enhancements: Customization & Testing

*   Added Big Five personality traits as test case options in the UI dropdown (`options`) and prompt logic (`specific_prompts`).
*   Added default positive/negative example constants (`DEFAULT_POSITIVE_EXAMPLES`, `DEFAULT_NEGATIVE_EXAMPLES`).
*   Enhanced Streamlit UI (`main`):
    *   Added `st.expander` for "Advanced Prompting Options".
    *   Added optional `st.text_area` inputs for user-provided positive examples, negative examples, and forbidden words.
    *   Improved layout with headers and columns.
    *   Updated button logic to pass new optional inputs to `generate_items`.
    *   Used `st.rerun()` for immediate UI updates after generation/clearing.
    *   Changed item display to `st.text_area` for easier copy/paste.
*   Updated `generate_items` function:
    *   Modified signature and docstring to accept optional examples and forbidden words.
    *   Added logic to use user-provided examples/forbidden words or fall back to defaults/none.
    *   Dynamically adjust persona, background info, and instructions based on focus (Comm. Part. vs. Big Five).
    *   Implemented filtering of generated items based on forbidden words list (case-insensitive regex).
    *   Improved error logging with `traceback.format_exc()`.
*   Added `import re` to `app.py`.
*   Updated `PROJECTPLAN.md` to include and mark new Phase 1 tasks as complete.
*   Fixed bug: Refined item parsing logic in `generate_items` using regex (`re.match`) to reliably remove LLM-generated list markers and prevent double indexing in the UI display.

## [Date TBD] - Phase 2: Embedding Service Implementation

*   Created `src/` directory and `src/__init__.py`.
*   Created `src/embedding_service.py`:
    *   Implemented initial `get_embeddings` function to fetch dense and sparse embeddings from OpenAI (`text-embedding-3-small`).
    *   Added `joblib` caching (`@memory.cache`) to `_fetch_embeddings_from_api` to avoid redundant API calls.
    *   Refactored caching (`PicklingError` fix): Moved OpenAI client instantiation inside the cached function (`_fetch_embeddings_from_api`) to prevent passing the unpickleable client object as an argument.
*   Integrated into `app.py`:
    *   Added "Generate Embeddings" button and section.
    *   Used `st.session_state` to store `dense_embeddings` and `sparse_embeddings`.
    *   Added `st.metric` to display embedding status (shape/count).
    *   Updated "Clear History" to also clear embeddings.
*   Refactored prompting logic:
    *   Created `src/prompting.py`.
    *   Moved `DEFAULT_POSITIVE_EXAMPLES`, `DEFAULT_NEGATIVE_EXAMPLES`, and prompt/parsing/filtering functions (`create_system_prompt`, `create_user_prompt`, `parse_generated_items`, `filter_unique_items`) from `app.py` to `src/prompting.py`. (Note: `specific_prompts` remains local to `app.py`).
    *   Updated `app.py` imports to use `src.prompting`.
*   Revised embedding strategy based on API limitations:
    *   Discovered OpenAI API (`v1.x`) does not directly return sparse embeddings as initially assumed (or capability was removed/changed).
    *   Refactored `src/embedding_service.py`:
        *   Renamed `get_embeddings` to `get_dense_embeddings`, focusing only on OpenAI dense vectors.
        *   Added `get_sparse_embeddings_tfidf` function using `sklearn.feature_extraction.text.TfidfVectorizer` for local sparse embedding generation.
        *   Added `scikit-learn` and `scipy` imports.
    *   Updated `app.py`:
        *   Imported new embedding functions.
        *   Renamed session state to `sparse_embeddings_tfidf`.
        *   Split UI section 4 into "4.1 Dense Embeddings (OpenAI)" and "4.2 Sparse Embeddings (TF-IDF)" with separate buttons and status displays.
*   Updated `PROJECTPLAN.md` to reflect TF-IDF approach for sparse embeddings and marked relevant Phase 2 tasks as complete.

## [Date TBD] - Phase 3: EGA Foundation - Similarity Matrix

*   Created `src/ega_service.py`.
*   Implemented `calculate_similarity_matrix` function using `sklearn.metrics.pairwise.cosine_similarity` to handle both dense (NumPy) and sparse (SciPy) embeddings.
*   Added type hints, docstrings, basic error handling, and a `__main__` test block to `src/ega_service.py`.
*   Integrated `calculate_similarity_matrix` into `app.py`:
    *   Added session state variables (`similarity_matrix_dense`, `similarity_matrix_sparse`).
    *   Created a new UI Section 5 ("Calculate Similarity Matrix").
    *   Added buttons to trigger similarity calculation based on available embeddings.
    *   Added status display for calculated similarity matrices.
    *   Updated "Clear Item History" to reset similarity matrices.
*   Updated `PROJECTPLAN.md` to mark the similarity matrix calculation task as complete.

## [Date TBD] - Phase 3: TMFG Network Construction & Visualization

*   Implemented `construct_tmfg_network` function in `src/ega_service.py` using `networkx`.
    *   Included logic based on the iterative TMFG algorithm (Massara et al., 2016).
    *   Added docstrings, type hints, error handling, and tests to `__main__` block.
*   Integrated TMFG construction into `app.py` (Section 6):
    *   Added `networkx` and `matplotlib.pyplot` imports.
    *   Added session state for TMFG graphs (`tmfg_graph_dense`, `tmfg_graph_sparse`).
    *   Added UI elements (radio buttons for method/input, construct button).
    *   Implemented logic to call `construct_tmfg_network` based on selected similarity matrix.
    *   Displayed network status (node/edge count).
    *   Added basic network visualization using `networkx.draw` and `matplotlib.pyplot` within an expander.
    *   Updated "Clear Item History" to reset graph state.
*   Updated `PROJECTPLAN.md` to mark the TMFG constructor task as complete.

## [Date TBD] - Phase 3: EBICglasso Network Construction & Stability Fix

*   Implemented `construct_ebicglasso_network` function in `src/ega_service.py`.
    *   Used `sklearn.covariance.GraphicalLassoCV` to estimate sparse precision matrix.
    *   Set `assume_centered=True` for use with similarity matrices.
    *   Calculated partial correlations for edge weights.
    *   Added docstrings, type hints, error handling, and tests.
*   Integrated EBICglasso into `app.py` (Section 6):
    *   Added "EBICglasso" option to network method selection.
    *   Added session state for Glasso graphs (`glasso_graph_dense`, `glasso_graph_sparse`).
    *   Implemented logic to call `construct_ebicglasso_network`.
    *   Ensured status display and visualization work for EBICglasso.
    *   Updated "Clear Item History" to reset Glasso graph state.
*   Resolved persistent `FloatingPointError: Non SPD result` in `construct_ebicglasso_network`:
    *   Applied `sklearn.covariance.shrunk_covariance` (Ledoit-Wolf) to the input similarity matrix.
    *   Added scaled diagonal jitter (`epsilon * np.eye`, where `epsilon` is proportional to matrix std dev) after shrinkage to enforce positive-definiteness.
    *   Implemented a retry mechanism: if `GraphicalLassoCV.fit` fails with `FloatingPointError`, increase jitter (`epsilon *= 10`) and retry fitting once.
    *   Re-raised final errors as `RuntimeError` for better handling in the Streamlit UI.
*   Updated `PROJECTPLAN.md` to mark the EBICglasso implementation task as complete.

## [Date TBD] - Phase 3: Walktrap Community Detection & Visualization

*   Implemented `detect_communities_walktrap` function in `src/ega_service.py`:
    *   Converts `networkx.Graph` to `igraph.Graph`, handling node mapping and edge weights.
    *   Uses `igraph.Graph.community_walktrap()` to perform detection.
    *   Returns membership dictionary mapped back to original node IDs and the `igraph.VertexClustering` object.
    *   Added error handling for `igraph` import, missing weights, and conversion/detection errors.
    *   Included test cases in `__main__` block.
*   Integrated community detection into `app.py` (Section 6):
    *   Added session state variables for community membership and modularity for each graph type.
    *   Ensured community state is cleared by "Clear Item History".
    *   Added "Detect Communities" button, enabled only after a graph is constructed.
    *   Implemented button logic to call `detect_communities_walktrap`, store results in session state, and handle potential errors (`ImportError`, `KeyError`, `RuntimeError`).
    *   Updated network status display to include metrics for number of communities and modularity.
    *   Modified `matplotlib` network plot to color nodes based on detected community membership using a colormap (`viridis`).
    *   Added an optional legend to the plot showing community colors.
*   Fixed `ImportError` in `app.py` by removing incorrect import of `filter_forbidden_words` (logic is handled within `parse_generated_items`).
*   Updated `PROJECTPLAN.md` to mark Walktrap implementation and related UI/Plot tasks as complete.

## [Date TBD] - Phase 3: Walktrap Robustness & Visualization Enhancements

*   Refactored `detect_communities_walktrap` in `src/ega_service.py` to handle `igraph` error (`Vertex with zero strength found`) when using TMFG with sparse embeddings:
    *   Identifies nodes with zero strength (sum of edge weights).
    *   Runs Walktrap only on the subgraph containing positive-strength nodes.
    *   Assigns isolated/zero-strength nodes to community ID `-1`.
    *   Added `IGRAPH_AVAILABLE` flag check within `src/ega_service.py`.
*   Enhanced network visualization in `app.py`:
    *   Created `plot_network_with_communities` function.
    *   Nodes colored based on community membership (using `viridis` colormap).
    *   Isolated nodes (community `-1`) colored grey.
    *   Added a legend differentiating communities and isolated nodes.
*   Added UI toggle (radio buttons) in `app.py` to show/hide item numbers (node labels) on the network graph:
    *   Added `show_network_labels` to session state.
    *   Updated plotting functions (`plot_network_with_communities` and basic `nx.draw`) to respect the toggle.
    *   Labels show extracted item number (e.g., "1" instead of "Item 1") when visible.

## [Date TBD] - Phase 3 Completion: TEFI Calculation

*   Implemented `calculate_tefi` function in `src/ega_service.py`:
    *   Calculates a TEFI variant based on the standardized difference between average within-community and between-community similarities.
    *   Takes the similarity matrix and community membership dictionary as input.
    *   Handles potential errors and edge cases (e.g., < 2 communities, isolated nodes).
    *   Added basic test cases to `__main__` block.
*   Added placeholder `calculate_nmi` function in `src/ega_service.py`:
    *   Includes basic structure and checks for future implementation.
    *   Currently returns `np.nan` or raises `NotImplementedError` as NMI comparison requires a second clustering (handled in Phase 5/6).
*   Integrated TEFI calculation into `app.py` (Section 6):
    *   Added session state variables (`tefi_*_*`).
    *   `calculate_tefi` is called after successful community detection.
    *   Added a new metric display column for TEFI.
    *   NMI metric display shows "N/A" placeholder.
    *   Updated "Clear Item History" to reset TEFI state.

## [Date TBD] - App Workflow: Editable Item Pool & Confirmation Step

*   Refactored Section 3 of `app.py` to improve workflow flexibility and testing:
    *   Replaced the static display of generated items with an editable `st.text_area` (`item_pool_editor`) which serves as the primary item pool before confirmation.
    *   Added `item_pool_text` and `items_confirmed` to `st.session_state`.
    *   Modified "Generate New Items" button to append new unique items to the `item_pool_text` state, allowing incremental generation.
    *   Added `parse_items_from_text` helper function to reliably extract items from the text area, handling list markers and empty lines.
    *   Added a new primary button "Confirm & Use This Item Pool":
        *   Reads and parses the content of `item_pool_text` using `parse_items_from_text`.
        *   Sets `st.session_state.previous_items` to this final list.
        *   Sets `st.session_state.items_confirmed = True`.
        *   Clears all downstream state variables (embeddings, matrices, graphs, communities, metrics) to ensure subsequent steps use the confirmed items.
        *   Disables the text area and generation/clearing buttons once confirmed.
    *   Renamed "Clear Item History" to "Clear Item History & Start Over" and updated it to reset `item_pool_text` and `items_confirmed` state as well.
    *   Updated subsequent sections (4, 5, 6) to be conditional on `st.session_state.items_confirmed == True`.

## [Date TBD] - Phase 4: Initial UVA Implementation

*   Implemented `calculate_wto` function in `src/ega_service.py`:
    *   Calculates the Weighted Topological Overlap matrix based on the standard formula using a similarity/adjacency matrix.
    *   Added type hints, docstrings, and basic tests.
*   Implemented `remove_redundant_items_uva` function in `src/ega_service.py`:
    *   Takes an initial similarity matrix and item labels as input.
    *   Iteratively calculates wTO (using `calculate_wto`) on the current subset of the *initial similarity matrix*.
    *   Removes the most redundant item (highest wTO, tie-breaking with lowest sum similarity) if wTO exceeds a threshold.
    *   Returns the list of remaining items and a log of removed items.
    *   Added type hints, docstrings, and basic tests.
*   Integrated UVA into `app.py` (New Section 7):
    *   Added necessary imports and session state variables (`uva_*`).
    *   Added UI elements: Section header, slider for wTO threshold (default 0.20), button to run UVA.
    *   Implemented button logic to call `remove_redundant_items_uva` using the similarity matrix selected in Section 6 (EGA).
    *   Added display for UVA results (metrics for removed/remaining items, table of removed items, list of final items).
    *   Made subsequent sections (8: bootEGA, 9: Export) conditional on UVA completion.
    *   Updated "Clear Item History & Start Over" button to reset UVA state.
*   Fixed `AttributeError` in `app.py` by re-adding missing initialization for `show_network_labels` in session state.

## [Date TBD] - Phase 4: UVA Refinement & Bug Fixes

*   Refactored `remove_redundant_items_uva` in `src/ega_service.py`:
    *   Changed function signature to accept `graph: nx.Graph` and `graph_type: str` instead of `initial_similarity_matrix: np.ndarray`.
    *   Modified internal logic to calculate wTO based on the adjacency matrix derived from the *current subgraph* within the iterative loop, using absolute edge weights.
    *   Updated tie-breaking logic to use the sum of absolute connection strengths from the current subgraph.
    *   Added new test cases for the refactored function in the `__main__` block.
*   Updated Section 7 (UVA) in `app.py`:
    *   Modified the call to `remove_redundant_items_uva` to pass the selected `networkx.Graph` (TMFG or EBICglasso from Section 6 state) and the corresponding `graph_type` string ('tmfg' or 'glasso').
    *   Corrected logic to read the selected network method and input type from the correct session state keys (`network_method_select`, `input_matrix_select`) set by Section 6's radio buttons, ensuring the UVA input description is accurate.
*   Fixed `ImportError: cannot import name 'generate_items'` in `app.py`:
    *   Removed `generate_items` from the `src.prompting` import list.
    *   Restored the `generate_items` function definition within `app.py` (it was incorrectly removed in a previous cleanup).
*   Fixed `StreamlitDuplicateElementId` error in `app.py` by adding `key="focus_area_selectbox"` to the focus area `st.selectbox` in Section 1.
*   Updated `PROJECTPLAN.md` to mark Phase 4 UVA refactoring tasks as complete.

## [Date TBD] - Phase 5: bootEGA Core Logic & Integration

*   Implemented core bootEGA functions in `src/ega_service.py`:
    *   `_run_bootstrap_single`: Helper for parallel execution, performs one bootstrap sample EGA run.
    *   `run_bootega_resampling`: Manages parallel/sequential execution of `_run_bootstrap_single` for N bootstraps, includes progress reporting.
    *   `perform_bootega_stability_analysis`: Orchestrates the iterative stability analysis, calling resampling, calculating stability, removing unstable items, and returning final results.
*   Integrated bootEGA into `app.py` (Section 8):
    *   Added UI inputs for N bootstraps, stability threshold, and parallel processing toggle.
    *   Added "Run bootEGA" button logic to call `perform_bootega_stability_analysis`.
    *   Included input validation and preparation steps (fetching required data like post-UVA items, embeddings, original communities, initial NMI).
    *   Implemented progress bar using `st.progress` and a callback function (`update_progress`).
    *   Added logic to run final EGA on stable items and calculate final NMI.
    *   Stored results (stable items, stability scores, removed log, NMI values) in session state.
*   Applied several fixes based on testing and feedback:
    *   Corrected `detect_communities_walktrap` calls to remove invalid `item_labels` argument.
    *   Standardized bootEGA session state keys (`bootega_removed_log`, `bootega_final_stability_scores`).
    *   Fixed `TypeError` related to `callable` type hint by importing and using `collections.abc.Callable`.
    *   Cleaned up `igraph` import logic and removed unused imports in `ega_service.py`.
    *   Corrected placement of `st.rerun()` in `app.py` bootEGA handler to ensure analysis runs before rerun.
    *   Fixed undefined `update_progress` callback and `initial_nmi` variable in `app.py`.
    *   Corrected session state key construction for fetching original community membership in `app.py`.
*   **Note:** UI elements for displaying the detailed bootEGA *results* (stability scores, removed items list, NMI comparison) are implemented in the backend logic storage but not yet added to the Streamlit interface as per `PROJECTPLAN.md`.
