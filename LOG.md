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

## [Date TBD] - Phase 5: bootEGA Results Display

*   Implemented UI elements in `app.py` (Section 8) to display the results of the bootEGA analysis after successful completion:
    *   Added a "bootEGA Results" subheader.
    *   Display summary metrics using `st.metric` in columns (Items Before, Items Removed, Items Stable, Initial vs. Final NMI).
    *   Display the log of removed items using `st.dataframe`.
    *   Display the final stability scores for remaining items using `st.dataframe`.
    *   Display the final list of stable items using a disabled `st.text_area`.
*   Added conditional logic to show results only when `bootega_status` is "Completed", display an error message if "Error", and show an informational message otherwise.
*   Updated `PROJECTPLAN.md` to mark Phase 5 UI display tasks as complete.

## [Date TBD] - Top P Parameter Addition

*   Enhanced item generation controls in `app.py` (Section 1):
    *   Added "Top P (nucleus sampling)" slider with range 0.0-1.0, default value 1.0, step 0.05.
    *   Added descriptive help text explaining nucleus sampling functionality.
    *   Positioned slider immediately after the temperature slider for logical parameter grouping.
*   Updated `generate_items` function signature in `app.py`:
    *   Added `top_p: float` parameter after temperature in function signature.
    *   Updated docstring to document the new top_p parameter and its range.
    *   Modified OpenAI API call (`client.chat.completions.create`) to include `top_p=top_p` parameter.
*   Updated item generation button logic in `app.py` (Section 3):
    *   Modified call to `generate_items()` to pass the `top_p` slider value.
*   This enhancement provides users with additional control over the diversity and focus of generated items through nucleus sampling, complementing the existing temperature parameter for comprehensive generation tuning.

## [Date TBD] - Quality of Life Improvements: NMI Calculation & Item Text Display

*   **Fixed NMI calculation in EGA Section (Section 6)**:
    *   Replaced placeholder `np.nan` assignment with proper NMI calculation comparing detected communities against a baseline.
    *   Implemented baseline comparison where each item is assigned to its own community (maximum separation baseline).
    *   Added proper error handling with fallback to `np.nan` if NMI calculation fails.
    *   Users now see actual NMI values instead of persistent "N/A" in the Fit Metrics display.

*   **Enhanced item text display throughout UVA and bootEGA sections**:
    *   Added `get_item_text_from_label()` helper function to map "Item X" labels back to actual item text using regex parsing and index lookup.
    *   Added `format_items_with_text()` helper function to format item lists with actual text for user-friendly display.
    *   **UVA Results (Section 7)**:
        *   "Items Removed by UVA" table now shows actual item text instead of "Item X" labels.
        *   "Final Item Pool after UVA" text area displays actual item sentences instead of "1. Item 48" format.
    *   **bootEGA Results (Section 8)**:
        *   "Items Removed During bootEGA Iterations" table shows actual item text with proper formatting.
        *   "Final Stability Scores for Stable Items" table displays actual item text instead of labels.
        *   "Final Stable Item Pool" text area shows numbered list with actual item sentences.

*   **Benefits**: These improvements significantly enhance user experience by showing meaningful item content instead of opaque labels, making results more interpretable and actionable for researchers and practitioners.

## [Date TBD] - Phase 6.1: CSV Export Implementation

*   **Implemented comprehensive CSV export functionality in Section 9**:
    *   Added three specialized CSV generation functions with full error handling and validation.
    *   Created dynamic export interface that adapts based on analysis completion status.
    *   Integrated actual item text mapping throughout all export functions.

*   **CSV Export Functions**:
    *   `generate_final_items_csv()`: Complete dataset with stable items, community assignments, stability scores, retention status, and metadata.
    *   `generate_analysis_summary_csv()`: Pipeline metrics (TEFI, NMI, counts), configuration parameters, and key statistics in structured format.
    *   `generate_removed_items_csv()`: Detailed log of items filtered during UVA and bootEGA stages with removal reasons and scores.

*   **Export Interface Features**:
    *   **Full Pipeline Export (after bootEGA)**: Three-column layout with individual CSV downloads, data previews, and record counts.
    *   **Partial Export (after UVA only)**: Basic CSV with UVA results to support intermediate workflow needs.
    *   **Smart File Naming**: Automatic timestamping (e.g., `final_items_20241218_1430.csv`) for version control.
    *   **Data Validation**: Comprehensive error handling with user-friendly error messages for edge cases.

*   **Key CSV Features**:
    *   **Final Items CSV**: 12 columns including `Item_Text`, `Community_ID`, `Stability_Score`, `UVA_Retained`, `bootEGA_Retained`, `Embedding_Method`, etc.
    *   **Analysis Summary CSV**: 18 key metrics in name-value pairs covering item counts, network metrics, parameters, and configuration.
    *   **Removed Items CSV**: Full traceability with `Removal_Stage`, `Removal_Reason`, `Score`, and `Iteration` for both UVA and bootEGA filtering.

*   **User Experience Enhancements**:
    *   Real-time data previews showing first few rows before download.
    *   Clear status indicators and download counts (e.g., "Preview: 43 total items").
    *   Intuitive emoji-based section headers and helpful captions.
    *   Graceful handling of edge cases (no stable items, missing data, etc.).

*   This completes the **High Priority** portion of Phase 6, providing researchers with production-ready CSV exports for further analysis, publication, and practical application.

## [Date TBD] - Critical Bug Fix: CSV Community ID Issue

**Issue:** Every item in the exported CSV file was incorrectly assigned `Community ID = -1`, regardless of actual community detection results.

**Root Cause Analysis:**
*   **Data Type Mismatch**: The `stable_items` variable contained node labels like `"Item 17"`, `"Item 18"` (created during network construction), while `st.session_state.previous_items` contained actual item texts like `"I speak clearly when I'm excited."`
*   **Faulty Index Mapping**: The code attempted to find item labels in actual item texts:
    ```python
    stable_indices = [st.session_state.previous_items.index(item) for item in stable_items if item in st.session_state.previous_items]
    ```
*   **Empty Results**: This always resulted in `stable_indices = []` because strings like `"Item 17"` are never found in actual item texts
*   **Cascade Failure**: Empty `stable_indices` caused the final EGA reconstruction to skip, leaving `bootega_final_community_membership = None` and defaulting all items to `Community ID = -1` in CSV export
*   **Duplicate Bug**: The same issue existed in `src/ega_service.py` in the `perform_bootega_stability_analysis` function

**Solution Implemented:**
1. **Added `label_to_index()` Helper Function** (in both `app.py` and `src/ega_service.py`):
   ```python
   def label_to_index(label: str) -> int:
       """Convert 'Item N' → N-1. Raise ValueError if pattern absent."""
       m = re.search(r'\bItem\s+(\d+)\b', label)
       if not m:
           raise ValueError(f"Cannot parse item label: {label!r}")
       return int(m.group(1)) - 1
   ```

2. **Fixed Index Mapping in bootEGA Final EGA** (`app.py` line ~1139):
   ```python
   # Before (broken):
   stable_indices = [st.session_state.previous_items.index(item) for item in stable_items if item in st.session_state.previous_items]
   
   # After (fixed):
   stable_indices = [label_to_index(lbl) for lbl in stable_items]
   ```

3. **Fixed Index Mapping in bootEGA Stability Analysis** (`src/ega_service.py` line ~1136):
   ```python
   # Before (broken):
   current_indices = [initial_items.index(item) for item in current_items if item in initial_items]
   
   # After (fixed):
   current_indices = [label_to_index(lbl) for lbl in current_items]
   ```

4. **Code Cleanup**: Removed all debug statements (`st.write("🔬 **DEBUG**...")` and `print("DEBUG...")`) from production code across:
   *   `app.py`: Community detection, bootEGA final EGA, CSV export functions
   *   `src/ega_service.py`: Walktrap detection, NMI calculation, bootstrap functions

**Testing Results:**
*   CSV export now correctly assigns community IDs (e.g., 0, 1, 2) instead of universal `-1`
*   Final EGA reconstruction runs successfully on stable items
*   Community sizes and stability scores populate correctly
*   No impact on other pipeline functionality

**Files Modified:**
*   `app.py`: Added `label_to_index()`, fixed 2 index mapping bugs, removed debug statements
*   `src/ega_service.py`: Added `label_to_index()`, fixed 1 index mapping bug, removed debug statements
*   `LOG.md`: Documented issue and resolution

This fix ensures the CSV export functionality works as intended, providing researchers with accurate community assignments for stable items after the full bootEGA pipeline.

## [Date TBD] - Quality of Life Feature: Reset Analysis Button

*   **Implemented global analysis reset functionality**:
    *   Added a new "Reset Analysis" section at the bottom of the page (after Section 9: Export Results).
    *   Created a prominent reset button with trash icon and clear descriptive text.
    *   Implemented two-stage confirmation dialog to prevent accidental data loss.

*   **User Experience Features**:
    *   **Initial Button**: Shows "🗑️ Reset Analysis" button with help text explaining it clears all data.
    *   **Confirmation Dialog**: When clicked, displays a warning with detailed list of what will be permanently cleared:
        *   All generated items
        *   All embeddings and similarity matrices
        *   All network graphs and community detections
        *   All UVA and bootEGA results
        *   All configuration settings
    *   **Final Confirmation**: Two buttons - "✅ Yes, Reset Everything" (primary) and "❌ Cancel" (secondary).

*   **Technical Implementation**:
    *   Added `reset_all_session_state()` function that completely clears all session state keys except temporary UI flags.
    *   Uses `st.session_state.show_reset_confirmation` flag to manage the confirmation dialog state.
    *   Properly handles state cleanup and page refresh with `st.rerun()`.
    *   When confirmed, displays success message and automatically restarts the analysis from Section 1.

*   **Safety Features**:
    *   Requires explicit two-step confirmation to prevent accidental resets.
    *   Clear warning messages explaining the irreversible nature of the action.
    *   Visual differentiation between confirmation and cancellation buttons.

*   **Benefits**: This feature allows users to quickly start over without having to manually clear individual sections or refresh the browser, improving workflow efficiency and user experience. Particularly useful during testing, experimentation, or when switching between different analysis configurations.

## [Date TBD] - Phase 6.2: Professional PDF Report Generation

*   **Implemented comprehensive PDF report generation system using ReportLab**:
    *   Created new `src/pdf_report.py` module with professional document templates and formatting.
    *   Added `reportlab==4.2.5` dependency to `requirements.txt` and installed successfully.
    *   Integrated PDF generation into Section 9 (Export Results) with user-friendly interface.

*   **PDF Report Architecture**:
    *   **Custom Template Class** (`SLTReportTemplate`): Professional document layout with headers, footers, and page numbering.
    *   **Advanced Styling System**: Custom paragraph styles with professional typography, colors, and spacing.
    *   **Modular Content Functions**: Separate functions for tables, charts, and different report sections.

*   **Report Structure & Content**:
    *   **Cover Page**: Analysis title, focus area, generation timestamp, and key metrics summary.
    *   **Executive Summary**: Analysis overview, key findings, retention rates, and network fit metrics.
    *   **Methodology**: Comprehensive 6-phase AI-GENIE pipeline explanation with detailed descriptions.
    *   **Generated Items Section**: Complete table of initial AI-generated items with formatting.
    *   **Network Analysis**: TMFG/EBICglasso results with embedded network visualizations and metrics.
    *   **UVA Results**: Unique Variable Analysis findings with removed items and wTO thresholds.
    *   **bootEGA Stability**: Bootstrap analysis results with stability scores and NMI comparisons.
    *   **Technical Appendix**: Configuration parameters, software versions, and reproducibility information.

*   **Visualization Integration**:
    *   **Network Plot Generation**: Created `generate_network_plot_for_pdf()` with publication-quality formatting.
    *   **High-Resolution Graphics**: 150 DPI PNG embeddings with proper aspect ratios and legends.
    *   **Community Color Mapping**: Consistent viridis colormap with proper legend for isolated nodes.
    *   **Automatic Plot Cleanup**: Memory management with `plt.close()` to prevent accumulation.

*   **Professional Formatting Features**:
    *   **Responsive Tables**: Auto-sized columns with alternating row colors and proper text wrapping.
    *   **Typography**: Helvetica font family with size hierarchy and proper spacing.
    *   **Color Scheme**: Professional dark blue headers with light blue accents and beige table backgrounds.
    *   **Page Layout**: Letter size with 1-inch margins and proper content flow.
    *   **Error Handling**: Graceful handling of missing data with "N/A" placeholders and descriptive messages.

*   **User Interface Enhancements**:
    *   **Smart Availability**: PDF generation enabled based on analysis completion status.
    *   **Preview Metrics**: Shows focus area, item counts, and analysis status before generation.
    *   **Progress Feedback**: Spinner during generation with clear success/error messages.
    *   **Timestamped Downloads**: Automatic filename generation with format `SLTItemGen_Analysis_Report_YYYYMMDD_HHMM.pdf`.
    *   **Comprehensive Error Handling**: Detailed error messages with expandable technical details.

*   **Technical Implementation Details**:
    *   **Session State Integration**: Seamless reading from all existing Streamlit session state variables.
    *   **Memory Efficiency**: Temporary file handling with automatic cleanup to prevent disk bloat.
    *   **Cross-Platform Compatibility**: Uses standard library modules for broad deployment compatibility.
    *   **Modular Design**: Separate functions for each report section allowing easy customization.

*   **Quality Assurance Features**:
    *   **Data Validation**: Checks for missing or incomplete analysis results with appropriate fallbacks.
    *   **Format Consistency**: Standardized number formatting (4 decimal places) and date/time stamps.
    *   **Content Adaptation**: Report sections automatically adapt based on available analysis results.
    *   **Professional Standards**: Follows academic report formatting conventions with proper citations and methodology disclosure.

*   **Updated PROJECTPLAN.md**: Marked all Phase 6 tasks as complete, moving project towards Phase 7 (Deployment & Polish).

*   **Benefits**: This completes the high-priority Phase 6 deliverable, providing researchers with publication-quality PDF reports that include all analysis results, visualizations, and technical details needed for academic work, grant applications, and clinical documentation. The professional formatting ensures reports are suitable for sharing with stakeholders, supervisors, and research collaborators.

## [Date TBD] - PDF Report Refinement: Network Plot Item Numbers

*   **Enhanced PDF Network Visualization**:
    *   Modified `generate_network_plot_for_pdf()` function to always display item numbers on network nodes.
    *   Implemented the same label extraction logic used in the main UI (extracts numbers from "Item X" format).
    *   Set font size to 10pt for optimal readability in PDF format.
    *   Ensures PDF reports provide clear item identification for easier cross-referencing with item tables.

*   **Technical Implementation**: Uses regex pattern `r'\d+$'` to extract item numbers from node labels, with fallback to full node name if pattern not found. Labels are always displayed regardless of the UI toggle state, ensuring consistent PDF output for professional documentation.

## [Date TBD] - Major Refactor: Shift to User-Driven Prompt System

**Context**: Following user feedback requesting removal of fixed dropdown list and implementation of freeform prompt capabilities for maximum assessment flexibility.

**Problem**: The existing system used hardcoded dropdown options (e.g., "Big Five", "Communicative Participation") which limited users to predefined assessment domains and prevented custom assessment development.

**Solution Implemented**: Complete architectural shift from preset topic model to user-driven prompt model while preserving all backend safeguards and functionality.

### **UI/UX Changes (Section 1)**:
*   **Removed** hardcoded dropdown menu with predefined options (`options` array and `specific_prompts` dictionary)
*   **Added** `st.text_input` for "Assessment Topic" (organizational/labeling purposes)
*   **Added** `st.text_area` for "Custom Prompt Instructions" (detailed user specifications)
*   **Enhanced** validation logic to require both topic and custom prompt before enabling generation
*   **Added** helpful placeholder examples and guidance text for user onboarding

### **Backend Architecture Changes**:
*   **Modified** `generate_items()` function signature:
    *   Replaced `prompt_focus: str` parameter with `topic: str` and `custom_prompt: str`
    *   Removed dependency on `specific_prompts` dictionary lookup
    *   Added intelligent persona detection based on prompt content analysis
*   **Implemented** dynamic persona selection logic:
    *   Child-focused assessments: Detects keywords like "child", "children", "communication"
    *   Adult personality assessments: Detects "personality", "big five", "adult", "self-report"
    *   General psychometric assessment: Default fallback for other domains
*   **Updated** user prompt construction to use custom instructions directly instead of template substitution

### **Prompt Engineering Enhancements**:
*   **Enhanced** `src/prompting.py` with `is_child_focused` parameter for better instruction customization
*   **Added** three-tier instruction system:
    *   Legacy Big Five handling (deprecated but maintained for compatibility)
    *   Child-focused assessments with observational statement requirements
    *   General psychometric assessment with flexible population targeting
*   **Maintained** all existing safeguards:
    *   Duplicate avoidance using `previous_items`
    *   Forbidden words filtering with regex boundary matching
    *   Professional formatting constraints and output structure requirements

### **Export/Reporting System Updates**:
*   **Updated** all session state references from `focus_area_selectbox` to `assessment_topic`:
    *   `app.py`: PDF report preview section
    *   `src/export.py`: Analysis summary CSV generation
    *   `src/pdf_report.py`: Cover page and technical appendix sections
*   **Ensured** backward compatibility with existing analysis pipeline and export functionality

### **Quality Assurance Features**:
*   **Preserved** all core functionality:
    *   Temperature and Top P parameter controls
    *   Optional positive/negative examples and forbidden words
    *   Advanced prompting options in expandable section
    *   Complete 6-phase AI-GENIE pipeline compatibility
*   **Enhanced** user guidance with validation messages and capability hints
*   **Maintained** professional assessment standards and output quality controls

### **Key Benefits Achieved**:
1. **Maximum Flexibility**: Users can now assess any domain with custom instructions without code modifications
2. **Intelligent Adaptation**: System automatically detects assessment type and adapts persona/instructions appropriately  
3. **Professional Standards**: All quality controls, formatting requirements, and psychometric best practices preserved
4. **Seamless Integration**: Zero impact on downstream analysis phases (EGA, UVA, bootEGA, export/reporting)
5. **Enhanced UX**: Clear guidance, validation feedback, and intuitive workflow progression

### **Example Usage Scenarios Enabled**:
*   *"Generate items to measure social anxiety in adolescents during peer interactions"*
*   *"Create assessment items for reading comprehension difficulties in elementary students"*  
*   *"Develop items measuring workplace communication effectiveness for adults with autism"*
*   *"Generate observational items for executive function skills in preschoolers"*

### **Technical Implementation Notes**:
*   User prompt format: `f"Generate {n} items based on the following instructions:\n\n{custom_prompt.strip()}"`
*   Persona detection uses keyword matching with `any(keyword in custom_prompt.lower() for keyword in [...])`
*   Background info dynamically incorporates user-provided topic: `f"TARGET DOMAIN: {topic}\n"`
*   Backward compatibility maintained through `is_big_five_test=False` parameter in `create_system_prompt()`

### **Files Modified**:
*   `app.py`: Main UI refactor, generate_items() function signature and logic
*   `src/prompting.py`: Enhanced create_system_prompt() with is_child_focused parameter and three-tier instructions
*   `src/export.py`: Updated session state key references for topic retrieval
*   `src/pdf_report.py`: Updated topic references in report generation

**Result**: Successfully transformed SLTItemGen from a domain-specific tool to a flexible, general-purpose psychometric item generation platform while maintaining all professional standards and analytical capabilities. Users can now develop assessment instruments for any domain using custom instructions, with the system intelligently adapting its approach based on content analysis.
