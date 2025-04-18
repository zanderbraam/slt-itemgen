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
    *   Moved `DEFAULT_POSITIVE_EXAMPLES`, `DEFAULT_NEGATIVE_EXAMPLES`, and prompt/parsing/filtering functions from `app.py` to `src/prompting.py`.
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