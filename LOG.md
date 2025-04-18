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