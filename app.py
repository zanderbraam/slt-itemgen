import os
import toml
import re
import traceback

import streamlit as st
from openai import OpenAI, APIError

from src.prompting import (
    DEFAULT_NEGATIVE_EXAMPLES,
    DEFAULT_POSITIVE_EXAMPLES,
    create_system_prompt,
    create_user_prompt,
    parse_generated_items,
    filter_unique_items,
)
from src.embedding_service import get_dense_embeddings, get_sparse_embeddings_tfidf


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
