import os
import toml
import re

import streamlit as st
from openai import OpenAI


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
DEFAULT_POSITIVE_EXAMPLES = (
    "POSITIVE EXAMPLES (Items to emulate):\n"
    "- The child seems more motivated to engage in social interaction.\n"
    "- The child seems more willing to continue during a social interaction.\n"
    "- The child shows intent to continue attending social interaction.\n"
    "- The child seems willing to be challenged during social interactions.\n"
    "- The child seems willing to use (or not use) AAC to communicate with others in specific situations.\n"
    "- The child seems to be motivated to communicate.\n"
    "- The child initiates doing an activity with others.\n"
    "- The child seems interested in a social activity.\n"
    "- The child seems to persist in social interaction.\n"
    "- The child seems to show excitement during a communicative activity.\n"
    "- The child seems to portray positive body language.\n"
    "- The child seems to portray a positive energy level during a communicative activity.\n"
    "- The child is actively taking part in a communicative activity.\n"
    "- The child contributes to the discussion during a communicative activity.\n"
    "- The child is asking questions during a communicative activity.\n"
    "- The child is making suggestions during a communicative activity.\n"
    "- The child seems to be listening or observing attentively during a communicative activity.\n"
    "- The child seems to be taking the lead during a communicative activity.\n"
    "- The child expresses or is willing to try new things during a communicative activity.\n"
    "- The child seems to desire to communicate with others in specific situations."
)

DEFAULT_NEGATIVE_EXAMPLES = (
    "NEGATIVE EXAMPLES (Items to AVOID):\n"
    "- How well does your child communicate with peers during playtime or group activities? (Avoid direct questions to parent/SLT)\n"
    "- To what degree does your child initiate and sustain conversations with friends? (Avoid direct questions)\n"
    "- How often does your child successfully understand and respond to social cues from peers? (Avoid frequency/success ratings)\n"
    "- How effectively does your child ask for help when needed in various settings? (Avoid effectiveness ratings)\n"
    "- To what extent does your child use communication strategies to express their feelings or concerns? (Avoid extent ratings)\n"
    "- How often does your child use alternative communication methods (e.g., gestures, pictures) when verbal communication is challenging? (Avoid frequency/method focus)"
)

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

    # --- Basic Controls ---
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

    # --- Advanced Prompting Options ---
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

    # --- Generation and Display ---
    st.header("3. Generate and View Items")

    # Initialize session state for generated items
    if 'previous_items' not in st.session_state:
        st.session_state.previous_items = []

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
                        st.error("No new unique items were generated. Check logs or try adjusting parameters.")

    with col2:
        if st.button("Clear Item History"):
            st.session_state.previous_items = []
            st.success("Item history cleared.")
            st.rerun()

    # Display generated items
    st.subheader("Generated Item Pool (Cumulative)")
    if st.session_state.previous_items:
        unique_items = list(dict.fromkeys(st.session_state.previous_items)) # Maintain order, ensure unique
        st.write(f"Total unique items: {len(unique_items)}")
        # Use a text area for better copy-paste
        st.text_area("Items:", "\n".join(f"{i+1}. {item}" for i, item in enumerate(unique_items)), height=300)
    else:
        st.info("No items generated yet in this session.")


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
        # 1. Retrieve base user instruction prompt
        if prompt_focus not in specific_prompts:
            raise ValueError(f"Invalid prompt focus: {prompt_focus}")
        user_prompt_instruction = specific_prompts[prompt_focus].format(n=n)

        # 2. Determine Persona and Background Info based on focus
        # TODO: Refine this - Big Five might need a different persona/background
        is_big_five_test = prompt_focus.startswith("TEST: Big Five")
        if is_big_five_test:
            persona = "You are an expert psychometrician and test developer specializing in personality assessment."
            background_info = "TARGET DOMAIN: Big Five Personality Traits (Adult Self-Report)"
            # Use provided examples or omit if none provided for Big Five
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
            # Use user examples if provided, otherwise use defaults
            positive_examples_to_use = positive_examples or DEFAULT_POSITIVE_EXAMPLES
            negative_examples_to_use = negative_examples or DEFAULT_NEGATIVE_EXAMPLES

        # 3. Format Examples
        # Add headers if user provided examples without them
        if positive_examples_to_use and not positive_examples_to_use.strip().lower().startswith("positive examples"):
            positive_examples_formatted = f"POSITIVE EXAMPLES (Items to emulate):\n{positive_examples_to_use.strip()}"
        else:
            positive_examples_formatted = positive_examples_to_use.strip()

        if negative_examples_to_use and not negative_examples_to_use.strip().lower().startswith("negative examples"):
            negative_examples_formatted = f"NEGATIVE EXAMPLES (Items to AVOID):\n{negative_examples_to_use.strip()}"
        else:
            negative_examples_formatted = negative_examples_to_use.strip()

        # 4. Format Duplicate Avoidance Prompt
        duplicate_avoidance_prompt = ""
        if previous_items:
            items_str = "\n".join(f"- {item}" for item in previous_items)
            duplicate_avoidance_prompt = (
                "\nIMPORTANT: DO NOT repeat or closely rephrase any items from the following list of previously generated items:\n"
                f"{items_str}\n"
                "ENSURE all newly generated items are distinct from this list."
            )

        # 5. Format Forbidden Words Prompt
        forbidden_words_prompt = ""
        forbidden_words_list = []
        if forbidden_words_str:
            forbidden_words_list = [word.strip() for word in forbidden_words_str.split('\n') if word.strip()]
            if forbidden_words_list:
                words_str = ", ".join(f"'{word}'" for word in forbidden_words_list)
                forbidden_words_prompt = f"\nCRITICAL: DO NOT use any of the following words in your generated items: {words_str}."

        # 6. Construct the System Prompt
        system_prompt_parts = [
            persona,
            "Your task is to generate high-quality psychometric items based on the user request.",
            f"\nBACKGROUND INFO:\n{background_info}"
        ]
        if positive_examples_formatted:
            system_prompt_parts.append(f"\n{positive_examples_formatted}")
        if negative_examples_formatted:
            system_prompt_parts.append(f"\n{negative_examples_formatted}")

        system_prompt_parts.append(duplicate_avoidance_prompt) # Add even if empty for structure
        system_prompt_parts.append(forbidden_words_prompt) # Add even if empty

        # Add general instructions based on focus area
        if is_big_five_test:
            system_prompt_parts.append(
                "\nINSTRUCTIONS:\n"
                "1. FOLLOW the user's request to generate items for the specified Big Five trait.\n"
                "2. Generate items suitable for general adult self-report (e.g., 'I often feel...', 'I am the life...').\n"
                "3. Maintain a neutral tone.\n"
                "4. Adhere strictly to the duplicate avoidance and forbidden words instructions.\n"
                "5. OUTPUT ONLY the generated items as a numbered list (e.g., '1. Item text\\n2. Item text'). DO NOT include any other text, preamble, or explanation."
            )
        else: # Communicative Participation
            system_prompt_parts.append(
                "\nINSTRUCTIONS:\n"
                "1. FOLLOW the user's request to generate items for the specified focus area.\n"
                "2. Generate items that are OBSERVATIONAL statements about a child (e.g., 'The child seems...', 'The child initiates...').\n"
                "3. Ensure items are appropriate for children aged 6-11 with communication difficulties.\n"
                "4. Maintain a neutral, objective tone.\n"
                "5. AVOID direct questions, frequency ratings, or effectiveness judgments (see NEGATIVE EXAMPLES if provided).\n"
                "6. Adhere strictly to the duplicate avoidance and forbidden words instructions.\n"
                "7. OUTPUT ONLY the generated items as a numbered list (e.g., '1. Item text\\n2. Item text'). DO NOT include any other text, preamble, or explanation."
            )

        system_prompt = "\n".join(system_prompt_parts)

        # 7. Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_instruction}
        ]

        # Log the prompt for debugging if needed (optional)
        # print("--- SYSTEM PROMPT ---")
        # print(system_prompt)
        # print("--- USER PROMPT ---")
        # print(user_prompt_instruction)

        # 8. Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max(200, 20 * n), # Adjusted token estimation
            n=1,
            temperature=temperature,
            stop=None
        )

        # 9. Extract and process the generated text
        generated_text = response.choices[0].message.content.strip()
        items = []
        potential_items = generated_text.split('\n')
        for line in potential_items:
            line = line.strip()
            if not line: continue # Skip empty lines

            # Use regex to remove common list markers and capture the rest
            match = re.match(r'^\s*[\d\.\-\*\)]+\s*(.*)', line)
            if match:
                item_text = match.group(1).strip()
            else:
                item_text = line # Assume no list marker if regex doesn't match

            if not item_text: continue # Skip if only marker was present

            # Filter based on forbidden words (case-insensitive check)
            is_forbidden = False
            if forbidden_words_list:
                for forbidden_word in forbidden_words_list:
                    # Use word boundaries to avoid partial matches within words
                    if re.search(r'\b' + re.escape(forbidden_word) + r'\b', item_text, re.IGNORECASE):
                        is_forbidden = True
                        # print(f"Filtered item due to forbidden word '{forbidden_word}': {item_text}") # Optional debug print
                        break

            if not is_forbidden:
                items.append(item_text)

        # 10. Filter out duplicates relative to previous items
        new_unique_items = []
        seen_items = set(previous_items) # Use set for efficient lookup
        for item in items:
            if item not in seen_items:
                new_unique_items.append(item)
                seen_items.add(item)

        return new_unique_items

    except Exception as e:
        st.error(f"An error occurred during item generation: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return []


if __name__ == "__main__":
    main()
