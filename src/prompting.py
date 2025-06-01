import re

# --- Default Examples ---

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

# --- Prompt Construction Functions ---

def create_system_prompt(
    persona: str,
    background_info: str,
    positive_examples: str | None,
    negative_examples: str | None,
    previous_items: list[str],
    forbidden_words_str: str | None,
    is_big_five_test: bool
) -> tuple[str, list[str]]:
    """Constructs the detailed system prompt for the LLM.

    Args:
        persona: The AI persona description.
        background_info: Context about the target domain and use case.
        positive_examples: Formatted string of positive examples.
        negative_examples: Formatted string of negative examples.
        previous_items: List of items generated previously in the session.
        forbidden_words_str: User-provided string of forbidden words (one per line).
        is_big_five_test: Flag indicating if the focus is Big Five.

    Returns:
        A tuple containing:
            - The fully constructed system prompt string.
            - A list of forbidden words derived from forbidden_words_str.
    """
    # Format Examples
    positive_examples_formatted = ""
    if positive_examples and not positive_examples.strip().lower().startswith("positive examples"):
        positive_examples_formatted = f"POSITIVE EXAMPLES (Items to emulate):\n{positive_examples.strip()}"
    elif positive_examples:
        positive_examples_formatted = positive_examples.strip()

    negative_examples_formatted = ""
    if negative_examples and not negative_examples.strip().lower().startswith("negative examples"):
        negative_examples_formatted = f"NEGATIVE EXAMPLES (Items to AVOID):\n{negative_examples.strip()}"
    elif negative_examples:
        negative_examples_formatted = negative_examples.strip()

    # Format Duplicate Avoidance Prompt
    duplicate_avoidance_prompt = ""
    if previous_items:
        items_str = "\n".join(f"- {item}" for item in previous_items)
        duplicate_avoidance_prompt = (
            "\nIMPORTANT: DO NOT repeat or closely rephrase any items from the following list of previously generated items:\n"
            f"{items_str}\n"
            "ENSURE all newly generated items are distinct from this list."
        )

    # Format Forbidden Words Prompt
    forbidden_words_prompt = ""
    forbidden_words_list = []
    if forbidden_words_str:
        forbidden_words_list = [word.strip() for word in forbidden_words_str.split('\n') if word.strip()]
        if forbidden_words_list:
            words_str = ", ".join(f"'{word}'" for word in forbidden_words_list)
            forbidden_words_prompt = f"\nCRITICAL: DO NOT use any of the following words in your generated items: {words_str}."

    # Construct the System Prompt parts
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
    return system_prompt, forbidden_words_list


def create_user_prompt(prompt_focus: str, n: int, specific_prompts: dict[str, str]) -> str:
    """Creates the user prompt instruction based on the selected focus.

    Args:
        prompt_focus: The key for the specific prompt template.
        n: The number of items to request.
        specific_prompts: The dictionary mapping focus keys to prompt templates.

    Returns:
        The formatted user prompt string.

    Raises:
        ValueError: If prompt_focus is not in specific_prompts.
    """
    if prompt_focus not in specific_prompts:
        raise ValueError(f"Invalid prompt focus: {prompt_focus}")
    return specific_prompts[prompt_focus].format(n=n)


# --- Item Parsing and Filtering ---

def parse_generated_items(generated_text: str, forbidden_words_list: list[str]) -> list[str]:
    """Parses the raw text output from the LLM into a list of items, filtering forbidden words.

    Args:
        generated_text: The raw string response from the LLM.
        forbidden_words_list: A list of words to filter out.

    Returns:
        A list of parsed and filtered item strings.
    """
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
    return items


def filter_unique_items(items: list[str], previous_items: list[str]) -> list[str]:
    """Filters a list of items to include only those not present in previous_items.

    Args:
        items: The list of newly generated items.
        previous_items: The list of items generated in previous runs.

    Returns:
        A list containing only the unique new items.
    """
    new_unique_items = []
    seen_items = set(previous_items) # Use set for efficient lookup
    for item in items:
        if item not in seen_items:
            new_unique_items.append(item)
            seen_items.add(item) # Add to seen immediately to handle duplicates within the new list
    return new_unique_items
