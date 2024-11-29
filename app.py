import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Options and specific prompts
options = [
    "Engagement in a communicative activity",
    "Motivation during a communicative activity",
    "Persistence in a communicative activity",
    "Social connection in a communicative activity",
    "Sense of belonging in a communicative activity",
    "Affect during a communicative activity"
]

specific_prompts = {
    "Engagement in a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ engagement in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Motivation during a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ motivation during "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Persistence in a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ persistence in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Social connection in a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ social connection in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Sense of belonging in a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ sense of belonging in "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    ),
    "Affect during a communicative activity": (
        "Generate 10 items to measure a child with communication difficulties’ affect during "
        "a communicative activity where knowledge, information, ideas, and feelings are exchanged."
    )
}


def main():
    st.title("SLTItemGen: Communicative Participation Item Generator")
    st.write("""
        Generate custom items aimed at measuring the communicative participation outcomes of children with communication difficulties.
        """)

    selected_option = st.selectbox("Select a focus area for the items:", options)

    temperature = st.slider(
        "Select model creativity (temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,  # Default value
        step=0.1,
        help=(
            "Temperature controls the creativity of the generated items. "
            "Lower values (e.g., 0.0) make the output more focused and deterministic, "
            "while higher values (e.g., 1.0) increase creativity and randomness."
        ),
    )

    if st.button("Generate"):
        with st.spinner('Generating items...'):
            items = generate_questions(selected_option, temperature)
            if items:
                st.subheader("Generated Items:")
                for item in items:
                    st.write("- " + item)
            else:
                st.error("An error occurred while generating items. Please try again.")


def generate_questions(selected_option, temperature):
    try:
        # Retrieve the specific prompt based on the selected option
        specific_prompt = specific_prompts[selected_option]

        # Background information
        background_info = (
            "The aim is to develop an instrument that measures the communicative participation outcomes "
            "of children with communication difficulties between 6 and 11 years old.\n"
            "Communicative participation is participation in life situations where knowledge, "
            "information, ideas and feelings are exchanged. It also means understanding and being understood "
            "in a social context by applying verbal and non-verbal communication skills.\n\n"
            "Communicative participation outcomes involve children with communication difficulties participating "
            "in situations where they understand and are understood in social contexts using verbal or nonverbal "
            "communication skills. This may include participating in a group activity at school, playing cards with friends, "
            "initiating a conversation with a friend or family member, being involved in school and community activities, "
            "and engaging in play with others. The communicative participation outcomes included in this measure will be used "
            "by speech-language therapists (SLTs) and parents of children with communication difficulties. It is intended that "
            "Speech Language Therapists use this measure when assessing children with communication difficulties."
        )

        # Examples of relevant items
        relevant_items = (
            "Examples of relevant items:\n"
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

        # Examples of non-relevant items
        non_relevant_items = (
            "Examples of non-relevant items:\n"
            "- How well does your child communicate with peers during playtime or group activities?\n"
            "- To what degree does your child initiate and sustain conversations with friends?\n"
            "- How often does your child successfully understand and respond to social cues from peers?\n"
            "- How effectively does your child ask for help when needed in various settings?\n"
            "- To what extent does your child use communication strategies to express their feelings or concerns?\n"
            "- How often does your child use alternative communication methods (e.g., gestures, pictures) when verbal communication is challenging?"
        )

        # Construct the system prompt
        system_prompt = (
            "You are an expert in child speech and language therapy.\n\n"
            f"{background_info}\n\n"
            f"{relevant_items}\n\n"
            f"{non_relevant_items}\n\n"
            "Please generate items similar to the relevant examples and avoid items like the non-relevant examples. "
            "Only return the items!"
        )

        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": specific_prompt}
        ]

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with "gpt-3.5-turbo" if needed
            messages=messages,
            max_tokens=10000,
            n=1,
            temperature=temperature,
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content.strip()

        # Split the text into individual items
        items = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering or bullet points
                item = line.lstrip('0123456789.-) ').strip()
                items.append(item)
            elif line:
                items.append(line)

        # Ensure we have exactly 10 items
        if len(items) > 10:
            items = items[:10]
        elif len(items) < 10:
            st.warning("Less than 10 items were generated.")

        return items

    except Exception as e:
        # Handle exceptions (e.g., API errors)
        st.error(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
