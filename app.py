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


def main():
    st.title("QuestGen: Communicative Participation Question Generator")
    st.write("""
    Generate custom question sets aimed at measuring the communicative participation outcomes of children with communication difficulties.
    """)

    # Placeholder options
    options = [
        "Verbal Communication Skills",
        "Non-Verbal Communication Skills",
        "Social Interaction",
        "Understanding and Comprehension",
        "Initiating Conversations",
        "Participating in Group Activities"
    ]

    selected_option = st.selectbox("Select a focus area for the questions:", options)

    if st.button("Generate"):
        with st.spinner('Generating questions...'):
            questions = generate_questions(selected_option)
            if questions:
                st.subheader("Generated Questions:")
                for question in questions:
                    st.write("- " + question)
            else:
                st.error("An error occurred while generating questions. Please try again.")


def generate_questions(selected_option):
    try:
        # Construct the prompt
        prompt = (
            f"Generate 10 assessment questions aimed at measuring {selected_option.lower()} "
            f"for children aged 6 to 11 with communication difficulties. "
            "The questions should be appropriate for speech-language therapists and parents to use."
        )

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in child speech and language therapy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0.7,
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content.strip()

        # Split the text into individual questions
        questions = [q.strip() for q in generated_text.split('\n') if q.strip()]

        return questions

    except Exception as e:
        # Handle exceptions (e.g., API errors)
        st.error(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
