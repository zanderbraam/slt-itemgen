import streamlit as st
# import openai  # Uncomment this when I have the API key
# from dotenv import load_dotenv  # If you're using dotenv for environment variables


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
        generate_questions(selected_option)


def generate_questions(selected_option):
    # Placeholder for the actual API call
    # Replace this code with OpenAI API call when I have the API key

    # Dummy questions for demonstration purposes
    dummy_questions = [
        f"Question 1 about {selected_option}",
        f"Question 2 about {selected_option}",
        f"Question 3 about {selected_option}",
        f"Question 4 about {selected_option}",
        f"Question 5 about {selected_option}",
        f"Question 6 about {selected_option}",
        f"Question 7 about {selected_option}",
        f"Question 8 about {selected_option}",
        f"Question 9 about {selected_option}",
        f"Question 10 about {selected_option}",
    ]

    st.subheader("Generated Questions:")
    for question in dummy_questions:
        st.write("- " + question)


if __name__ == "__main__":
    main()
