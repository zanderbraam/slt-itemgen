import os
import numpy as np
import joblib
from openai import OpenAI, APIError
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Initialize joblib memory for caching
# Note: Cache persistence is limited by Streamlit Cloud's ephemeral filesystem.
CACHE_DIR = "./cache/joblib_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

# --- Dense Embedding Logic (OpenAI - Cached) ---

@memory.cache
def _fetch_dense_embeddings_from_api(
    items: tuple[str], model: str = "text-embedding-3-small"
) -> np.ndarray | None:
    """Fetches dense embeddings from OpenAI API.

    This function is cached using joblib. The OpenAI client is instantiated
    within the function only on cache miss.

    Args:
        items: A tuple of strings (items) to embed.
        model: The embedding model to use.

    Returns:
        A numpy array of dense embeddings (N x D), or None if an error occurs.
    """
    try:
        client = OpenAI()
        print(">>> Making ACTUAL API call to OpenAI...")
        response = client.embeddings.create(
            input=list(items),
            model=model,
            dimensions=1536
        )

        dense_embeddings = []
        if response.data:
            # Sort based on index to ensure order matches input items
            sorted_data = sorted(response.data, key=lambda x: x.index)
            for embedding_object in sorted_data:
                if embedding_object.embedding:
                    dense_embeddings.append(embedding_object.embedding)
                else:
                    # This case should be less likely if the API call succeeded
                    print(f"Warning: Missing dense embedding for item index {embedding_object.index}")
                    # Handle missing embedding, e.g., return None or raise error
                    return None # Or handle appropriately

        if len(dense_embeddings) != len(items):
             print(f"Warning: Number of embeddings received ({len(dense_embeddings)}) does not match number of items ({len(items)}).")
             # Decide how to handle this - returning None for now
             return None

        return np.array(dense_embeddings)

    except APIError as e:
        print(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during dense embedding generation: {e}")
        traceback.print_exc()
        return None


# --- Public Interface for Dense Embeddings ---

def get_dense_embeddings(
    items: list[str]
) -> np.ndarray | None:
    """Gets dense embeddings for a list of items from OpenAI, using caching.

    Relies on the OPENAI_API_KEY environment variable being set.

    Args:
        items: A list of strings (items) to embed.

    Returns:
        A numpy array of dense embeddings (N x D), or None if an error occurs.
    """
    if not items:
        return None

    items_tuple = tuple(items)
    dense_embeddings = _fetch_dense_embeddings_from_api(items_tuple)
    return dense_embeddings

# --- Sparse Embedding Logic (TF-IDF - Local) ---

def get_sparse_embeddings_tfidf(
    items: list[str]
) -> csr_matrix | None:
    """Generates sparse embeddings (TF-IDF) for a list of items locally.

    Args:
        items: A list of strings (items) to embed.

    Returns:
        A scipy csr_matrix representing the TF-IDF sparse embeddings, or None if an error occurs.
    """
    if not items:
        return None
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(items)
        # The fit_transform returns a scipy.sparse.csr_matrix which is suitable
        return tfidf_matrix
    except Exception as e:
        print(f"An error occurred during TF-IDF sparse embedding generation: {e}")
        traceback.print_exc()
        return None


# Example Usage (Optional: Can be run directly for testing)
if __name__ == "__main__":
    test_items = [
        "How often do you feel overwhelmed by your responsibilities?",
        "Do you enjoy meeting new people?",
        "How easily do you get stressed in challenging situations?",
        "Are you the life of the party?",
        "Do you worry about things that might go wrong?",
    ]

    print("--- Testing Dense Embeddings (OpenAI) ---")
    try:
        print(f"Requesting dense embeddings for {len(test_items)} items...")
        dense = get_dense_embeddings(test_items)

        if dense is not None:
            print(f"Dense embeddings shape: {dense.shape}")
        else:
            print("Failed to retrieve dense embeddings.")

        # Test caching
        print("\nCalling get_dense_embeddings again (should use cache)...")
        dense2 = get_dense_embeddings(test_items)
        if dense2 is not None:
            print(f"Cached Dense embeddings shape: {dense2.shape}")

    except ImportError:
        print("Please install OpenAI library: pip install openai")
    except Exception as e:
        print(f"An error occurred in the dense embedding example: {e}")
        if "api_key" in str(e).lower():
             print("Ensure the OPENAI_API_KEY environment variable is set.")
        else:
            traceback.print_exc()

    print("\n--- Testing Sparse Embeddings (TF-IDF) ---")
    try:
        print(f"Generating TF-IDF sparse embeddings for {len(test_items)} items...")
        sparse_tfidf = get_sparse_embeddings_tfidf(test_items)

        if sparse_tfidf is not None:
            print(f"TF-IDF sparse matrix shape: {sparse_tfidf.shape}")
            print(f"TF-IDF matrix type: {type(sparse_tfidf)}")
        else:
            print("Failed to generate TF-IDF sparse embeddings.")

    except ImportError:
        print("Please install scikit-learn: pip install scikit-learn scipy")
    except Exception as e:
        print(f"An error occurred in the sparse embedding example: {e}")
        traceback.print_exc() 