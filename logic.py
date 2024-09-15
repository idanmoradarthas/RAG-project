from typing import Tuple, List, Union, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

SYSTEM_PROMPT: str = """You are a helpful, respectful, and honest assistant. Always
answer as helpfully as possible, while being safe and unbiased. For
yes or no questions, answer with 'Yes' or 'No.' If not, provide the
most accurate, short and concise answer. Use provided context
information to answer questions accurately when available."""


def filter_indices(scores: np.ndarray, chunks_indices: np.ndarray,
                   threshold: float = 0.2) -> Tuple[List[float], List[int]]:
    """
    Filters out indices where the score is below a specified threshold.

    :param scores: A 2D numpy array containing similarity scores from the index search.
    :param chunks_indices: A 2D numpy array containing indices of the retrieved chunks.
    :param threshold: The minimum similarity score for an index to be included (default: 0.2).
    :return: A tuple containing filtered scores and their corresponding indices as lists.
    :raises ValueError: If the input arrays are empty, misaligned, or have incorrect shapes.

    """
    if scores.ndim != 2 or chunks_indices.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    if scores.shape[0] == 0 or scores.shape[1] == 0 or chunks_indices.shape[0] == 0 or chunks_indices.shape[1] == 0:
        raise ValueError("Input arrays cannot be empty")
    if scores.shape != chunks_indices.shape:
        raise ValueError("Input arrays must have the same shape")

    mask = scores[0] > threshold
    filtered_scores = scores[0][mask].tolist()
    filtered_indices = chunks_indices[0][mask].tolist()
    return filtered_scores, filtered_indices


def generate_numbered_context(context_sentences: List[Union[str, Any]],
                              start: int = 1) -> str:
    """
    Generates a numbered list of context sentences.

    :param context_sentences: A list of context sentences.
    :param start: The starting number for the list (default is 1).
    :return: A string with each sentence numbered on a new line.
    :raises ValueError: If the input list is empty or start is less than 1.
    """
    if not context_sentences:
        raise ValueError("The list of context sentences cannot be empty.")
    if start < 1:
        raise ValueError("The start parameter must be 1 or greater.")

    return "\n".join(f"{i}. {str(sentence).strip()}"
                     for i, sentence in enumerate(context_sentences, start=start)) + "\n"


def encode_user_query(user_query: str, encoder: SentenceTransformer) -> np.ndarray:
    """
    Encodes a user query into a vector and normalizes it using L2 normalization.

    :param user_query: The user's query in text form.
    :param encoder: The encoder model to convert the query into a vector.
    :return: The L2-normalized vector representation of the user query as a 2D numpy array with shape (1, n),
             where n is the dimensionality of the encoded vector.
    :raises ValueError: If the user query is empty or the encoder fails.
    """
    if not user_query.strip():
        raise ValueError("User query cannot be empty.")

    user_query_vector = encoder.encode(user_query)
    user_query_vector = user_query_vector.reshape(1, -1)
    faiss.normalize_L2(user_query_vector)
    return user_query_vector


def search_index(user_query_vector: np.ndarray, index: faiss.Index,
                 top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param user_query_vector: The vector representation of the user query.
                              Should be a 2D numpy array with shape (1, dimension).
    :param index: The FAISS index to search in.
    :param top_k: The number of top results to retrieve. Default is 5.
    :return: A tuple containing:
             - scores: A 2D numpy array of shape (1, top_k) with the similarity scores.
             - indices: A 2D numpy array of shape (1, top_k) with the indices of the top_k results.
    :raises ValueError: If the query vector is invalid, top_k is less than 1, or the index search fails.
    """
    if user_query_vector.ndim != 2 or user_query_vector.shape[0] != 1:
        raise ValueError("user_query_vector must be a 2D array with shape (1, dimension)")

    if top_k <= 1:
        raise ValueError("top_k parameter must be 1 or greater.")

    scores, indices = index.search(user_query_vector, top_k)
    return scores, indices


def generate_prompt(user_query: str, encoder: SentenceTransformer, index: faiss.Index, all_chunks: List[str],
                    tokenizer: AutoTokenizer) -> str:
    """
    Generates a prompt for the language model based on the user query and the most relevant context.

    :param user_query: The user's query in text form.
    :param encoder: The encoder model to convert the query into a vector.
    :param index: The FAISS index to search for relevant context.
    :param all_chunks: A list of all context chunks to retrieve from.
    :param tokenizer: The tokenizer used to format the final prompt.
    :return: The formatted prompt ready to be input into a language model.
    :raises ValueError: If the user query is empty, encoding fails, or index search fails.
    :raises IndexError: If all_chunks is empty or the retrieved indices are out of range.
    """
    if not all_chunks:
        raise IndexError("all_chunks cannot be empty.")

    # Encode the user query
    user_query_vector = encode_user_query(user_query, encoder)
    # Now let's search the index
    scores, chunks_indices = search_index(user_query_vector, index)
    # Reduce indices with less than zero similarity score
    _, filtered_indices = filter_indices(scores, chunks_indices)
    if len(filtered_indices) > 0:
        # Get the context
        try:
            similar_chunks = [all_chunks[idx] for idx in filtered_indices]
        except IndexError:
            raise IndexError("Retrieved indices are out of range for all_chunks.")
        # Generate the prompt
        context = f"### Context:\n{generate_numbered_context(similar_chunks)}"
        user_query_text = f"### User Query:\n{user_query}"
        instructions = "### Instructions:\n" + \
                       "Please provide a concise and accurate response based on " + \
                       "the context above if it is relevant."
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n{user_query_text}\n\n{instructions}"}
        ]
    else:
        # If no context is found
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_query}"}
        ]
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
