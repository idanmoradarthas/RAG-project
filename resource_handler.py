import pickle
from pathlib import Path
from typing import List

import faiss


def get_all_chunks(chunk_file_path_str: str) -> List[str]:
    """
    Reads and un-pickles the 'rag-mini-wikipedia' dataset chunks from a file.

    :param chunk_file_path_str: The file path to the pickle file containing the chunks.
    :return: The chunks as a list of text.
    :raises FileNotFoundError: If the pickle file does not exist.
    :raises pickle.UnpicklingError: If there is an error unpickling the file.
    """
    try:
        with Path(chunk_file_path_str).open("rb") as chunks_pickled:
            return pickle.load(chunks_pickled)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"Chunk file not found: {fnf_error}")
    except pickle.UnpicklingError as up_error:
        raise pickle.UnpicklingError(f"Error unpickling the file: {up_error}")


def read_vector_index(faiss_index_file_path_str: str) -> faiss.Index:
    """
    Reads the FAISS index for the 'rag-mini-wikipedia' dataset.

    :param faiss_index_file_path_str: The file path to the FAISS index file.
    :return: The FAISS index.
    :raises FileNotFoundError: If the index file does not exist.
    :raises OSError: If there is an error reading the index.
    """
    try:
        return faiss.read_index(faiss_index_file_path_str)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"FAISS index file not found: {fnf_error}")
    except OSError as os_error:
        raise OSError(f"Error reading the FAISS index file: {os_error}")
