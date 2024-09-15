import faiss
import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

from logic import filter_indices, generate_numbered_context, encode_user_query, search_index, generate_prompt, \
    SYSTEM_PROMPT


def test_filter_indices():
    filtered_scores, filtered_indices = filter_indices(np.array([[0.7461505, 0.65898716, 0.1, -0.5587512, -0.5586386]]),
                                                       np.array([[505, 624, 519, 625, 240]]), 0.2)
    assert filtered_scores == [0.7461505, 0.65898716]
    assert filtered_indices == [505, 624]


def test_filter_indices_empty_result():
    filtered_scores, filtered_indices = filter_indices(
        np.array([[-0.7461505, -0.65898716, 0.1, -0.5587512, -0.5586386]]),
        np.array([[505, 624, 519, 625, 240]]), 0.2)
    assert len(filtered_scores) == 0
    assert len(filtered_indices) == 0


@pytest.mark.parametrize("scores, indices, error, error_message", [
    (np.array([]), np.array([]), ValueError, "Input arrays must be 2-dimensional"),
    (np.array([[]]), np.array([[]]), ValueError, "Input arrays cannot be empty"),
    (np.array([[0.7461505, 0.65898716, 0.1, -0.5587512]]), np.array([[505, 624, 519, 625, 240]]), ValueError,
     "Input arrays must have the same shape")
], ids=["test_filter_indices_empty_list", "test_filter_indices_empty_inner_list", "test_filter_indices_misaligned"])
def test_filter_indices_error(scores, indices, error, error_message):
    with pytest.raises(error) as error_info:
        filter_indices(scores, indices)
    assert str(error_info.value) == error_message


def test_generate_numbered_context():
    context_sentences = ["This is a sentence.", "Another sentence here.", "More context."]
    result = generate_numbered_context(context_sentences)

    assert result == "1. This is a sentence.\n2. Another sentence here.\n3. More context.\n"


@pytest.mark.parametrize("context_sentences, start, error_message", [
    ([], 1, "The list of context sentences cannot be empty."),
    (["This is a sentence.", "Another sentence here.", "More context."], 0, "The start parameter must be 1 or greater.")
], ids=["test_generate_numbered_context_no_context", "test_generate_numbered_context_start_less_1"])
def test_generate_numbered_context_error(context_sentences, start, error_message):
    with pytest.raises(ValueError) as error_info:
        generate_numbered_context(context_sentences, start)
    assert str(error_info.value) == error_message


def test_encode_user_query(mocker):
    user_query = "What is the capital of France?"

    # Create a mock array with a reshape method
    mock_array = mocker.Mock(spec=np.ndarray)
    mock_array.reshape.return_value = np.array([[1, 1, 1, 1]])

    mock_encoder = mocker.Mock()
    mock_encoder.encode.return_value = mock_array

    mock_normalize_l2 = mocker.patch('logic.faiss.normalize_L2')

    query_vector = encode_user_query(user_query, mock_encoder)

    assert query_vector.shape == (1, 4)
    mock_encoder.encode.assert_called_once_with(user_query)
    mock_array.reshape.assert_called_once_with(1, -1)
    mock_normalize_l2.assert_called_once()

    # Check that normalize_L2 was called with the reshaped array
    np.testing.assert_array_equal(mock_normalize_l2.call_args[0][0], np.array([[1, 1, 1, 1]]))


def test_encode_user_query_empty_query(mocker):
    with pytest.raises(ValueError) as error_info:
        encode_user_query("", mocker.Mock())
    assert str(error_info.value) == "User query cannot be empty."


def test_search_index(mocker):
    user_query_vector = np.array([[1, 1, 1, 1]])
    top_k = 5
    index = mocker.Mock(spec=faiss.Index)

    scores = np.array([0.74615026, 0.65898675, 0.57043445, 0.5587513, 0.55863833])
    indices = np.array([505, 624, 519, 625, 240])
    index.search.return_value = scores, indices
    search_index(user_query_vector, index, top_k)

    index.search.assert_called_once_with(user_query_vector, top_k)


@pytest.mark.parametrize("user_query_vector, top_k, error_message", [
    (np.array([1, 1, 1, 1]), 5, "user_query_vector must be a 2D array with shape (1, dimension)"),
    (np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), 5, "user_query_vector must be a 2D array with shape (1, dimension)"),
    (np.array([[1, 1, 1, 1]]), 0, "top_k parameter must be 1 or greater.")
], ids=["test_search_index_user_query_less_than_2d", "test_search_index_user_query_wrong_dimension",
        "test_search_index_top_k_less_than_1"])
def test_search_index_error(mocker, user_query_vector, top_k, error_message):
    index = mocker.Mock()
    with pytest.raises(ValueError) as error_info:
        search_index(user_query_vector, index, top_k)
    assert str(error_info.value) == error_message


def test_generate_prompt_empty_chunks(mocker):
    with pytest.raises(IndexError) as error_info:
        generate_prompt("", mocker.Mock(), mocker.Mock(), [], mocker.Mock())
    assert str(error_info.value) == "all_chunks cannot be empty."


def test_generate_prompt_context_found(mocker):
    user_query = "What is the capital of France?"
    encoder = mocker.Mock()
    index = mocker.Mock()
    all_chunks = []
    for i in range(1, 1001):
        all_chunks.append(f"chunk {i}")
    tokenizer = mocker.Mock(spec=PreTrainedTokenizerBase)

    user_query_vector = np.array([[1, 1, 1, 1]])
    mock_encode_user_query = mocker.patch("logic.encode_user_query")
    mock_encode_user_query.return_value = user_query_vector

    scores = np.array([0.74615026, 0.65898675, 0.57043445, 0.5587513, 0.55863833])
    indices = np.array([505, 624, 519, 625, 240])
    mock_search_index = mocker.patch("logic.search_index")
    mock_search_index.return_value = scores, indices

    mock_filter_indices = mocker.patch("logic.filter_indices")
    mock_filter_indices.return_value = scores, indices

    mock_generate_numbered_context = mocker.patch("logic.generate_numbered_context")
    mock_generate_numbered_context.return_value = ("1. This is a sentence.\n2. Another sentence here.\n"
                                                   "3. More context.\n")

    generate_prompt(user_query, encoder, index, all_chunks, tokenizer)

    context = f"### Context:\n1. This is a sentence.\n2. Another sentence here.\n3. More context.\n"
    user_query_text = f"### User Query:\n{user_query}"
    instructions = "### Instructions:\n" + \
                   "Please provide a concise and accurate response based on " + \
                   "the context above if it is relevant."
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n{user_query_text}\n\n{instructions}"}
    ]

    mock_encode_user_query.assert_called_once_with(user_query, encoder)
    mock_search_index.assert_called_once_with(user_query_vector, index)
    mock_filter_indices.assert_called_once_with(scores, indices)
    mock_generate_numbered_context.assert_called_once_with(
        ['chunk 506', 'chunk 625', 'chunk 520', 'chunk 626', 'chunk 241'])
    tokenizer.apply_chat_template.assert_called_once_with(prompt, tokenize=False, add_generation_prompt=True)


def test_generate_prompt_retrieve_indices_error(mocker):
    user_query = "What is the capital of France?"
    encoder = mocker.Mock()
    index = mocker.Mock()
    all_chunks = []
    for i in range(1, 11):
        all_chunks.append(f"chunk {i}")

    user_query_vector = np.array([[1, 1, 1, 1]])
    mock_encode_user_query = mocker.patch("logic.encode_user_query")
    mock_encode_user_query.return_value = user_query_vector

    scores = np.array([0.74615026, 0.65898675, 0.57043445, 0.5587513, 0.55863833])
    indices = np.array([505, 624, 519, 625, 240])
    mock_search_index = mocker.patch("logic.search_index")
    mock_search_index.return_value = scores, indices

    mock_filter_indices = mocker.patch("logic.filter_indices")
    mock_filter_indices.return_value = scores, indices

    with pytest.raises(IndexError) as error_info:
        generate_prompt(user_query, encoder, index, all_chunks, mocker.Mock())
    assert str(error_info.value) == "Retrieved indices are out of range for all_chunks."


def test_generate_prompt_context_not_found(mocker):
    user_query = "What is the capital of France?"
    encoder = mocker.Mock()
    index = mocker.Mock()
    all_chunks = []
    for i in range(1, 1001):
        all_chunks.append(f"chunk {i}")
    tokenizer = mocker.Mock(spec=PreTrainedTokenizerBase)

    user_query_vector = np.array([[1, 1, 1, 1]])
    mock_encode_user_query = mocker.patch("logic.encode_user_query")
    mock_encode_user_query.return_value = user_query_vector

    scores = np.array([0.74615026, 0.65898675, 0.57043445, 0.5587513, 0.55863833])
    indices = np.array([505, 624, 519, 625, 240])
    mock_search_index = mocker.patch("logic.search_index")
    mock_search_index.return_value = scores, indices

    mock_filter_indices = mocker.patch("logic.filter_indices")
    mock_filter_indices.return_value = [], []

    generate_prompt(user_query, encoder, index, all_chunks, tokenizer)

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_query}"}
    ]

    mock_encode_user_query.assert_called_once_with(user_query, encoder)
    mock_search_index.assert_called_once_with(user_query_vector, index)
    mock_filter_indices.assert_called_once_with(scores, indices)
    tokenizer.apply_chat_template.assert_called_once_with(prompt, tokenize=False, add_generation_prompt=True)
