import pickle

import faiss
import pytest

from resource_handler import get_all_chunks, read_vector_index


def test_get_all_chunks_success(mocker):
    mock_data = ["chunk1", "chunk2"]
    pickled_data = pickle.dumps(mock_data)
    mocker.patch("pathlib.Path.open", mocker.mock_open(read_data=pickled_data))

    result = get_all_chunks("mock_path.pkl")

    assert result == mock_data


def test_get_all_chunks_file_not_found(mocker):
    mocker.patch("pathlib.Path.open", side_effect=FileNotFoundError("invalid_path.pkl not found"))

    with pytest.raises(FileNotFoundError) as file_not_found_error:
        get_all_chunks("invalid_path.pkl")

    assert str(file_not_found_error.value) == "Chunk file not found: invalid_path.pkl not found"


def test_get_all_chunks_unpickling_error(mocker):
    mocker.patch("pathlib.Path.open", mocker.mock_open(read_data="not a pickle"))
    mocker.patch("pickle.load", side_effect=pickle.UnpicklingError("mock_path.pkl not a pickle"))

    with pytest.raises(pickle.UnpicklingError) as unpickling_error:
        get_all_chunks("mock_path.pkl")

    assert str(unpickling_error.value) == "Error unpickling the file: mock_path.pkl not a pickle"


def test_read_vector_index_success(mocker):
    mock_faiss_index = mocker.Mock(spec=faiss.Index)
    mocker.patch("faiss.read_index", return_value=mock_faiss_index)

    result = read_vector_index("mock_index_path.index")
    assert result == mock_faiss_index


@pytest.mark.parametrize(
    "exception, expected_message",
    [
        (FileNotFoundError("mock_index_path.index"), "FAISS index file not found: mock_index_path.index"),
        (OSError("mock_index_path.index"), "Error reading the FAISS index file: mock_index_path.index"),
    ],
    ids=["test_read_vector_index_file_not_found", "test_read_vector_index_os_error"]
)
def test_read_vector_index_exceptions(mocker, exception, expected_message):
    mocker.patch("faiss.read_index", side_effect=exception)

    with pytest.raises(type(exception)) as exc_info:
        read_vector_index("mock_index_path.index")

    assert str(exc_info.value) == expected_message
