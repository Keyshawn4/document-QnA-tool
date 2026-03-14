from document_QnA_tool import generate_hash, doc_to_chunks
import pytest
from pathlib import Path
import hashlib

@ pytest.fixture
def mock_path(tmp_path):
    return tmp_path / "data" / "mock_document.pdf"

def test_generate_hash_len(mock_path):
    assert len(generate_hash(mock_path)) == 64, "sha256 always produces a 64 character hex string"

def test_generate_hash_determinism(mock_path):
    assert generate_hash(mock_path) == generate_hash(mock_path)

def test_generate_hash_uniqueness(mock_path):
    test_path = mock_path.parent / "mock2_document.pdf"
    assert generate_hash(mock_path) != generate_hash(test_path)

def test_doc_to_chunks_path_exists(mock_path):
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        doc_to_chunks(mock_path)

def test_doc_to_chunks_path_directory_not_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Path is a directory, not a file"):
        doc_to_chunks(tmp_path)

def test_doc_to_chunks_path_wrong_file_type(tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "mock_document.json"
    test_file.write_text('{"Example": Test}')
    with pytest.raises(ValueError, match="Not a supported file type"):
        doc_to_chunks(test_file)
    