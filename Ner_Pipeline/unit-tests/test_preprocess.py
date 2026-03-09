# test_preprocess.py

import pytest
from pathlib import Path
from steps.preprocess import process_single_file
from unittest.mock import patch

@patch("steps.preprocess.load_brat")
@patch("steps.preprocess.clean_text")
@patch("steps.preprocess.tokenize_with_offsets")
@patch("steps.preprocess.label_tokens_with_iob")
def test_process_single_file(mock_label_iob, mock_tokenize, mock_clean, mock_load_brat):
    # Arrange
    dummy_file_id = "example"
    dummy_input_dir = Path("/dummy/path")

    # Mock return values
    mock_load_brat.return_value = [{
        "text": "Original text.",
        "entities": [{"start": 0, "end": 8, "label": "CELL", "entity": "Original"}]
    }]
    mock_clean.return_value = "Cleaned text."
    mock_tokenize.return_value = [("Cleaned", 0, 7), ("text", 8, 12)]
    mock_label_iob.return_value = [("Cleaned", "B-CELL"), ("text", "O")]

    # Act
    result = process_single_file(dummy_file_id, dummy_input_dir)

    # Assert
    mock_load_brat.assert_called_once()
    mock_clean.assert_called_once_with("Original text.")
    mock_tokenize.assert_called_once_with("Cleaned text.")
    mock_label_iob.assert_called_once()
    assert result == [("Cleaned", "B-CELL"), ("text", "O")]
