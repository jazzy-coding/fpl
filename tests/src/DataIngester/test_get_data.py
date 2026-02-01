"""Unit tests for get_data.py module."""

import pytest
from unittest.mock import patch, Mock

from src.DataIngester import fetch_all_players_data, fetch_player_data, fetch_static_data, FPL_BASE_URL, PLAYER_URL


class TestFetchStaticData:
    """Tests for the fetch_static_data function."""

    @patch('src.DataIngester.get_data.requests.get')
    def test_fetch_static_data_success(self, mock_get):
        """Test successful data fetch."""
        mock_response = Mock()
        expected_data = {'elements': []}
        mock_response.json.return_value = expected_data
        mock_get.return_value = mock_response

        data = fetch_static_data()
        assert data == expected_data
        mock_get.assert_called_once_with(FPL_BASE_URL)
