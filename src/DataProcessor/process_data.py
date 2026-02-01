"""Module responsible for processing FPL data."""
from typing import List
import asyncio
import aiohttp
import polars as pl

from src.DataIngester.get_data import fetch_all_players_data, PLAYER_URL

def extract_teams_data(static_data: dict) -> pl.DataFrame:
    """Extract teams data from static FPL data.

    Args:
        static_data (dict): The static FPL data containing teams information.

    Returns:
        pl.DataFrame: Processed DataFrame containing teams data.
    """
    return pl.DataFrame(static_data["teams"])


def extract_player_ids_data(static_data: dict) -> pl.DataFrame:
    """Extract player IDs and names from static FPL data.

    Args:
        static_data (dict): The static FPL data containing player information.

    Returns:
        pl.DataFrame: Processed DataFrame containing player IDs and names.
    """
    player_info = [
        {"id": player["id"], "first_name": player["first_name"], "last_name": player["web_name"]}
        for player in static_data["elements"]
    ]
    
    return pl.DataFrame(player_info)


def extract_detailed_player_data(player_ids: List[int]) -> pl.DataFrame:
    """Extract detailed player data for given player IDs.

    Args:
        player_ids (List[int]): List of player IDs to fetch detailed data for.

    Returns:
        pl.DataFrame: Processed DataFrame containing detailed player data.
    """
    detailed_player_data = await fetch_all_players_data(player_ids)
    return pl.DataFrame(detailed_player_data)


def extract_player_position_mapping(static_data: dict) -> dict[str, str]:
    """Extract mapping of player position IDs to position names.

    Args:
        static_data (dict): The static FPL data containing position information.

    Returns:
        dict[str, str]: Mapping of position IDs to position names.
    """
    element_types = static_data["element_types"]
    return {
        element_type["id"]: element_type["singular_name"]
        for element_type in element_types
    }


def extract_upcoming_fixtures(static_data: dict) -> pl.DataFrame:
    """Extract upcoming fixtures from static FPL data.

    Args:
        static_data (dict): The static FPL data containing fixtures information.

    Returns:
        pl.DataFrame: Processed DataFrame containing upcoming fixtures.
    """
    return pl.DataFrame(static_data["elements"])


def unpack_upcoming_fixtures(data: List[tuple[int, dict]]) -> List[dict]:
    """Unpack upcoming fixtures from player data.

    Args:
        data (List[tuple[int, dict]]): List of tuples containing player IDs and their corresponding data.

    Returns:
        List[dict]: List of dictionaries containing unpacked fixture data.
    """
    rows = []

    for player_id, payload in data:
        if isinstance(payload, dict):
            fixtures = payload.get("fixtures", [])

            for fixture in fixtures:
                rows.append({
                    "player_id": player_id,
                    **fixture
                })

    return rows


def unpack_played_fixtures(data: List[tuple[int, dict]]) -> List[dict]:
    """Unpack played fixtures from player data.

    Args:
        data (List[tuple[int, dict]]): List of tuples containing player IDs and their corresponding data.

    Returns:
        List[dict]: List of dictionaries containing unpacked played fixture data.
    """
    rows = []

    for player_id, payload in data:
        if isinstance(payload, dict):
            matches = payload.get("history", [])

            for match_ in matches:
                rows.append({
                    "player_id": player_id,
                    **match_
                })

    return rows


