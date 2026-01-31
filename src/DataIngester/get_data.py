"""This module is responsible for ingesting data from the FPL API."""
from typing import List
import requests
import asyncio
import aiohttp

import polars as pl

FPL_BASE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
PLAYER_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

def fetch_static_data() -> pl.DataFrame:
    """Fetches static data from the FPL API.
    
    Returns:
        pl.DataFrame: DataFrame containing static data.
    """
    pass


async def fetch_player_data(session: aiohttp.ClientSession, player_id: int) -> dict:
    """Fetches individual player data from the FPL API.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for the request.
        player_id (int): The ID of the player to fetch data for.
    
    Returns:
        dict: Dictionary containing player data.
    """
    pass


async def fetch_all_players_data(player_ids: List[int]) -> List[dict]:
    """Fetches data for all players asynchronously.
    
    Args:
        player_ids (List[int]): List of player IDs to fetch data for.

    Returns:
        List[dict]: List of dictionaries containing player data.
    """
    pass


