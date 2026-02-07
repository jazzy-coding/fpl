"""This module is responsible for ingesting data from the FPL API."""

from typing import List, Sequence
import requests
import asyncio
import aiohttp

FPL_BASE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
PLAYER_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"


def fetch_static_data() -> dict:
    """Fetches static data from the FPL API.

    Returns:
        dict: Dictionary containing static data.
    """
    data = requests.get(FPL_BASE_URL)
    return data.json()


async def fetch_player_data(
    session: aiohttp.ClientSession, player_id: int
) -> tuple[int, dict]:
    """Fetches individual player data from the FPL API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for the request.
        player_id (int): The ID of the player to fetch data for.

    Returns:
        dict: Dictionary containing player data.
    """
    async with session.get(PLAYER_URL.format(playerID=player_id)) as response:
        data = await response.json()
        return player_id, data


async def fetch_all_players_data(
    player_ids: List[int], concurrency: int = 20
) -> Sequence[tuple[int, dict]]:
    """Fetches data for all players asynchronously.

    Args:
        player_ids (List[int]): List of player IDs to fetch data for.
        concurrency (int): Number of concurrent requests to make. Default is 20.

    Returns:
        Sequence[tuple[int, dict]]: Sequence of tuples containing player IDs and their corresponding data.
    """
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_player_data(session, player_id) for player_id in player_ids]
        return await asyncio.gather(*tasks)
