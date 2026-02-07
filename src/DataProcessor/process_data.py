"""Module responsible for processing FPL data."""
from typing import List, Sequence
import asyncio
from collections import defaultdict
import polars as pl

from src.DataIngester.get_data import fetch_all_players_data, fetch_static_data, PLAYER_URL

FDR_MULTIPLIER_MAP = {
    1: 1.2,
    2: 1.1,
    3: 1.0,
    4: 0.9,
    5: 0.8
}

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


def extract_detailed_player_data(player_ids: List[int]) -> Sequence[tuple[int, dict]]:
    """Run `fetch_all_players_data` safely from sync code and return a pl.DataFrame.

    Ensures the coroutine runs either with `asyncio.run()` or inside a fresh thread+loop
    if an event loop is already running. Normalizes returned data into a list of
    dict rows so Polars doesn't interpret tuples as columns.

    Args:
        player_ids (List[int]): List of player IDs to fetch data for.

    Returns:
        dict[int, dict]: Dictionary mapping player IDs to their corresponding data.
    """
    coro = fetch_all_players_data(player_ids)

    try:
        # If there's no running loop, this runs the coroutine normally.
        data = asyncio.run(coro)
    except RuntimeError:
        # Running inside an existing loop (e.g. notebook, embedded). Run in a thread.
        import concurrent.futures

        def _run_in_thread(ids):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(fetch_all_players_data(ids))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            data = ex.submit(_run_in_thread, player_ids).result()

    return data


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


def extract_upcoming_fixtures(static_data: dict) -> dict:
    """Extract upcoming fixtures from static FPL data.

    Args:
        static_data (dict): The static FPL data containing fixtures information.

    Returns:
        pl.DataFrame: Processed DataFrame containing upcoming fixtures.
    """
    return static_data["elements"]


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


def calculate_fdr(upcoming_df: pl.DataFrame, player_name_map: dict, fdr_multiplier_map: dict, n_weeks: int = 5) -> dict:
    """Calculate Fixture Difficulty Ratings (FDR) for players.

    Args:
        upcoming_df (pl.DataFrame): DataFrame containing upcoming fixtures.
        player_name_map (dict): Mapping of player IDs to player names.
        fdr_multiplier_map (dict): Mapping of difficulty ratings to multipliers.
        n_weeks (int): Number of upcoming weeks to consider for FDR calculation.

    Returns:
        dict: Dictionary containing FDR information for each player.
    """
    fdr_dict = {}

    unique_player_ids = upcoming_df["player_id"].unique()

    for player_id in unique_player_ids:
        player_fixtures = upcoming_df.filter(pl.col("player_id") == player_id)
        sorted_fixtures = player_fixtures.sort("event")
        next_n_difficulties = sorted_fixtures["difficulty"].head(n_weeks).to_list()
        fdr_dict[player_id] = {
            "name": player_name_map.get(player_id, "Unknown"),
            "difficulties": list(next_n_difficulties),
            "multipliers": [fdr_multiplier_map[d] for d in next_n_difficulties]
        }

    return fdr_dict


def extract_player_name_map(player_ids_df: pl.DataFrame) -> dict:
    """Extract mapping of player IDs to player names.

    Args:
        player_ids_df (pl.DataFrame): DataFrame containing player IDs and names.

    Returns:
        dict: Mapping of player IDs to player names.
    """
    player_name_map = {
    row["id"]: f"{row['first_name']} {row['last_name']}"
    for row in player_ids_df.select(["id", "first_name", "last_name"]).to_dicts()
    }
    return player_name_map


def preprocess_match_data(matches_df: pl.DataFrame, player_ids_df: pl.DataFrame, teams_df: pl.DataFrame, gw_played: int) -> pl.DataFrame:
    """Preprocess match data by selecting relevant columns and renaming them.

    Args:
        matches_df (pl.DataFrame): DataFrame containing match data.
        player_ids_df (pl.DataFrame): DataFrame containing player IDs and names.
        teams_df (pl.DataFrame): DataFrame containing team data.
        gw_played (int): Minimum number of gameweeks a player must have played to be included.

    Returns:
        pl.DataFrame: Preprocessed DataFrame with selected and renamed columns.
    """
    matches_df = matches_df.join(
              player_ids_df.select([
                  pl.col('id').alias('player_id'),
                  pl.col('first_name'),
                  pl.col('last_name')
              ]),
              on='player_id',
              how='left'
            )
    
    matches_df = matches_df.join(
                teams_df.select([
                pl.col('id').alias('opponent_team'),
                pl.col('name').alias('opponent_team_name')
            ]),
                on='opponent_team',
                how='left'
            )
    
    matches_df = matches_df.with_columns([
                pl.col('kickoff_time').cast(pl.Datetime),
                pl.col('influence').cast(pl.Float64),
                pl.col('creativity').cast(pl.Float64),
                pl.col('threat').cast(pl.Float64),
                pl.col('ict_index').cast(pl.Float64),
                pl.col('expected_goals').cast(pl.Float64),
                pl.col('expected_assists').cast(pl.Float64),
                pl.col('expected_goal_involvements').cast(pl.Float64),
                pl.col('expected_goals_conceded').cast(pl.Float64),
            ])
    
    matches_df = matches_df.with_columns([
                pl.col('value') / 10
            ])
    
    counts = (
        matches_df
        .group_by("player_id")
        .agg(pl.len().alias("n_matches"))
    )

    valid_players = counts.filter(pl.col("n_matches") >= gw_played)

    matches_df = matches_df.join(
        valid_players.select("player_id"),
        on="player_id",
        how="inner"
    )
    # unfinished - continue from here

    return matches_df


def make_player_position_map(static_data: dict, position_map: dict) -> dict[int, str]:
    """Create a mapping of player IDs to their positions.

    Args:
        static_data (dict): The static FPL data containing player information.
        position_map (dict): Mapping of element types to position names.

    Returns:
        dict[int, str]: Mapping of player IDs to their corresponding positions.
    """
    player_position_map = {element["id"]: position_map[element["element_type"]] for element in static_data['elements']}
    return player_position_map


def make_player_positions_df(static_data: dict, position_map: dict) -> pl.DataFrame:
    """Create a DataFrame mapping player IDs to their positions.

    Args:
        static_data (dict): The static FPL data containing player information.
        position_map (dict): Mapping of element types to position names.

    Returns:
        pl.DataFrame: DataFrame containing player IDs and their corresponding positions.
    """
    player_position_map = make_player_position_map(static_data, position_map)

    # Convert the dictionary to the correct format for pl.from_dict
    positions_df = pl.from_dict(
        data={
            "player_id": list(player_position_map.keys()),
            "position": list(player_position_map.values())
        },
        schema={
            "player_id": pl.Int64,
            "position": pl.Utf8
        }
    )

    return positions_df


def make_position_to_player_ids_list_map(player_position_map: dict) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Create a mapping of positions to lists of player IDs.

    Args:
        player_position_map (dict): Mapping of player IDs to their positions.

    Returns:
        tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]: Tuple of mappings of positions to lists of player IDs.
    """
    # Create a dict of position -> list of player IDs
    players_by_position = defaultdict(list)

    for pid, pos in player_position_map.items():
        players_by_position[pos].append(pid)

    # This creates a mapping: player_id -> row index for hierarchical model
    idx_map_by_position = {}

    for pos, pid_list in players_by_position.items():
        idx_map = {pid: i for i, pid in enumerate(pid_list)}
        idx_map_by_position[pos] = idx_map

    gk_idx_map = idx_map_by_position["Goalkeeper"]
    def_idx_map = idx_map_by_position["Defender"]
    mid_idx_map = idx_map_by_position["Midfielder"]
    fwd_idx_map = idx_map_by_position["Forward"]

    return gk_idx_map, def_idx_map, mid_idx_map, fwd_idx_map


def filter_matches_by_position(matches_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Filter matches DataFrame by player positions.

    Args:
        matches_df (pl.DataFrame): DataFrame containing match data with player positions.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple of DataFrames filtered by position (GK, DEF, MID, FWD).
    """
    gk_matches = matches_df.filter(pl.col("position") == "Goalkeeper")
    def_matches = matches_df.filter(pl.col("position") == "Defender")
    mid_matches = matches_df.filter(pl.col("position") == "Midfielder")
    fwd_matches = matches_df.filter(pl.col("position") == "Forward")

    return gk_matches, def_matches, mid_matches, fwd_matches


def ensure_unique(
        gk_data: pl.DataFrame, 
        def_data: pl.DataFrame, 
        mid_data: pl.DataFrame, 
        str_data: pl.DataFrame
        ) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Ensure that player IDs are unique within each position DataFrame and create index mappings.

    Args:
        gk_data (pl.DataFrame): DataFrame containing goalkeeper match data.
        def_data (pl.DataFrame): DataFrame containing defender match data.
        mid_data (pl.DataFrame): DataFrame containing midfielder match data.
        str_data (pl.DataFrame): DataFrame containing forward match data.

    Returns:
        tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]: Tuple of index mappings for each position.
    """
    # Only unique player_ids in positions
    gk_player_ids = gk_data["player_id"].unique().to_list()
    gk_idx_map = {pid: i for i, pid in enumerate(gk_player_ids)}

    def_player_ids = def_data["player_id"].unique().to_list()
    def_idx_map = {pid: i for i, pid in enumerate(def_player_ids)}

    mid_player_ids = mid_data["player_id"].unique().to_list()
    mid_idx_map = {pid: i for i, pid in enumerate(mid_player_ids)}

    str_player_ids = str_data["player_id"].unique().to_list()
    str_idx_map = {pid: i for i, pid in enumerate(str_player_ids)}

    return gk_idx_map, def_idx_map, mid_idx_map, str_idx_map