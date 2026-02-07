"""Main program entry point for the FPL points sim."""

from src.DataIngester.get_data import fetch_all_players_data, fetch_static_data
from src.DataProcessor.process_data import *

from src.BayesianModels.GKModel import fit_gk_model
from src.BayesianModels.DEFModel import fit_def_model
from src.BayesianModels.MIDModel import fit_mid_model
from src.BayesianModels.STRModel import fit_str_model

from src.MonteCarlo.utils import extract_posteriors
from src.MonteCarlo.monte_carlo_sim import simulate_squad_mc


def main() -> None:
    """Main function to run the FPL points simulation."""

    # Step 1: Fetch and preprocess data
    static_data = fetch_static_data()
    teams_df = extract_teams_data(static_data)
    player_ids_df = extract_player_ids_data(static_data)
    player_ids = player_ids_df["id"].to_list()
    raw_player_data = extract_detailed_player_data(player_ids)
    player_position_mapping = extract_player_position_mapping(static_data)
    upcoming_df = pl.DataFrame(unpack_upcoming_fixtures(raw_player_data))
    matches_df = pl.DataFrame(unpack_played_fixtures(raw_player_data))
    player_name_map = extract_player_name_map(player_ids_df)
    fdr_dict = calculate_fdr(upcoming_df, player_name_map, FDR_MULTIPLIER_MAP)
    matches_df = preprocess_match_data(matches_df, player_ids_df, teams_df, 24)
    positions_df = make_player_positions_df(static_data, player_position_mapping)
    matches_df = matches_df.join(positions_df, on="player_id", how="left")
    player_position_map = make_player_position_map(static_data, player_position_mapping)

    gk_idx_map, def_idx_map, mid_idx_map, fwd_idx_map = (
        make_position_to_player_ids_list_map(player_position_map)
    )
    gk_matches, def_matches, mid_matches, fwd_matches = filter_matches_by_position(
        matches_df
    )

    gk_player_idx = return_final_position_indices(gk_idx_map, gk_matches)
    def_player_idx = return_final_position_indices(def_idx_map, def_matches)
    mid_player_idx = return_final_position_indices(mid_idx_map, mid_matches)
    fwd_player_idx = return_final_position_indices(fwd_idx_map, fwd_matches)

    gk_obs = prepare_obs_arrays(matches_df, gk_idx_map)
    def_obs = prepare_obs_arrays(matches_df, def_idx_map)
    mid_obs = prepare_obs_arrays(matches_df, mid_idx_map)
    fwd_obs = prepare_obs_arrays(matches_df, fwd_idx_map)

    n_defenders = len(def_idx_map)
    n_midfielders = len(mid_idx_map)
    n_goalkeepers = len(gk_idx_map)
    n_forwards = len(fwd_idx_map)

    gk_player_idx = gk_obs["player_idx"]
    def_player_idx = def_obs["player_idx"]
    mid_player_idx = mid_obs["player_idx"]
    fwd_player_idx = fwd_obs["player_idx"]

    gk_minutes = gk_obs["minutes"]
    def_minutes = def_obs["minutes"]
    mid_minutes = mid_obs["minutes"]
    fwd_minutes = fwd_obs["minutes"]

    minutes_params_gk = compute_minutes_beta_params(gk_matches)
    minutes_params_def = compute_minutes_beta_params(def_matches)
    minutes_params_mid = compute_minutes_beta_params(mid_matches)
    minutes_params_fwd = compute_minutes_beta_params(fwd_matches)

    # Step 2: Fit Bayesian models
    gk_trace = fit_gk_model(
        goals=gk_obs["goals"],
        assists=gk_obs["assists"],
        saves=gk_obs["saves"],
        goals_conceded=gk_obs["gc"],
        minutes=gk_obs["minutes"],
        yellow_cards=gk_obs["yc"],
        red_cards=gk_obs["rc"],
        n_goalkeepers=n_goalkeepers,
        player_idx=gk_player_idx,
    )

    def_model = fit_def_model(
        goals=def_obs["goals"],
        assists=def_obs["assists"],
        defensive_contributions=def_obs["dc"],
        goals_conceded=def_obs["gc"],
        minutes=def_obs["minutes"],
        yellow_cards=def_obs["yc"],
        red_cards=def_obs["rc"],
        n_defenders=n_defenders,
        player_idx=def_player_idx,
    )

    mid_model = fit_mid_model(
        goals=mid_obs["goals"],
        assists=mid_obs["assists"],
        defensive_contributions=mid_obs["dc"],
        minutes=mid_obs["minutes"],
        goals_conceded=mid_obs["gc"],
        yellow_cards=mid_obs["yc"],
        red_cards=mid_obs["rc"],
        n_midfielders=n_midfielders,
        player_idx=mid_player_idx,
    )

    fwd_model = fit_str_model(
        goals=fwd_obs["goals"],
        assists=fwd_obs["assists"],
        defensive_contributions=fwd_obs["dc"],
        minutes=fwd_obs["minutes"],
        goals_conceded=fwd_obs["gc"],
        yellow_cards=fwd_obs["yc"],
        red_cards=fwd_obs["rc"],
        n_strikers=n_forwards,
        player_idx=fwd_player_idx,
    )

    # Step 3: Simulate points for the upcoming 5 gameweeks
    gk_posteriors = extract_posteriors(
        gk_trace,
        gk_idx_map,
        "Goalkeeper",
        minutes_params_gk,
        minutes_params_def,
        minutes_params_mid,
        minutes_params_fwd,
    )
    def_posteriors = extract_posteriors(
        def_model,
        def_idx_map,
        "Defender",
        minutes_params_gk,
        minutes_params_def,
        minutes_params_mid,
        minutes_params_fwd,
    )
    mid_posteriors = extract_posteriors(
        mid_model,
        mid_idx_map,
        "Midfielder",
        minutes_params_gk,
        minutes_params_def,
        minutes_params_mid,
        minutes_params_fwd,
    )
    fwd_posteriors = extract_posteriors(
        fwd_model,
        fwd_idx_map,
        "Forward",
        minutes_params_gk,
        minutes_params_def,
        minutes_params_mid,
        minutes_params_fwd,
    )

    squad_posteriors = {}
    squad_posteriors.update(gk_posteriors)
    squad_posteriors.update(def_posteriors)
    squad_posteriors.update(mid_posteriors)
    squad_posteriors.update(fwd_posteriors)

    N_weeks = 5
    # Initialize FDR dictionaries using actual multipliers from fdr_dict
    FDR_goals_dict_players = {
        pid: np.array(fdr_dict[pid]["multipliers"]) for pid in fdr_dict.keys()
    }
    FDR_assists_dict_players = {
        pid: np.array(fdr_dict[pid]["multipliers"]) for pid in fdr_dict.keys()
    }
    FDR_dc_dict_players = {
        pid: np.array(fdr_dict[pid]["multipliers"]) for pid in fdr_dict.keys()
    }
    FDR_cs_dict_players = {
        pid: np.array(fdr_dict[pid]["multipliers"]) for pid in fdr_dict.keys()
    }

    points_mc = simulate_squad_mc(
        squad_posteriors=squad_posteriors,
        FDR_goals_dict=FDR_goals_dict_players,
        FDR_assists_dict=FDR_assists_dict_players,
        FDR_dc_dict=FDR_dc_dict_players,
        FDR_cs_dict=FDR_cs_dict_players,
        N_weeks=N_weeks,
        N_sim=4000,
    )

    # player_id -> name
    player_name_map = {
        row["player_id"]: f"{row['first_name']} {row['last_name']}"
        for row in matches_df.select(["player_id", "first_name", "last_name"])
        .unique()
        .to_dicts()
    }

    # player_id -> position
    player_position_map = {
        row["player_id"]: row["position"]
        for row in matches_df.select(["player_id", "position"]).unique().to_dicts()
    }

    rows = []
    for pid, stats in points_mc.items():
        rows.append(
            {
                "player_id": pid,
                "name": player_name_map.get(pid, "Unknown"),
                "position": player_position_map.get(pid, "Unknown"),
                "mean_points": stats["mean"],
                "median_points": stats["median"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "p90": stats["p90"],
            }
        )

    mc_df = pl.DataFrame(rows)

    mc_df.filter(pl.col("position") == "Goalkeeper").sort(
        "mean_points", descending=True
    ).head(20)


if __name__ == "__main__":
    main()
