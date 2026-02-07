"""Module responsible for running Monte Carlo simulations of player performance."""

from typing import Optional
import numpy as np


def simulate_squad_mc(
    squad_posteriors: dict[int, dict],   # dict: player_id -> {"position": str, "lambda_g":..., "lambda_a":..., "lambda_x":..., "lambda_gc":..., "lambda_yc":..., "lambda_rc":..., "alpha_0":..., "beta_0":..., "alpha_60":..., "beta_60":...}
    FDR_goals_dict: dict[int, np.ndarray],    # dict: player_id -> array(N_weeks)
    FDR_assists_dict: Optional[dict[int, np.ndarray]] = None,
    FDR_dc_dict: Optional[dict[int, np.ndarray]] = None,
    FDR_cs_dict: Optional[dict[int, np.ndarray]] = None,
    N_weeks: int = 5,
    N_sim: int=10000
) -> dict[int, dict]:
    """Simulate FPL points for a squad of players over a number of weeks using Monte Carlo methods.

    Args:
        squad_posteriors (dict): Dictionary mapping player IDs to their posterior distributions and parameters.
        FDR_goals_dict (dict): Dictionary mapping player IDs to arrays of fixture difficulty ratings for goals.
        FDR_assists_dict (Optional[dict]): Dictionary mapping player IDs to arrays of fixture difficulty ratings for assists.
        FDR_dc_dict (Optional[dict]): Dictionary mapping player IDs to arrays of fixture difficulty ratings for defensive contributions.
        FDR_cs_dict (Optional[dict]): Dictionary mapping player IDs to arrays of fixture difficulty ratings for clean sheets.
        N_weeks (int): Number of weeks to simulate.
        N_sim (int): Number of Monte Carlo simulations to run per player.

    Returns:
        dict: Dictionary mapping player IDs to their simulation results, including weekly points and summary statistics.
    """

    results = {}

    for player_id, post in squad_posteriors.items(): # Changed player_name to player_id
        position = post["position"]

        # --- Posterior samples ---
        lambda_g_samples  = post["lambda_g"][:N_sim]
        lambda_a_samples  = post["lambda_a"][:N_sim]
        lambda_x_samples  = post["lambda_x"][:N_sim] if post.get("lambda_x") is not None else np.zeros(N_sim)
        lambda_gc_samples = post["lambda_gc"][:N_sim] if post.get("lambda_gc") is not None else np.zeros(N_sim)
        lambda_yc_samples = post["lambda_yc"][:N_sim]
        lambda_rc_samples = post["lambda_rc"][:N_sim]

        # Minutes priors
        alpha_0 = post["alpha_0"]
        beta_0  = post["beta_0"]
        alpha_60 = post["alpha_60"]
        beta_60  = post["beta_60"]

        # FDR
        FDR_goals  = FDR_goals_dict[player_id] # Changed player_name to player_id
        FDR_assists = FDR_assists_dict[player_id] if FDR_assists_dict else FDR_goals # Changed player_name to player_id
        FDR_dc = FDR_dc_dict[player_id] if FDR_dc_dict else np.ones(N_weeks) # Changed player_name to player_id
        FDR_cs = FDR_cs_dict[player_id] if FDR_cs_dict else np.ones(N_weeks) # Changed player_name to player_id

        # Storage
        points_weeks = np.zeros((N_sim, N_weeks))

        for w in range(N_weeks):
            # --- Sample minutes ---
            p0_samples  = np.random.beta(alpha_0, beta_0, size=N_sim)
            p60_samples = np.random.beta(alpha_60, beta_60, size=N_sim)

            Z0 = np.random.binomial(1, p0_samples)
            Z60 = np.zeros(N_sim)
            played_mask = (Z0 == 1)
            Z60[played_mask] = np.random.binomial(1, p60_samples[played_mask])

            M = np.zeros(N_sim)
            full = (Z0 == 1) & (Z60 == 1)
            part = (Z0 == 1) & (Z60 == 0)
            M[full] = np.random.randint(75, 91, size=np.sum(full))
            M[part] = np.random.randint(1, 60, size=np.sum(part))
            played_60 = (M >= 60).astype(int)

            # --- Effective rates ---
            g_eff  = lambda_g_samples  * M / 90 * FDR_goals[w]
            a_eff  = lambda_a_samples  * M / 90 * FDR_assists[w]
            x_eff  = lambda_x_samples  * M / 90
            gc_eff = lambda_gc_samples * M / 90 * FDR_cs[w]   # clean sheets
            yc_eff = lambda_yc_samples * M / 90
            rc_eff = lambda_rc_samples * M / 90

            # --- Sample events ---
            G  = np.random.poisson(g_eff)
            A  = np.random.poisson(a_eff)
            X  = np.random.poisson(x_eff)  # dc or saves
            GC = np.random.poisson(gc_eff)
            YC = np.random.poisson(yc_eff)
            RC = np.random.poisson(rc_eff)

            # --- Compute FPL points ---
            points = np.zeros(N_sim)

            # Minutes points
            points[M > 0]  += 1
            points[M >= 60] += 1

            # Goals and assists
            points += 4*G + 3*A

            # Defensive contributions / saves
            if position in ["Defender", "Midfielder"]:
                points += 2 * (X >= 12)
            elif position == "GK":
                points += X  # 1 point per save, adjust if needed

            # Clean sheets
            if position in ["Defender", "Midfielder", "Goalkeeper"]:
                CS_prob = np.exp(-gc_eff)
                CS_draws = np.random.binomial(1, CS_prob)
                cs_points = 4 if position in ["Defender", "Goalkeeper"] else 1
                points += CS_draws * cs_points * played_60

            # Cards
            points += -1 * YC - 3 * RC

            points_weeks[:, w] = points

        total_points = points_weeks.sum(axis=1)

        results[player_id] = {
            "points_weeks": points_weeks,
            "total_points": total_points,
            "mean": np.mean(total_points),
            "median": np.median(total_points),
            "p25": np.percentile(total_points, 25),
            "p75": np.percentile(total_points, 75),
            "p90": np.percentile(total_points, 90)
        }

    return results