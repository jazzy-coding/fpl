"""Module responsible for modelling Defenders in a Bayesian framework."""

import numpy as np
import pymc as pm
import arviz as az


def fit_def_model(
    goals: np.ndarray,
    assists: np.ndarray,
    defensive_contributions: np.ndarray,
    goals_conceded: np.ndarray,
    minutes: np.ndarray,
    yellow_cards: np.ndarray,
    red_cards: np.ndarray,
    n_defenders: int,
    player_idx: np.ndarray,
) -> az.InferenceData:
    """Fit a Bayesian model for defenders using PyMC.

    Args:
        goals (np.ndarray): Array of goals scored by each defender.
        assists (np.ndarray): Array of assists made by each defender.
        defensive_contributions (np.ndarray): Array of defensive contributions by each defender.
        goals_conceded (np.ndarray): Array of goals conceded by each defender.
        minutes (np.ndarray): Array of minutes played by each defender.
        yellow_cards (np.ndarray): Array of yellow cards received by each defender.
        red_cards (np.ndarray): Array of red cards received by each defender.
        n_defenders (int): Number of defenders in the dataset.
        player_idx (np.ndarray): Index array mapping observations to defenders.

    Returns:
        az.InferenceData: The trace of the fitted model.
    """
    with pm.Model() as def_model:

        # 6 latent rates now:
        # goals, assists, defensive contributions, goals conceded, yellow cards, red cards
        mu = pm.Normal("mu", 0.0, 1.5, shape=6)
        sigma = pm.HalfNormal("sigma", [1.0, 1.0, 1.0, 1.0, 0.25, 0.1])

        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(n_defenders, 6))

        log_lambda = pm.Deterministic("log_lambda", mu + z * sigma)

        lambda_g = log_lambda[:, 0]
        lambda_a = log_lambda[:, 1]
        lambda_dc = log_lambda[:, 2]
        lambda_gc = log_lambda[:, 3]
        lambda_yc = log_lambda[:, 4]
        lambda_rc = log_lambda[:, 5]

        exposure = minutes / 90.0

        pm.Poisson("goals", mu=np.exp(lambda_g[player_idx]) * exposure, observed=goals)

        pm.Poisson(
            "assists", mu=np.exp(lambda_a[player_idx]) * exposure, observed=assists
        )

        pm.Poisson(
            "dc",
            mu=np.exp(lambda_dc[player_idx]) * exposure,
            observed=defensive_contributions,
        )

        pm.Poisson(
            "goals_conceded",
            mu=np.exp(lambda_gc[player_idx]) * exposure,
            observed=goals_conceded,
        )

        pm.Poisson(
            "yellow_cards",
            mu=np.exp(lambda_yc[player_idx]) * exposure,
            observed=yellow_cards,
        )

        pm.Poisson(
            "red_cards", mu=np.exp(lambda_rc[player_idx]) * exposure, observed=red_cards
        )

        def_trace = pm.sample(2000, tune=2000, target_accept=0.95)

    return def_trace
