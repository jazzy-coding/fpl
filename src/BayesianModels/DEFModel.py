"""Module responsible for modelling Defenders in a Bayesian framework."""

import numpy as np
import pymc as pm


def fit_def_model(
    goals,
    assists,
    defensive_contributions,
    goals_conceded,
    minutes,
    yellow_cards,
    red_cards,
    n_defenders,
    player_idx
):

  with pm.Model() as def_model:

    # 6 latent rates now:
    # goals, assists, defensive contributions, goals conceded, yellow cards, red cards
    mu = pm.Normal("mu", 0.0, 1.5, shape=6)
    sigma = pm.HalfNormal("sigma", [1.0, 1.0, 1.0, 1.0, 0.25, 0.1])

    z = pm.Normal(
    "z",
    mu=0.0,
    sigma=1.0,
    shape=(n_defenders, 6)
    )

    log_lambda = pm.Deterministic(
        "log_lambda",
        mu + z * sigma
        )

    lambda_g  = log_lambda[:, 0]
    lambda_a  = log_lambda[:, 1]
    lambda_dc = log_lambda[:, 2]
    lambda_gc = log_lambda[:, 3]
    lambda_yc = log_lambda[:, 4]
    lambda_rc = log_lambda[:, 5]

    exposure = minutes / 90.0

    pm.Poisson(
        "goals",
        mu=np.exp(lambda_g[player_idx]) * exposure,
        observed=goals
    )

    pm.Poisson(
        "assists",
        mu=np.exp(lambda_a[player_idx]) * exposure,
        observed=assists
    )

    pm.Poisson(
        "dc",
        mu=np.exp(lambda_dc[player_idx]) * exposure,
        observed=defensive_contributions
    )

    pm.Poisson(
        "goals_conceded",
        mu=np.exp(lambda_gc[player_idx]) * exposure,
        observed=goals_conceded
    )

    pm.Poisson(
        "yellow_cards",
        mu=np.exp(lambda_yc[player_idx]) * exposure,
        observed=yellow_cards
    )

    pm.Poisson(
        "red_cards",
        mu=np.exp(lambda_rc[player_idx]) * exposure,
        observed=red_cards
    )

    def_trace = pm.sample(2000, tune=2000, target_accept=0.95)

    return def_trace