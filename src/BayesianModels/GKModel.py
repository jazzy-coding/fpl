"""Module responsible for modelling Goalkeepers in a Bayesian framework."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def fit_gk_model(
    goals,
    assists,
    saves,
    goals_conceded,
    minutes,
    yellow_cards,
    red_cards,
    n_goalkeepers,
    player_idx
):

  with pm.Model() as gk_model:

    # 6 latent rates:
    # goals, assists, saves, goals conceded, yellow cards, red cards
    mu = pm.Normal("mu", 0.0, 1.5, shape=6)

    # sigma has been tightened for yellow and red cards due to their low variance and rarity
    sigma = pm.HalfNormal("sigma", [1.0, 1.0, 1.0, 1.0, 0.25, 0.1])

    z = pm.Normal(
    "z",
    mu=0.0,
    sigma=1.0,
    shape=(n_goalkeepers, 6)
    )

    log_lambda = pm.Deterministic(
        "log_lambda",
        mu + z * sigma
        )

    lambda_g  = log_lambda[:, 0]
    lambda_a  = log_lambda[:, 1]
    lambda_s = log_lambda[:, 2]
    lambda_gc = log_lambda[:, 3]
    lambda_yc = log_lambda[:, 4]
    lambda_rc = log_lambda[:, 5]

    exposure = np.maximum(minutes / 90.0, 1e-6)

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
        "saves",
        mu=np.exp(lambda_s[player_idx]) * exposure,
        observed=saves
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

    gk_trace = pm.sample(2000, tune=2000, target_accept=0.95)

  return gk_trace
