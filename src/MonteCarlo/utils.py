"""Utility functions for the Monte Carlo simulation."""

import numpy as np
from typing import Optional

import arviz as az


def extract_posteriors(
    trace: az.InferenceData,
    idx_map: dict[int, int],
    position: str,
    minutes_params_gk: dict[int, tuple],
    minutes_params_def: dict[int, tuple],
    minutes_params_mid: dict[int, tuple],
    minutes_params_str: dict[int, tuple],
) -> dict[int, dict]:
    """Extract posterior samples and parameters for each player from the trace.

    Args:
        trace (az.InferenceData): The trace object containing posterior samples.
        idx_map (dict[int, int]): Dictionary mapping player IDs to their corresponding indices in the trace.
        position (str): The position of the players ("Goalkeeper", "Defender", "Midfielder", "Forward").
        minutes_params_gk (dict[int, tuple]): Dictionary mapping goalkeeper player IDs to their minutes beta parameters.
        minutes_params_def (dict[int, tuple]): Dictionary mapping defender player IDs to their minutes beta parameters.
        minutes_params_mid (dict[int, tuple]): Dictionary mapping midfielder player IDs to their minutes beta parameters.
        minutes_params_str (dict[int, tuple]): Dictionary mapping forward player IDs to their minutes beta parameters.

    Returns:
        dict[int, dict]: Dictionary mapping player IDs to their posterior samples and parameters.
    """
    posteriors = {}
    for pid, i in idx_map.items():
        # Determine minutes_params based on position
        if position == "Goalkeeper":
            params_dict = minutes_params_gk
        elif position == "Defender":
            params_dict = minutes_params_def
        elif position == "Midfielder":
            params_dict = minutes_params_mid
        elif position == "Forward":
            params_dict = minutes_params_str
        else:
            # Handle unexpected positions if necessary, or raise an error
            continue  # Skip this player if position is unknown

        alpha_0, beta_0, alpha_60, beta_60 = params_dict[pid]

        # Access 'log_lambda' from the trace and exponentiate to get rates
        # The shape of trace.posterior["log_lambda"].values is typically (chain, draw, player_idx, latent_rate_idx)
        log_lambda_samples = trace.posterior["log_lambda"].values[:, :, i, :]

        posteriors[pid] = {
            "lambda_g": np.exp(log_lambda_samples[:, :, 0]).flatten(),
            "lambda_a": np.exp(log_lambda_samples[:, :, 1]).flatten(),
            "lambda_x": np.exp(log_lambda_samples[:, :, 2]).flatten(),  # dc or saves
            "lambda_gc": np.exp(log_lambda_samples[:, :, 3]).flatten(),
            "lambda_yc": np.exp(log_lambda_samples[:, :, 4]).flatten(),
            "lambda_rc": np.exp(log_lambda_samples[:, :, 5]).flatten(),
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "alpha_60": alpha_60,
            "beta_60": beta_60,
            "position": position,
        }
    return posteriors
