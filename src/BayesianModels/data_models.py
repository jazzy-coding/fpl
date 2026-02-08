"""Pydantic Data Models for Bayesian Models."""

from typing import Annotated, Dict

import numpy as np
from pydantic import BaseModel, ConfigDict, AfterValidator

def validate_obs(array: np.ndarray) -> np.ndarray:
    """Validate observation arrays.
    
    Args:
        array (np.ndarray): an array of observations

    Returns:
        np.ndarray: a validated array of observations
    """
    if array.ndim != 1:
        raise ValueError('Observation array must be 1D.')
    if not np.isfinite(array).all():
        raise ValueError('NaN or Inf present in observation array.')
    return array


ObservationArray = Annotated[np.ndarray, AfterValidator(validate_obs)]


class Observations(BaseModel):
    """Base data model for observations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    player_observations: Dict[int, ObservationArray]

