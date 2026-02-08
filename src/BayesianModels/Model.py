"""Abstract Base Class for a Hierarchical Bayesian Model."""
from abc import ABC, abstractmethod
from arviz import InferenceData


class Model(ABC):
    "ABC for a Bayesian Hierarchical Model."

    @abstractmethod
    def fit(**model_args) -> InferenceData:
        """Fit method for an Abstract Model.
        
        Args:
            model_args: a dictionary containing goals, assists, DC... etc. observation arrays
            
        Returns:
            InferenceData: an Arviz model trace object
        """
        pass