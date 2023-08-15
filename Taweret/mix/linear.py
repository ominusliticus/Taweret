# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a
# variance.
#
# Modified by K. Ingles

import _mypackage

import numpy as np

from multiprocessing import cpu_count

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from pathlib import Path

from pymc import Model as PyMCModel
from pymc import sample as pymc_sample

from Taweret.core.base_mixer import BaseMixer
# option to wrap BaseModel in PyTensors?
from Taweret.core.base_model import BaseModel

eps = 1.0e-20


# Should all numpy arrays by pytensor objects?


class LinearMixerGlobal(BaseMixer):
    def __init__(
        self,
        pymc_model: PyMCModel,
        inference_parameter_names: List[str],
        nargs_for_each_model: Optional[List[int]] = None,
        n_mix: int = 2,
    ):
        self.pymc_model = pymc_model

    def evaluate(self) -> np.ndarray:
        return NotImplemented

    def evaluate_weights(self):
        return NotImplemented

    @property
    def map(self):
        return NotImplemented

    def predict(self):
        return NotImplemented

    def predict_weights(self):
        return NotImplemented

    def prior_predict(self):
        return NotImplemented

    @property
    def posterior(self):
        return NotImplemented

    @property
    def prior(self):
        return NotImplemented

    def set_prior(self):
        return NotImplemented

    def train(
        self,
        outdir: Path,
        kwargs_for_sampler: Dict[str, Any],
        model_parameters: Optional[List[List[Any]]] = None,
        steps: int = 2000,
        burn: int = 50,
        walkers: int = 20,
    ):
        """
        Run sampler to learn weights (and model parameters). Method should also
        create class member variables that store the posterior and other
        diagnostic quantities important for plotting.MAP values should also
        caluclate and set as member variable of class

        Parameters:
        -----------
        y_exp : np.ndarray
            experimental observables to compare models with
        y_err : np.ndarray
            gaussian error bars on observables
        model_parameters : Optional[List[List[Any]]]]
            dictionary which contains list of model parameters for each model
        kwargs_for_sampler: Optional[Dict[str, Any]]
        steps: int
            Number of steps for MCMC per model (defaults to 2000)
        burn: int
            Number of burn-in steps for MCMC per model (defaults to 50)
        walkers: int
            Number of walkers per model (defaults to 20)

        Return:
        -------
        self.m_posterior : np.ndarray
            the mcmc chain return from sampler
        """
        if kwargs_for_sampler is None:
            raise Exception("Need to proivde default config")

        with self.pymc_model:
            idata = pymc_sample(**kwargs_for_sampler)

        return idata
