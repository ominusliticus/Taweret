import _mypackage

from Taweret.mix.linear import LinearMixerLocal
from Taweret.core.base_model import BaseModel

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from typing import Union

import my_plotting as mp


class TruthModel:
    def __init__(self, switching_point: float = 0.0):
        self.switching_point = switching_point

    def _evaluate(self, x: float) -> float:
        if x < self.switching_point:
            return np.cos(2 * np.pi * (x - self.switching_point))
        else:
            return np.exp(-(x - self.switching_point))

    def evaluate(
            self,
            x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return np.array([self._evaluate(entry) for entry in x])


class Model_Cos(BaseModel):
    def __init__(self):
        self.m_prior = None

    def evaluate(self, x: np.ndarray):
        return np.cos(2 * np.pi * x)

    def log_likelihood_elementwise(
            self,
            x_exp: np.ndarray,
            y_exp: np.ndarray,
            y_err: np.ndarray,
    ) -> np.ndarray:
        result = np.sum((self.evaluate(x_exp) - y_exp) ** 2 / y_err ** 2)
        result += np.sum(np.log(2 * np.pi * y_err ** 2))
        return -0.5 * result

    def set_prior(self):
        pass


class Model_Exp(BaseModel):
    def __init__(self):
        self.m_prior = None

    def evaluate(self, x: np.ndarray):
        return np.exp(-x)

    def log_likelihood_elementwise(
            self,
            x_exp: np.ndarray,
            y_exp: np.ndarray,
            y_err: np.ndarray,
    ) -> np.ndarray:
        result = np.sum((self.evaluate(x_exp) - y_exp) ** 2 / y_err ** 2)
        result += np.sum(np.log(2 * np.pi * y_err ** 2))
        return -0.5 * result

    def set_prior(self):
        pass


def test_with_bessel(number_data_points: int = 10):
    truth = TruthModel(switching_point=-1.0)

    x_exp = np.linspace(-5, 5, number_data_points)
    y_exp = truth.evaluate(x_exp)
    # y_exp = np.array([norm(loc=ybar, scale=0.05).rvs() for ybar in y_exp])
    y_err = np.full_like(fill_value=0.05, a=y_exp)

    x_true = np.linspace(-5, 5, 100)
    y_true = truth.evaluate(x_true)

    path = Path(__file__).parent.absolute()
    if not (path / 'plots').exists():
        (path / 'plots').mkdir()

    mixing_model = LinearMixerLocal(
        models={'model 1': Model_Cos(), 'model 2': Model_Exp()},
        n_mix=2
    )
    mixing_model.set_prior(
        example_local_variable=np.array([0]),
        local_variables_ranges=np.array([[-10, 10]]),
        deterministic_priors=True,
        polynomial_order=2
    )
    posterior = mixing_model.train(
        y_exp=y_exp,
        y_err=y_err,
        outdir=path / 'plots',
        local_variables=x_exp,
        # burn=200,
        steps=2_000,
        thinning=1,
    )

    fig1, ax1 = plt.subplots(ncols=4, nrows=1, figsize=(4 * 7, 7))
    fig1.patch.set_facecolor('white')
    labels = [
        r'$\mu_{(1,0)}$',
        r'$\sigma_{(1,0)}$',
        r'$\mu_{(1,0)}$',
        r'$\sigma_{(1,0)}$'
    ]
    for i in range(posterior.shape[0]):
        ax1[i].hist(posterior[i], histtype='step', label=labels[i])
        mp.costumize_axis(ax1[i], '', r'$P($' + labels[i] + r'$)$')
        ax1[i].set_ylim(-2, 2)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    fig2.patch.set_facecolor('white')
    ax2.plot(x_true, y_true, lw=2, color='black', label='truth')
    output = np.array(
        [
            mixing_model.evaluate(
                local_variables=np.array([x]),
                sample=dict(
                    (key, val)
                    for key, val in zip(mixing_model.prior.keys(),
                                        mixing_model.map)
                )
            )
            for x in x_true
        ]
    )
    ax2.plot(x_true, output, lw=2, color='red', label='map', ls='dashed')

    prediction = mixing_model.predict(local_variables=x_true)
    ax2.plot(x_true, prediction[2],
             lw=2, color='blue', label='mean', ls='dotted')
    ax2.fill_between(
        x_true,
        prediction[2] + 2 * prediction[3],
        prediction[2] - 2 * prediction[3],
        color='blue',
        alpha=0.5,
        label=r'95\% post'
    )
    mp.costumize_axis(ax2, r'$x$', r'$f(x)$')
    ax2.legend(fontsize=20)
    fig2.tight_layout()
    fig2.savefig('plots/test_local_1_output.pdf')

    fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    fig3.patch.set_facecolor('white')
    weights = np.array(
        [
            mixing_model.evaluate_weights(
                local_variables=np.array([x]),
                sample=dict(
                    (key, val)
                    for key, val in zip(mixing_model.prior.keys(),
                                        mixing_model.map)
                )
            )
            for x in x_true
        ]
    )
    ax3.plot(x_true, weights[:, 0], lw=2, color='red', label=r'$w_{\cos}$')
    ax3.plot(x_true, weights[:, 1], lw=2, color='blue', label=r'$w_{\exp}$')
    ax3.legend(fontsize=20)
    fig3.tight_layout()
    fig3.savefig('plots/test_local_1_weights.pdf')


if __name__ == "__main__":
    test_with_bessel(number_data_points=40)
