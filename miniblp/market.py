from typing import Optional

import numpy as np

from .common import Vector, Theta2
from .data import Individuals, Products
from .iteration import Iteration, IterationResult


class Market:
    """ A market underlying the BLP model. """
    name: str
    individuals: Individuals
    products: Products

    log_market_shares: Vector
    logit_delta: Vector

    def __init__(self, individuals: Individuals, products: Products):
        market_ids = np.unique(individuals.market_ids)
        assert market_ids == np.unique(products.market_ids)
        assert len(market_ids) == 1

        self.name = market_ids[0]
        self.individuals = individuals
        self.products = products

        # Compute the closed form (logit) mean utility
        self.log_market_shares = np.log(self.products.market_shares)
        log_share_outside_option = np.log(1 - self.products.market_shares.sum())
        self.logit_delta = self.log_market_shares - log_share_outside_option

    def __repr__(self):
        return f"<Market: {self.name}>"

    def compute_mu(self, theta2: Theta2):
        pi = theta2.pi
        sigma = theta2.sigma

        random_coefficients = sigma @ self.individuals.nodes.T
        if self.individuals.D > 0:
            random_coefficients += pi @ self.individuals.demographics.T

        return self.products.X2 @ random_coefficients

    def compute_market_share(self, delta: Vector, theta2: Theta2) -> Vector:
        # J x I array
        exp_utility = self.compute_mu(theta2)
        exp_utility += delta[:, np.newaxis]
        np.exp(exp_utility, out=exp_utility)

        # I array
        denominator = np.sum(exp_utility, axis=0)
        denominator += 1

        # J x I array of the probability that agent i chooses product j
        choice_probabilities = exp_utility / denominator

        # Integrate over agents to calculate the market share
        if self.individuals.weights is None:
            return np.sum(choice_probabilities, axis=1) / self.individuals.I
        else:
            return np.sum(self.individuals.weights * choice_probabilities, axis=1)

    def compute_delta(self, theta2: Theta2, iteration: Iteration, initial_delta: Optional[Vector] = None) -> IterationResult:
        """ Compute the mean utility for this market that equates observed and predicted market shares. """

        # Use closed form solution if no heterogeneity
        if self.products.K2 == 0:
            return IterationResult(self.logit_delta)

        def contraction(delta: Vector) -> Vector:
            computed_market_shares = self.compute_market_share(delta, theta2)
            return delta + self.log_market_shares - np.log(computed_market_shares)

        if initial_delta is None:
            initial_delta = self.logit_delta

        return iteration.iterate(initial_delta, contraction)
