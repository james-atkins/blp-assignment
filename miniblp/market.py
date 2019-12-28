from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy import linalg

from .common import Vector, Theta2, Matrix
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

    def compute_mu(self, theta2: Theta2) -> Matrix:
        random_coefficients = theta2.sigma @ self.individuals.nodes.T
        if self.individuals.D > 0:
            random_coefficients += theta2.pi @ self.individuals.demographics.T

        return self.products.X2 @ random_coefficients

    def compute_choice_probabilities(self, delta: Vector, theta2: Theta2) -> Matrix:
        mu = self.compute_mu(theta2)
        return _compute_choice_probabilities(delta, mu)

    def compute_market_shares(self, delta: Vector, theta2: Theta2):
        mu = self.compute_mu(theta2)
        return _compute_market_shares(delta, mu, self.individuals.weights)

    def compute_delta(self, mu: Matrix, iteration: Iteration, initial_delta: Vector) -> IterationResult:
        """ Compute the mean utility for this market that equates observed and predicted market shares. """

        #  Use closed form solution if no heterogeneity
        if self.products.K2 == 0:
            return IterationResult(self.logit_delta)
        else:
            log_market_shares = self.log_market_shares
            individual_weights = self.individuals.weights

            def contraction(delta: Vector) -> Vector:
                computed_market_shares = _compute_market_shares(delta, mu, individual_weights)
                return delta + log_market_shares - np.log(computed_market_shares)

            return iteration.iterate(initial_delta, contraction)

    def solve_demand(self, initial_delta: Vector, theta2: Theta2, iteration: Iteration, compute_jacobian: bool) -> Tuple[IterationResult, Optional[Matrix]]:
        # Solve the contraction mapping
        mu = self.compute_mu(theta2)
        result = self.compute_delta(mu, iteration, initial_delta)

        # Compute the Jacobian
        if result.success and compute_jacobian:
            jacobian = self.compute_delta_by_theta_jacobian(theta2, result.final_delta, mu)
        else:
            jacobian = None

        return result, jacobian

    def compute_delta_by_theta_jacobian(self, theta2: Theta2, delta: Vector, mu: Matrix) -> Matrix:
        choice_probabilities = _compute_choice_probabilities(delta, mu)
        shares_by_delta_jacobian = self.compute_share_by_delta_jacobian(choice_probabilities)
        shares_by_theta_jacobian = self.compute_share_by_theta_jacobian(choice_probabilities, theta2)
        return linalg.solve(shares_by_delta_jacobian, -shares_by_theta_jacobian)

    def compute_share_by_delta_jacobian(self, choice_probabilities):
        """ Compute the Jacobian of market shares with respect to delta. """
        diagonal_shares = np.diagflat(self.products.market_shares)
        weighted_probabilities = self.individuals.weights[:, np.newaxis] * choice_probabilities.T
        return diagonal_shares - choice_probabilities @ weighted_probabilities

    def compute_share_by_theta_jacobian(self, choice_probabilities: Vector, theta2: Theta2):
        """ Compute the Jacobian of market shares with respect to theta. """
        jacobian = np.empty(shape=(self.products.J, theta2.P))

        for p, parameter in enumerate(theta2.unfixed):
            v = parameter.agent_characteristic(self)
            x = parameter.product_characteristic(self)
            jacobian[:, p] = (choice_probabilities * v.T * (x - x.T @ choice_probabilities)) @ self.individuals.weights

        return jacobian


@njit
def _compute_choice_probabilities(delta: Vector, mu: Matrix) -> Matrix:
    """
    Compute choice probabilities

    Uses the the log-sum-exp trick, which is inspired from the pyblp code, translated to numba.
    """
    # J x I array
    utilities = np.expand_dims(delta, axis=1) + mu

    # Loop is equivalent to np.clip(utilities.max(axis=0, keepdims=True), 0, None)
    I, J = utilities.shape
    utility_reduction = np.zeros(J)
    for j in range(J):
        for i in range(I):
            if utilities[i, j] > utility_reduction[j]:
                utility_reduction[j] = utilities[i, j]

    utilities -= utility_reduction
    exp_utilities = np.exp(utilities)
    scale = np.exp(-utility_reduction)

    return exp_utilities / (scale + np.sum(exp_utilities, axis=0))


@njit
def _compute_market_shares(delta: Vector, mu: Matrix, individual_weights: Vector) -> Vector:
    choice_probabilities = _compute_choice_probabilities(delta, mu)
    return choice_probabilities @ individual_weights  # Integrate over agents to calculate the market share
