from typing import Optional, List

import numpy as np
import pandas as pd
import scipy.linalg as linalg

from .integration import NumericalIntegration
from .iteration import Iteration
from .common import Matrix, Vector, Theta1, Theta2
from .data import Products, Individuals
from .iteration import Iteration
from .market import Market


class Problem:
    markets: List[Market]
    iteration: Iteration

    def __init__(self,
                 linear_coefficients: str,
                 random_coefficients: str,
                 instruments: str,
                 demographics_formula: str,
                 product_data: pd.DataFrame,
                 demographics_data: pd.DataFrame,
                 integration: NumericalIntegration,
                 iteration: Iteration):

        self.products = Products.from_formula(linear_coefficients, random_coefficients, instruments, product_data)
        self.individuals = Individuals.from_formula(demographics_formula, demographics_data, self.products, integration)
        self.markets = [Market(individuals, products) for products, individuals in zip(self.products.split_markets(), self.individuals.split_markets())]
        self.integration = integration
        self.iteration = iteration

        # Compute initial GMM weighting matrix. This is updated in the second GMM step.
        try:
            self.W = linalg.inv(self.products.Z.T @ self.products.Z)
        except linalg.LinAlgError:
            raise ValueError("Failed to compute the GMM weighting matrix.")

    def solve(self, weighting: Optional[Matrix] = None):
        pass

    def _compute_delta(self, theta2: Theta2, initial_delta: Optional[Vector] = None) -> Vector:
        """ Compute the mean utility that equates observed and predicted market shares. """
        # The market share inversion is independent for each market so each can be computed in parallel,
        # but for now this is done sequentially.
        deltas = [market.compute_delta(theta2, self.iteration, initial_delta) for market in self.markets]
        return np.concatenate(deltas)

    def _concentrate_out_linear_parameters(self, delta: Vector) -> Theta1:
        """
        We need to perform a non-linear search over θ. We can reduce the time required by expressing θ₁ as a function of θ₂,
        θ₁ = (X₁' Z W Z' X₁)⁻¹ X₁' Z W Z' 𝛿(θ₂)

        Now the non-linear search can be limited to θ₂.
        """
        # Rewrite the above equation and solve for θ₁ rather than inverting matrices
        # X₁' Z W Z' X₁ θ₁ = X₁' Z W Z' 𝛿(θ₂)

        # TODO: X1, Z and W are fixed. So we only need to compute common and a once.
        X1 = self.products.X1
        Z = self.products.Z
        common = X1.T @ Z @ self.W @ Z.T
        a = common @ X1
        b = common @ delta

        # W is positive definite as it is a GMM weighting matrix and (Z' X₁) has full rank so a = (Z' X₁)' W (Z' X₁)
        # is also positive definite
        return linalg.solve(a, b, assume_a="pos")

    def _objective_function(self, theta2: Theta2):
        # Compute the residual
        delta = self._compute_delta(theta2)
        theta1 = self._concentrate_out_linear_parameters(delta)
        omega = delta - self.products.X1 @ theta1

        Z = self.products.Z
        W = self.W

        # TODO: Check if undefined and put very high values

        return omega.T @ Z @ W @ Z.T @ omega
