import itertools
import math
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg as linalg
import scipy.optimize

from .integration import NumericalIntegration
from .optimisation import Optimisation, ObjectiveResult, OptimisationResult
from .common import Matrix, Vector, Theta1, Theta2, is_vector
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
            self._update_weighting_matrix(linalg.inv(self.products.Z.T @ self.products.Z / self.products.J))
        except linalg.LinAlgError:
            raise ValueError("Failed to compute the GMM weighting matrix.")

    def solve(self, initial_sigma: Matrix, initial_pi: Matrix, optimisation: Optimisation, initial_delta: Optional[Vector] = None) -> OptimisationResult:
        theta2 = Theta2(self, initial_sigma=initial_sigma, initial_pi=initial_pi)

        def objective_wrapper(x: Vector, compute_jacobian: bool) -> Union[ObjectiveResult, float]:
            theta2.optimiser_parameters = x
            result = self._objective_function(theta2, compute_jacobian, initial_delta)
            print(result.objective)
            if compute_jacobian:
                return result
            else:
                return result.objective

        return optimisation.optimise(objective_wrapper, theta2.optimiser_parameters)

    def _objective_function(self, theta2: Theta2, compute_jacobian: bool, initial_delta: Optional[Vector]) -> ObjectiveResult:
        if initial_delta is not None:
            assert is_vector(initial_delta)
            assert len(initial_delta) == self.products.J
            initial_deltas = np.split(initial_delta, np.cumsum([market.products.J for market in self.markets]))
            assert len(initial_deltas) == len(self.markets)
        else:
            initial_deltas = itertools.repeat(None)

        deltas: List[Vector] = []
        jacobians: List[Matrix] = []

        # The market share inversion is independent for each market so each can be computed in parallel,
        # but for now this is done sequentially.
        for market, initial_delta in zip(self.markets, initial_deltas):
            result, jacobian = market.solve_demand(theta2, self.iteration, compute_jacobian, initial_delta)
            if not result.success:
                # If there are numerical issues or the contraction does not converge
                return ObjectiveResult(1, np.zeros((self.products.J, theta2.P)))
            else:
                deltas.append(result.final_delta)
                jacobians.append(jacobian)

        # Stack the deltas and Jacobians from all the markets
        delta = np.concatenate(deltas)
        jacobian = np.vstack(jacobians) if compute_jacobian else None

        theta1 = self._concentrate_out_linear_parameters(delta)
        omega = delta - self.products.X1 @ theta1

        g = omega.T @ self.products.Z / self.products.J
        objective_value = g @ self.W @ g.T

        if jacobian is None:
            gradient = None
        else:
            G = self.products.Z.T @ jacobian / self.products.J
            gradient = 2 * G.T @ self.W @ g

        return ObjectiveResult(objective_value, gradient)

    def _update_weighting_matrix(self, weighting: Matrix):
        self.W = weighting
        self._x1_z_w_z = self.products.X1.T @ self.products.Z @ self.W @ self.products.Z.T

    def _concentrate_out_linear_parameters(self, delta: Vector) -> Theta1:
        """
        We need to perform a non-linear search over θ. We can reduce the time required by expressing θ₁ as a function of θ₂,
        θ₁ = (X₁' Z W Z' X₁)⁻¹ X₁' Z W Z' 𝛿(θ₂)

        Now the non-linear search can be limited to θ₂.
        """
        # Rewrite the above equation and solve for θ₁ rather than inverting matrices
        # X₁' Z W Z' X₁ θ₁ = X₁' Z W Z' 𝛿(θ₂)

        a = self._x1_z_w_z @ self.products.X1
        b = self._x1_z_w_z @ delta

        # W is positive definite as it is a GMM weighting matrix and (Z' X₁) has full rank so a = (Z' X₁)' W (Z' X₁)
        # is also positive definite
        return linalg.solve(a, b, assume_a="pos")