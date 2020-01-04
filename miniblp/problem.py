import functools
import io
import math
import multiprocessing
import time
from typing import Optional, List, Tuple, Union, NamedTuple

import numpy as np
import pandas as pd
import scipy.linalg as linalg

from .common import Matrix, Vector, Theta1, Theta2
from .data import Products, Individuals, ProductFormulation, DemographicsFormulation
from .integration import NumericalIntegration, MonteCarloIntegration
from .iteration import Iteration, SimpleFixedPointIteration
from .market import Market
from .optimisation import Optimisation, OptimisationResult, SciPyOptimisation

FLOAT_FORMAT_STRING = "{:.2E}"


class GMMStepResult:
    problem: "Problem"
    success: bool
    optimisation_result: OptimisationResult
    theta1: Theta1
    theta2: Theta2
    omega: Vector
    delta: Vector
    # fixed_point_iterations: int
    # contraction_evaluations: int
    beta_estimates: pd.DataFrame
    sigma_estimates: pd.DataFrame
    pi_estimates: Optional[pd.DataFrame]

    def __init__(self, problem: "Problem", final_progress: "Progress", optimisation_result: OptimisationResult):
        self.problem = problem
        self.success = optimisation_result.success
        self.optimisation_result = optimisation_result
        self.theta1 = final_progress.theta1
        self.theta2 = final_progress.theta2
        self.omega = final_progress.omega
        self.delta = final_progress.delta

        # TODO: Standard errors
        self.beta_estimates = pd.DataFrame(
            self.theta1,
            index=self.problem.products.X1.column_names,
            columns=["estimate"]
        )

        x2_column_names = self.problem.products.X2.column_names
        self.sigma_estimates = pd.DataFrame(self.theta2.sigma, index=x2_column_names, columns=x2_column_names)

        if self.theta2.pi is None:
            self.pi_estimates = None
        else:
            self.pi_estimates = pd.DataFrame(self.theta2.pi,
                                        index=x2_column_names,
                                        columns=self.problem.individuals.demographics.column_names)

    def compute_weighting_matrix_heteroscedasticity(self) -> Matrix:
        """ Compute heteroscedasticity robust weighting matrix for 2nd GMM step. """
        # g = self.omega.T @ self.problem.products.Z
        g = self.omega[:, np.newaxis] * self.problem.products.Z
        g -= g.mean(axis=0)  # center moments
        S = sum(np.c_[g_n] @ np.c_[g_n].T for g_n in g) / self.problem.N
        try:
            return linalg.inv(S)
        except linalg.LinAlgError:
            raise ValueError("Failed to invert matrix S.")

    def format(self) -> str:
        buffer = io.StringIO()

        def formatter(x: float) -> str:
            if x == 0:
                return ""
            return FLOAT_FORMAT_STRING.format(x)

        # TODO: GMM steps, objective value etc

        print("Linear Coefficients", file=buffer)
        self.beta_estimates.to_string(buffer, float_format=formatter)

        print("\n\nRandom Coefficients (Sigma)", file=buffer)
        self.sigma_estimates.to_string(buffer, float_format=formatter)

        if self.pi_estimates is not None:
            print("\n\nRandom Coefficients (Pi)", file=buffer)
            self.pi_estimates.to_string(buffer, float_format=formatter)

        buffer.seek(0)
        return buffer.read()

    def __repr__(self):
        return self.format()


class TwoStepGMMResult(GMMStepResult):
    first_step: GMMStepResult


OneStepGMMResult = GMMStepResult

ProblemResult = Union[OneStepGMMResult, TwoStepGMMResult]


class Progress(NamedTuple):
    min_objective_value: float
    objective_value: float
    fixed_point_iterations: int
    contraction_evaluations: int
    omega: Optional[Vector]
    delta: Vector
    gradient: Optional[Matrix]
    theta1: Optional[Theta1]
    theta2: Theta2

    def print_header(self):
        print("{:<10} {:>12} {:<8}".format("Objective value", "Fixed point iterations", "Contraction evaluations"))

    def print(self):
        print("{:.10E} {:>10d} {:>10d}".format(self.objective_value, self.fixed_point_iterations, self.contraction_evaluations))


class Problem:
    markets: List[Market]
    T: int  # Number of markets
    N: int  # Number of products across all markets
    K1: int  # Number of linear product characteristics
    K2: int  # Number of nonlinear product characteristics
    I: int  # Number of individuals
    D: int  # Number of demographic variables

    def __init__(self,
                 product_formulation: ProductFormulation,
                 product_data: pd.DataFrame,
                 demographics_formulation: Optional[DemographicsFormulation] = None,
                 demographics_data: Optional[pd.DataFrame] = None,
                 integration: Optional[NumericalIntegration] = None,
                 seed: Optional[int] = None):

        if integration is None:
            integration = MonteCarloIntegration()

        self.products = Products.from_formula(product_formulation, product_data)
        self.individuals = Individuals.from_formula(demographics_formulation, demographics_data, self.products,
                                                    integration, seed)
        self.markets = [Market(individuals, products) for products, individuals in
                        zip(self.products.split_markets(), self.individuals.split_markets())]

        self.T = len(self.markets)
        self.N = self.products.J
        self.K1 = self.products.K1
        self.K2 = self.products.K2
        self.I = self.individuals.I
        self.D = self.individuals.D
        self.MD = self.products.MD

    def solve(self, sigma: Matrix, pi: Optional[Matrix] = None, *, method: str = "2s",
              iteration: Optional[Iteration] = None,
              optimisation: Optional[Optimisation] = None,
              parallel: bool = True) -> ProblemResult:

        step_start_time = time.time()

        theta2 = Theta2(self, initial_sigma=sigma, initial_pi=pi)

        if iteration is None:
            iteration = SimpleFixedPointIteration()

        if optimisation is None:
            optimisation = SciPyOptimisation("BFGS")

        if method not in {"1s", "2s"}:
            raise ValueError("method must be 1s or 2s.")

        # Compute initial weighting matrix
        try:
            W = linalg.inv(self.products.Z.T @ self.products.Z / self.N)
        except linalg.LinAlgError:
            raise ValueError("Failed to compute the GMM weighting matrix.")

        initial_deltas = [market.logit_delta for market in self.markets]

        if parallel:
            pool = multiprocessing.Pool()
        else:
            pool = None

        try:
            print("Running GMM step 1...")
            result_stage_1 = self._gmm_step(
                pool,
                theta2=theta2,
                initial_deltas=initial_deltas,
                W=W,
                iteration=iteration,
                optimisation=optimisation
            )

            if method == "1s":
                return result_stage_1

            # In second GMM step use the computed deltas from the first step as initial values
            initial_deltas = np.split(result_stage_1.delta, np.cumsum([market.products.J for market in self.markets]))[:-1]

            print("Computing weighting matrix...")
            W = result_stage_1.compute_weighting_matrix_heteroscedasticity()

            print("Running GMM step 2...")
            result_stage_2 = self._gmm_step(
                pool,
                theta2=result_stage_1.theta2,
                initial_deltas=initial_deltas,
                W=W,
                iteration=iteration,
                optimisation=optimisation
            )

            return result_stage_2  # TODO: right type
        finally:
            if pool is not None:
                pool.terminate()

    def _gmm_step(self, pool: Optional[multiprocessing.Pool], theta2: Theta2, initial_deltas: List[Vector], W: Matrix,
                  iteration: Iteration,
                  optimisation: Optimisation) -> GMMStepResult:

        concentrate_out_linear_parameters = self._make_concentrator(W)

        def compute_progress(progress: Progress, theta2: Theta2, compute_jacobian: bool) -> Progress:
            deltas: List[Vector] = []
            jacobians: List[Matrix] = []

            iterations: int = 0
            evaluations: int = 0

            solve_demand = functools.partial(
                Market.solve_demand,
                theta2=theta2,
                iteration=iteration,
                compute_jacobian=compute_jacobian
            )

            if pool is None:
                results = [solve_demand(market, initial_delta) for (market, initial_delta) in zip(self.markets, initial_deltas)]
            else:
                # The market share inversion is independent for each market so each can be computed in parallel
                results = pool.starmap(solve_demand, zip(self.markets, initial_deltas))

            for result, jacobian in results:
                iterations += result.iterations
                evaluations += result.evaluations
                if not result.success:
                    # Punish if there are numerical issues or the contraction does not converge
                    return Progress(
                        min_objective_value=progress.min_objective_value,
                        objective_value=math.inf,
                        fixed_point_iterations=iterations,
                        contraction_evaluations=evaluations,
                        omega=progress.omega,
                        delta=progress.delta,
                        gradient=np.zeros((self.products.J, theta2.P)),
                        theta1=progress.theta1,
                        theta2=theta2
                    )
                else:
                    deltas.append(result.final_delta)
                    jacobians.append(jacobian)

            # Stack the deltas and Jacobians from all the markets
            delta = np.concatenate(deltas)
            jacobian = np.vstack(jacobians) if compute_jacobian else None

            theta1 = concentrate_out_linear_parameters(delta)
            omega = delta - self.products.X1 @ theta1
            g_mean = omega.T @ self.products.Z / self.N
            objective_value = g_mean @ W @ g_mean.T

            if jacobian is None:
                gradient = None
            else:
                G_mean = self.products.Z.T @ jacobian / self.N
                gradient = 2 * G_mean.T @ W @ g_mean

            return Progress(
                min_objective_value=min(objective_value, progress.min_objective_value),
                objective_value=objective_value,
                fixed_point_iterations=iterations,
                contraction_evaluations=evaluations,
                omega=omega,
                delta=delta,
                gradient=gradient,
                theta1=theta1,
                theta2=theta2
            )

        progress = Progress(
            min_objective_value=math.inf,
            objective_value=math.inf,
            fixed_point_iterations=0,
            contraction_evaluations=0,
            omega=None,
            delta=np.concatenate(initial_deltas),
            gradient=np.zeros((theta2.P, 1)),
            theta1=None,
            theta2=theta2
        )
        progress.print_header()

        def objective_wrapper(x: Vector, compute_jacobian: bool) -> Union[Tuple[float, Matrix], float]:
            nonlocal progress
            theta2.optimiser_parameters = x
            progress = compute_progress(progress, theta2, compute_jacobian)
            progress.print()
            if compute_jacobian:
                return progress.objective_value, progress.gradient
            else:
                return progress.objective_value

        optimisation_result = optimisation.optimise(objective_wrapper, theta2.optimiser_parameters, theta2.bounds)
        theta2.optimiser_parameters = optimisation_result.solution

        return GMMStepResult(self, progress, optimisation_result)

    def _make_concentrator(self, W: Matrix):
        x1_z_w_z = self.products.X1.T @ self.products.Z @ W @ self.products.Z.T
        a = x1_z_w_z @ self.products.X1

        def concentrate_out_linear_parameters(delta: Vector) -> Theta1:
            """
            We need to perform a non-linear search over θ. We reduce the time required by expressing θ₁
            as a function of θ₂: θ₁ = (X₁' Z W Z' X₁)⁻¹ X₁' Z W Z' 𝛿(θ₂)

            Now the non-linear search can be limited to θ₂.
            """
            # Rewrite the above equation and solve for θ₁ rather than inverting matrices
            # X₁' Z W Z' X₁ θ₁ = X₁' Z W Z' 𝛿(θ₂)
            b = x1_z_w_z @ delta

            # W is positive definite as it is a GMM weighting matrix and (Z' X₁) has full rank so
            # a = (Z' X₁)' W (Z' X₁) is also positive definite
            return linalg.solve(a, b, assume_a="pos")

        return concentrate_out_linear_parameters

    def __repr__(self):
        buffer = io.StringIO()

        print("Dimensions:", file=buffer)
        self._format_dimensions(buffer)

        print("\n", file=buffer)

        print("Formulations:", file=buffer)
        self._format_formulations(buffer)

        buffer.seek(0)
        return buffer.read()

    def _format_dimensions(self, buffer: io.StringIO):
        pd.Series({
            "T": self.T,
            "N": self.N,
            "I": self.I,
            "D": self.D,
            "K1": self.K1,
            "K2": self.K2,
            "MD": self.MD
        }).to_frame().T.to_string(buffer, index=False, justify="center")

    def _format_formulations(self, buffer: io.StringIO):
        formulations = {"x1: Linear Characteristics": pd.Series(self.products.X1.column_names),
                        "x2: Nonlinear Characteristics": pd.Series(self.products.X2.column_names)}

        if self.individuals.demographics is not None:
            formulations["d: Demographics"] = pd.Series(self.individuals.demographics.column_names)

        pd.DataFrame(formulations).fillna("").T.to_string(buffer, justify="center")
