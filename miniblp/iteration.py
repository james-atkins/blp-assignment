from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numba import njit

from .common import Vector

Contraction = Callable[[Vector], Vector]


@dataclass
class IterationResult:
    final_delta: Vector
    iterations: int = 0
    evaluations: int = 0
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_message is None


class Iteration:
    """ Fixed point iteration. """
    def iterate(self, x0: Vector, contraction: Contraction) -> IterationResult:
        raise NotImplementedError


class SimpleFixedPointIteration(Iteration):
    def __init__(self, max_iterations: int = 5000, tolerance: float = 1e-14):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def iterate(self, x0: Vector, contraction: Contraction) -> IterationResult:
        x = x0
        iterations: int = 0

        while True:
            if iterations > self.max_iterations:
                return IterationResult(x, iterations, iterations, "Exceeded the maximum number of iterations")

            x_next = contraction(x)
            iterations += 1

            if not np.isfinite(x_next).all():
                return IterationResult(x, iterations, iterations, "Numerical issues detected.")

            if within_tolerance(x, x_next, self.tolerance):
                return IterationResult(x_next, iterations, iterations)

            x = x_next


class PhasedToleranceIteration(Iteration):
    def __init__(self, max_iterations: int = 5000, tolerance_first_100: float = 1e-8):
        self.tolerance_first_100 = tolerance_first_100
        self.max_iterations = max_iterations

    def iterate(self, x0: Vector, contraction: Contraction) -> IterationResult:
        x = x0
        iterations: int = 0
        tolerance = self.tolerance_first_100

        while True:
            if iterations > self.max_iterations:
                return IterationResult(x, iterations, iterations, "Exceeded the maximum number of iterations")

            x_next = contraction(x)
            iterations += 1

            if not np.isfinite(x_next).all():
                return IterationResult(x, iterations, iterations, "Numerical issues detected.")

            if within_tolerance(x, x_next, tolerance):
                return IterationResult(x_next, iterations, iterations)

            x = x_next

            # Phrased tolerance as suggested by Nevo
            # The tolerance level increases by an order of 10 for every additional 50 iterations
            # after the first 100.
            if iterations >= 100 and iterations % 50 == 0:
                tolerance /= 10


class SQUAREMIteration(Iteration):
    """
    Apply the SQUAREM acceleration method for fixed point iteration.

    Based on the pyblp code, Varadhan and Roland (2008)
    """

    def __init__(self, max_evaluations: int = 5000, tolerance: float = 1e-14, scheme: int = 3, step_min: float = 1.0, step_max: float = 1.0, step_factor: float = 4.0):
        self.max_evaluations = max_evaluations
        self.tolerance = tolerance
        self.scheme = scheme
        self.step_min = step_min
        self.step_max = step_max
        self.step_factor = step_factor

    def iterate(self, x0: Vector, contraction: Contraction) -> IterationResult:
        step_min = self.step_min
        step_max = self.step_max
        x = x0
        evaluations = 0
        iterations = 0

        while True:
            # first step
            x0, x = x, contraction(x)
            if not np.isfinite(x).all():
                return IterationResult(x, iterations, evaluations, "Numerical issues detected.")

            # check for convergence
            g0 = x - x0
            evaluations += 1

            if evaluations > self.max_evaluations:
                return IterationResult(x, iterations, evaluations, "Exceeded the maximum number of evaluations")

            if within_tolerance(x0, x, self.tolerance):
                return IterationResult(x, iterations, evaluations)

            # second step
            x1, x = x, contraction(x)
            if not np.isfinite(x).all():
                return IterationResult(x, iterations, evaluations, "Numerical issues detected.")

            # check for convergence
            g1 = x - x1
            evaluations += 1

            if evaluations > self.max_evaluations:
                return IterationResult(x, iterations, evaluations, "Exceeded the maximum number of evaluations")

            if within_tolerance(x1, x, self.tolerance):
                return IterationResult(x, iterations, evaluations)

            # compute the step length
            r = g0
            v = g1 - g0
            if self.scheme == 1:
                alpha = (r.T @ v) / (v.T @ v)
            elif self.scheme == 2:
                alpha = (r.T @ r) / (r.T @ v)
            else:
                alpha = -np.sqrt((r.T @ r) / (v.T @ v))

            # bound the step length and update its bounds
            alpha = -np.maximum(self.step_min, np.minimum(step_max, -alpha))
            if -alpha == step_max:
                step_max *= self.step_factor
            if -alpha == step_min and step_min < 0:
                step_min *= self.step_factor

            # acceleration step
            x2, x = x, x0 - 2 * alpha * r + alpha ** 2 * v
            x3, x = x, contraction(x)
            if not np.isfinite(x).all():
                return IterationResult(x, iterations, evaluations, "Numerical issues detected.")

            iterations += 1

            # check for convergence
            evaluations += 1
            if evaluations > self.max_evaluations:
                return IterationResult(x, iterations, evaluations, "Exceeded the maximum number of evaluations")

            if within_tolerance(x, x - x3, self.tolerance):
                return IterationResult(x, iterations, evaluations)


class ReturnIteration(Iteration):
    def iterate(self, x0: Vector, contraction: Contraction) -> IterationResult:
        return IterationResult(x0, 0, 0)


@njit
def within_tolerance(x: Vector, x_next: Vector, tolerance: float) -> bool:
    """
    Tolerance checking using the sup norm.

    Quicker than scipy.linalg.norm as it short circuits."""
    for i, j in zip(x, x_next):
        if abs(i - j) >= tolerance:
            return False

    return True
