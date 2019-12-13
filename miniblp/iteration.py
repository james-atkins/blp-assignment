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
                return IterationResult(x, iterations, "Exceeded the maximum number of iterations")

            x_next = contraction(x)
            iterations += 1

            if not np.isfinite(x_next).all():
                return IterationResult(x, iterations, "Numerical issues detected.")

            if within_tolerance(x, x_next, self.tolerance):
                return IterationResult(x_next, iterations)

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
                return IterationResult(x, iterations, "Exceeded the maximum number of iterations")

            x_next = contraction(x)
            iterations += 1

            if not np.isfinite(x_next).all():
                return IterationResult(x, iterations, "Numerical issues detected.")

            if within_tolerance(x, x_next, tolerance):
                return IterationResult(x_next, iterations)

            x = x_next

            # Phrased tolerance as suggested by Nevo
            # The tolerance level increases by an order of 10 for every additional 50 iterations
            # after the first 100.
            if iterations >= 100 and iterations % 50 == 0:
                tolerance /= 10

@njit
def within_tolerance(x: Vector, x_next: Vector, tolerance: float) -> bool:
    """
    Tolerance checking using the sup norm.

    Quicker than scipy.linalg.norm as it short circuits."""
    for i, j in zip(x, x_next):
        if abs(i - j) >= tolerance:
            return False

    return True
