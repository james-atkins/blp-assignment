from typing import Callable

import numpy as np
import scipy.linalg as linalg

from .common import Vector


Contraction = Callable[[Vector], Vector]


class ConvergenceError(Exception):
    pass


class Iteration:
    """ Fixed point iteration. """
    def iterate(self, x0: Vector, contraction: Contraction) -> Vector:
        raise NotImplementedError


class SimpleFixedPointIteration(Iteration):
    def __init__(self, max_iterations: int = 5000, tolerance: float = 1e-14):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def iterate(self, x0: Vector, contraction: Contraction) -> Vector:
        iterations: int = 0
        x = x0

        while True:
            if iterations > self.max_iterations:
                raise ConvergenceError("Exceeded the maximum number of iterations.")

            x_next = contraction(x)

            if not np.isfinite(x_next).all():
                raise ConvergenceError("Numerical issues detected.")

            # TODO: Phased tolerance as suggested by Nevo?
            if linalg.norm(x - x_next, ord=np.inf) < self.tolerance:
                break

            x = x_next
            iterations += 1

        return x_next


