from typing import Callable, NamedTuple, Optional

import numpy as np
import scipy.optimize

from .common import Matrix, Vector, Theta2


class ObjectiveResult(NamedTuple):
    objective: float
    gradient: Optional[Matrix]


class Optimisation:
    """ Optimisation of the GMM objective function. """

    # values, iterations, evaluations
    # verbose_objective_function: Callable[[Array, int, int], ObjectiveResult])


# objective_function(theta) -> (objective, gradient)
ObjectiveFunction: Callable[[Vector, ...], float]
JacobianFunction: Callable[[Vector, ...], Vector]


class NelderMeadOptimisation(Optimisation):

    objective_function: Callable[[Vector, ...], float]
    initial_guess: Vector
    jacobian: Callable[[Vector, ...], Vector]

    def optimise(self, objective_function: ObjectiveFunction):
        # Objective function to be minimised: f(x, *args) -> float

        pass

