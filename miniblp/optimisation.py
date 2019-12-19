from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Union, Tuple

import scipy.optimize

from .common import Matrix, Vector

ObjectiveFunction = Callable[[Vector, bool], Union[Tuple[float, Matrix], float]]


@dataclass
class OptimisationResult:
    success: bool
    solution: Vector
    objective: float
    jacobian: Optional[Matrix]
    termination_message: Optional[str]
    number_iterations: int
    number_evaluations: int


class Optimisation:
    def optimise(self, objective_function: ObjectiveFunction, initial: Vector) -> OptimisationResult:
        raise NotImplementedError


class SciPyOptimisation(Optimisation):
    def __init__(self, method: str, **kwargs):
        if method not in ("Nelder-Mead", "BFGS"):
            raise ValueError("Invalid optimisation method.")
        self._method = method

        if method in ("BFGS",):
            self._compute_jacobian = True
        else:
            self._compute_jacobian = False

        self._options = kwargs

    def optimise(self, objective_function: ObjectiveFunction, initial: Vector) -> OptimisationResult:
        result = scipy.optimize.minimize(objective_function, initial, args=(self._compute_jacobian,),
                                         method=self._method, jac=self._compute_jacobian, options=self._options)

        return OptimisationResult(
            success=result.success,
            solution=result.x,
            objective=result.fun,
            jacobian=result.jac if self._compute_jacobian is True else None,
            termination_message=result.message,
            number_iterations=result.nit,
            number_evaluations=result.nfev
        )
