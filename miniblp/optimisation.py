from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Union, Tuple, List

import scipy.optimize

from .common import Matrix, Vector

ObjectiveFunction = Callable[[Vector, bool], Union[Tuple[float, Matrix], float]]
Bounds = List[Tuple[Optional[float], Optional[float]]]

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
    def optimise(self, objective_function: ObjectiveFunction, initial: Vector, bounds: Bounds) -> OptimisationResult:
        raise NotImplementedError


class SciPyOptimisation(Optimisation):
    def __init__(self, method: str, supports_jacobian: bool, supports_bounds: bool, **kwargs):
        self._method = method
        self._jacobian = supports_jacobian
        self._bounds = supports_bounds
        self._options = kwargs

    def optimise(self, objective_function: ObjectiveFunction, initial: Vector, bounds: Bounds) -> OptimisationResult:
        result = scipy.optimize.minimize(
            objective_function,
            initial,
            args=(self._jacobian,),
            method=self._method,
            jac=self._jacobian,
            options=self._options,
            bounds=bounds if self._bounds else None
        )

        return OptimisationResult(
            success=result.success,
            solution=result.x,
            objective=result.fun,
            jacobian=result.jac if self._jacobian is True else None,
            termination_message=result.message,
            number_iterations=result.nit,
            number_evaluations=result.nfev
        )


class NelderMead(SciPyOptimisation):
    def __init__(self, **kwargs):
        super().__init__("Nelder-Mead", supports_jacobian=False, supports_bounds=False, **kwargs)


class Powell(SciPyOptimisation):
    def __init__(self, **kwargs):
        super().__init__("Powell", supports_jacobian=False, supports_bounds=False, **kwargs)


class BFGS(SciPyOptimisation):
    def __init__(self, **kwargs):
        super().__init__("BFGS", supports_jacobian=True, supports_bounds=False, **kwargs)


class LBFGSB(SciPyOptimisation):
    def __init__(self, **kwargs):
        super().__init__("L-BFGS-B", supports_jacobian=True, supports_bounds=True, **kwargs)