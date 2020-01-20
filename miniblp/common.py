from typing import Type, Iterable, Tuple, List, Any, Optional, Union

import numpy as np

Vector: Type[np.ndarray] = np.ndarray
Matrix: Type[np.ndarray] = np.ndarray

# TODO: Easier to use xarray rather than make own implementation?
class NamedMatrix:
    def __init__(self, data, column_names):
        _, num_columns = data.shape
        assert len(column_names) == num_columns
        self._data = data
        self.column_names = column_names

    def __array__(self):
        return self._data

    def __len__(self):
        return self._data.__len__()

    def __imul__(self, other):
        self._data *= other
        return self

    def __matmul__(self, other):
        return self._data.__matmul__(other)

    def plain(self) -> Matrix:
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return self._data.T


def is_vector(vector: np.ndarray) -> bool:
    return len(vector.shape) == 1


def is_matrix(matrix: Union[np.ndarray, NamedMatrix]) -> bool:
    return len(matrix.shape) == 2


def are_same_length(*arrays: Optional[Union[np.ndarray, NamedMatrix]]) -> bool:
    """ Check that the vectors and matrix in arrays have the same length. """
    iterator = filter(lambda array: array is not None, iter(arrays))
    try:
        first = len(next(iterator))
    except StopIteration:
        return True
    return all(first == len(rest) for rest in iterator)


Theta1: Type[np.ndarray] = np.ndarray


class Parameter:
    def __init__(self, index: Tuple[int, int]):
        self.index = index

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.index})>"

    def product_characteristic(self, market: "Market") -> Vector:
        return market.products.X2[:, [self.index[0]]]

    def agent_characteristic(self, market: "Market") -> Vector:
        """Get the agent characteristic associated with the parameter."""
        raise NotImplementedError


class SigmaParameter(Parameter):
    """ Information about a single parameter in sigma. """
    def agent_characteristic(self, market: "Market") -> Vector:
        return market.individuals.nodes[:, [self.index[1]]]


class PiParameter(Parameter):
    """ Information about a single parameter in pi. """
    def agent_characteristic(self, market: "Market") -> Vector:
        return market.individuals.demographics[:, [self.index[1]]]


class Theta2:
    """ Coefficients for the non-linear parameters. """

    def __init__(self, problem: "Problem", initial_sigma: Matrix, initial_pi: Optional[Matrix] = None):
        k2 = problem.products.K2
        if not (k2 == initial_sigma.shape[0] == initial_sigma.shape[1]):
            raise ValueError("sigma is not of the correct dimensions. Should be {} x {}".format(k2, k2))

        if problem.individuals.D == 0 and initial_pi is not None:
            raise ValueError("pi should be None - there are no demographics.")

        if problem.individuals.D > 0 and initial_pi is None:
            raise ValueError("Must specify an initial pi matrix.")

        if initial_pi is not None and (k2 != initial_pi.shape[0] or initial_pi.shape[1] != problem.individuals.D):
            raise ValueError("pi is not of the correct dimension. Should be {} x {}.".format(k2, problem.individuals.D))

        # Sigma should be lower triangle
        sigma = np.tril(initial_sigma)

        self.sigma = sigma
        self.pi = initial_pi

        self.fixed: List[Parameter] = []
        self.unfixed: List[Parameter] = []

        self._store(SigmaParameter, zip(*np.tril_indices_from(self.sigma)), zip(*np.nonzero(self.sigma)))
        if self.pi is not None:
            self._store(PiParameter, np.ndindex(self.pi.shape), zip(*np.nonzero(self.pi)))

        self.P = len(self.unfixed)

    @property
    def optimiser_parameters(self):
        """ Parameters to be optimised over. """
        params = np.empty(len(self.unfixed))

        for i, param in enumerate(self.unfixed):
            if isinstance(param, SigmaParameter):
                params[i] = self.sigma[param.index]
            elif isinstance(param, PiParameter):
                params[i] = self.pi[param.index]
            else:
                raise ValueError("Unknown parameter type")

        return params

    @optimiser_parameters.setter
    def optimiser_parameters(self, values):
        # The Powell solver sometimes gives zero dimensional inputs for some reason
        if values.ndim == 0:
            values = values[np.newaxis]

        if values.shape != (len(self.unfixed), ):
            raise ValueError("optimiser_parameters have the wrong shape.")

        for param, value in zip(self.unfixed, values):
            if isinstance(param, SigmaParameter):
                self.sigma[param.index] = value
            elif isinstance(param, PiParameter):
                self.pi[param.index] = value
            else:
                raise ValueError("Unknown parameter type")

    @property
    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """ (min, max) pairs for each element in self.optimiser_parameters, with None if unbounded in that direction """

        bounds = []

        for param in self.unfixed:
            if isinstance(param, SigmaParameter) and param.index[0] == param.index[1]:
                # Diagonals in the sigma matrix are standard deviations and so cannot be negative
                bounds.append((0, None))
            else:
                # Other terms are unrestricted
                bounds.append((None, None))

        return bounds

    def _store(self, parameter_cls: Type[Parameter], indices: Iterable[Tuple[int, int]], non_zero: Iterable[Tuple[int, int]]):
        non_zero = set(non_zero)
        for index in indices:
            parameter = parameter_cls(index)
            if index in non_zero:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

    def expand(self, theta_like: Vector, nullify: bool = False) -> Tuple[Vector, Vector]:
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = np.full_like(self.pi, np.nan)

        items = [
            (SigmaParameter, sigma_like),
            (PiParameter, pi_like),
        ]

        for parameter, value in zip(self.unfixed, theta_like):
            for parameter_type, values in items:
                if isinstance(parameter, parameter_type):
                    values[parameter.index] = value
                    break

        if not nullify:
            sigma_like[np.triu_indices_from(sigma_like, 1)] = 0
            for parameter in self.fixed:
                for parameter_type, values in items:
                    if isinstance(parameter, parameter_type):
                        values[parameter.index] = 0
                        break

        return sigma_like, pi_like
