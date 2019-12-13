from typing import Type, Iterable, Tuple, List

import numpy as np

Vector: Type[np.ndarray] = np.ndarray
Matrix: Type[np.ndarray] = np.ndarray


def is_vector(vector: np.ndarray) -> bool:
    return len(vector.shape) == 1


def is_matrix(matrix: np.ndarray) -> bool:
    return len(matrix.shape) == 2


def are_same_length(*arrays: np.ndarray) -> bool:
    """ Check that the vectors and matrix in arrays have the same length. """
    iterator = iter(arrays)
    try:
        first = len(next(iterator))
    except StopIteration:
        return True
    return all(first == len(rest) for rest in iterator)


Theta1: Type[np.ndarray] = np.ndarray


class Parameter:
    def __init__(self, index: Tuple[int, int]):
        self._index = index

    def product_characteristic(self, market: "Market") -> Vector:
        return market.products.X2[:, [self._index[0]]]

    def agent_characteristic(self, market: "Market") -> Vector:
        """Get the agent characteristic associated with the parameter."""
        raise NotImplementedError


class SigmaParameter(Parameter):
    """ Information about a single parameter in sigma. """
    def agent_characteristic(self, market: "Market") -> Vector:
        return market.individuals.nodes[:, [self._index[1]]]


class PiParameter(Parameter):
    """ Information about a single parameter in pi. """
    def agent_characteristic(self, market: "Market") -> Vector:
        return market.individuals.demographics[:, [self._index[1]]]


class Theta2:
    """ Coefficients for the non-linear parameters. """

    def __init__(self, problem: "Problem", initial_sigma: Matrix, initial_pi: Matrix):
        k2 = problem.products.K2
        if not (k2 == initial_sigma.shape[0] == initial_sigma.shape[1]):
            raise ValueError("sigma is not of the correct dimensions. Should be {} x {}".format(k2, k2))

        if k2 != initial_pi.shape[0] or initial_pi.shape[1] != problem.individuals.D:
            raise ValueError("pi is not of the correct dimension. Should be {} x {}.".format(k2, problem.individuals.D))

        self._k2 = k2

        # Sigma should be lower triangle
        sigma = np.tril(initial_sigma)

        # Sigma is a K2 x K2 matrix and Pi is a K2 x D matrix so
        # both are compressed into a K2 x (K2 + D) matrix
        self._data = np.hstack((sigma, initial_pi))

        self.fixed: List[Parameter] = []
        self.unfixed: List[Parameter] = []
        self._store(SigmaParameter, zip(*np.tril_indices_from(self.sigma)), zip(*np.nonzero(self.sigma)))
        self._store(PiParameter, np.ndindex(self.pi.shape), zip(*np.nonzero(self.pi)))
        self.P = len(self.unfixed)
        self._unfixed_params_indices = np.flatnonzero(self._data)

    @property
    def sigma(self) -> Matrix:
        return self._data[:, :self._k2]

    @property
    def pi(self):
        return self._data[:, self._k2:]

    @property
    def optimiser_parameters(self):
        """ Parameters to be optimised over. """
        return self._data.ravel()[self._unfixed_params_indices]

    @optimiser_parameters.setter
    def optimiser_parameters(self, values):
        if values.shape != self._unfixed_params_indices.shape:
            raise ValueError("optimiser_parameters have the wrong shape.")

        self._data.put(self._unfixed_params_indices, values)

    def _store(self, parameter_cls: Type[Parameter], indices: Iterable[Tuple[int, int]], non_zero: Iterable[Tuple[int, int]]):
        non_zero = list(non_zero)
        for index in indices:
            parameter = parameter_cls(index)
            if index in non_zero:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)
