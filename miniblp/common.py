from typing import Type

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


class Theta2:
    """ Coefficients for the non-linear parameters. """

    sigma: Matrix
    pi: Matrix

    def __init__(self, initial_sigma: Matrix, initial_pi: Matrix):

        k2_1, k2_2 = initial_sigma.shape
        k2_3, d = initial_pi.shape

        if not (k2_1 == k2_2 == k2_3):
            raise ValueError("sigma and pi are not of conforming dimensions.")

        # Sigma should be lower triangle
        sigma = np.tril(initial_sigma)

        # Sigma is a K2 x K2 matrix and Pi is a K2 x D matrix so
        # both are compressed into a K2 x (K2 + D) matrix
        self._data = np.hstack((sigma, initial_pi))

        # Views for pi and sigma
        self.sigma = self._data[:, :k2_1]
        self.pi = self._data[:, k2_1:]

        # Optimise over non-zero parameters
        self._unfixed_params = np.flatnonzero(self._data)

    @property
    def optimiser_parameters(self):
        """ Parameters to be optimised over. """
        return self._data.ravel()[self._unfixed_params]

    @optimiser_parameters.setter
    def optimiser_parameters(self, values):
        if values.shape != self._unfixed_params.shape:
            raise ValueError("optimiser_parameters have the wrong shape.")

        self._data.put(self._unfixed_params, values)
