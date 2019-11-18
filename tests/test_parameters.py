import numpy as np
from numpy.testing import assert_array_equal

import pytest

from miniblp.common import Theta2


INITIAL_SIGMA = np.eye(3)
INITIAL_PI = np.array([[1, 1],
                      [1, 1],
                      [1, 1]])


def test_sigma_pi():
    theta2 = Theta2(initial_sigma=INITIAL_SIGMA, initial_pi=INITIAL_PI)

    assert_array_equal(theta2.sigma, INITIAL_SIGMA)
    assert_array_equal(theta2.pi, INITIAL_PI)


def test_optimiser_parameters():
    theta2 = Theta2(INITIAL_SIGMA, INITIAL_PI)

    assert_array_equal(theta2.optimiser_parameters, np.ones(shape=9))

    with pytest.raises(ValueError):
        theta2.optimiser_parameters = np.ones(shape=2)

    theta2.optimiser_parameters = np.arange(0, 9)
    assert_array_equal(theta2.optimiser_parameters, np.arange(0, 9))

    assert_array_equal(theta2.sigma, np.array([[0, 0, 0],
                                               [0, 3, 0],
                                               [0, 0, 6]]))

    assert_array_equal(theta2.pi, [[1, 2],
                                   [4, 5],
                                   [7, 8]])
