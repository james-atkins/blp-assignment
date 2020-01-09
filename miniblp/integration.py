import functools
import itertools
from typing import Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd

from .common import Vector, Matrix


class NumericalIntegration:
    def integrate(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Vector]:
        """Build concatenated IDs, nodes, and weights for each market ID."""
        raise NotImplementedError


class MonteCarloIntegration(NumericalIntegration):
    def __init__(self, ns: int = 200):
        self._ns = ns

    def integrate(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Vector]:
        """Build concatenated IDs, nodes, and weights for each market ID."""
        ids_list: List[Vector] = []
        nodes_list: List[Matrix] = []
        weights_list: List[Vector] = []

        for market_id in market_ids:
            nodes, weights = state.normal(size=(self._ns, dimensions)), np.repeat(1/self._ns, self._ns)
            ids_list.append(np.repeat(market_id, len(nodes)))
            nodes_list.append(nodes)
            weights_list.append(weights)

        return np.concatenate(ids_list), np.vstack(nodes_list), np.concatenate(weights_list)


class PrecomputedIntegration(NumericalIntegration):
    def __init__(self, data: pd.DataFrame, nodes: List[str], weights: str):
        self._market_ids = np.asanyarray(data["market_id"])
        self._weights = np.asanyarray(data[weights])
        self._nodes = np.asanyarray(data[nodes])

    def integrate(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Optional[Vector]]:
        if dimensions > self._nodes.shape[1]:
            raise ValueError("Precomputed integration data has an insufficient number of dimensions.")

        ids_list: List[Vector] = []
        nodes_list: List[Matrix] = []
        weights_list: List[Vector] = []

        for market_id in market_ids:
            mask = self._market_ids == market_id
            ids_list.append(self._market_ids[mask])
            nodes_list.append(self._nodes[mask, :dimensions])
            weights_list.append(self._weights[mask])

        return np.concatenate(ids_list), np.vstack(nodes_list), np.concatenate(weights_list)


class GaussHermiteQuadrature(NumericalIntegration):
    """
    Code based on the pyblp
    """
    def __init__(self, level: int):
        self.level = level

    def integrate(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Vector]:
        """Build concatenated IDs, nodes, and weights for each market ID."""
        ids_list: List[Vector] = []
        nodes_list: List[Matrix] = []
        weights_list: List[Vector] = []

        for market_id in market_ids:
            nodes, weights = self._product_rule(dimensions)
            ids_list.append(np.repeat(market_id, len(nodes)))
            nodes_list.append(nodes)
            weights_list.append(weights)

        return np.concatenate(ids_list), np.vstack(nodes_list), np.concatenate(weights_list)

    def _quadrature_rule(self) -> Tuple[Vector, Vector]:
        """ Compute nodes and weights for the univariate Gauss-Hermite quadrature rule."""
        raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(self.level)
        return raw_nodes * np.sqrt(2), raw_weights / np.sqrt(np.pi)

    def _product_rule(self, dimensions: int) -> Tuple[Vector, Vector]:
        """ Generate nodes and weights for integration according to the Gauss-Hermite product rule."""
        base_nodes, base_weights = self._quadrature_rule()
        nodes = np.array(list(itertools.product(base_nodes, repeat=dimensions)))
        weights = functools.reduce(np.kron, itertools.repeat(base_weights, dimensions))
        return nodes, weights
