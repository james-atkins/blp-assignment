from typing import Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd

from .common import Vector, Matrix


class NumericalIntegration:
    def build(self, dimensions: int, state: np.random.RandomState) -> Tuple[Matrix, Optional[Vector]]:
        raise NotImplementedError

    def build_many(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Vector]:
        """Build concatenated IDs, nodes, and weights for each market ID."""
        ids_list: List[Vector] = []
        nodes_list: List[Matrix] = []
        weights_list: List[Vector] = []

        for market_id in market_ids:
            nodes, weights = self.build(dimensions, state)
            ids_list.append(np.repeat(market_id, len(nodes)))
            nodes_list.append(nodes)
            weights_list.append(weights)

        return np.concatenate(ids_list), np.vstack(nodes_list), np.concatenate(weights_list)


class MonteCarloIntegration(NumericalIntegration):
    def __init__(self, ns: int):
        self._ns = ns

    def build(self, dimensions: int, state: np.random.RandomState) -> Tuple[Matrix, Optional[Vector]]:
        return state.normal(size=(self._ns, dimensions)), np.repeat(1/self._ns, self._ns)


class PrecomputedIntegration(NumericalIntegration):
    def __init__(self, data: pd.DataFrame, nodes: List[str], weights: str):
        self._market_ids = np.asanyarray(data["market_id"])
        self._weights = np.asanyarray(data[weights])
        self._nodes = np.asanyarray(data[nodes])

    def build_many(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Matrix, Optional[Vector]]:
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
