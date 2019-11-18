from typing import Optional, List, Tuple, Iterable

import numpy as np

from .common import Vector


class NumericalIntegration:
    def __init__(self, size: int):
        self._size = size

    def build(self, dimensions: int, state: np.random.RandomState) -> Tuple[Vector, Optional[Vector]]:
        raise NotImplementedError

    def build_many(self, dimensions: int, market_ids: Iterable[str], state: np.random.RandomState) -> Tuple[Vector, Vector, Optional[Vector]]:
        """Build concatenated IDs, nodes, and weights for each market ID."""
        ids_list: List[Vector] = []
        nodes_list: List[Vector] = []
        weights_list: List[Optional[Vector]] = []

        for market_id in market_ids:
            nodes, weights = self.build(dimensions, state)
            ids_list.append(np.repeat(market_id, len(nodes)))
            nodes_list.append(nodes)
            weights_list.append(weights)

        if any(w is None for w in weights_list):
            return np.concatenate(ids_list), np.vstack(nodes_list), None
        else:
            return np.concatenate(ids_list), np.vstack(nodes_list), np.vstack(weights_list)


class MonteCarloIntegration(NumericalIntegration):
    def build(self, dimensions: int, state: np.random.RandomState) -> Tuple[Vector, Optional[Vector]]:
        return state.normal(size=(self._size, dimensions)), None