import itertools
from dataclasses import dataclass
from typing import Optional, Iterator, List, Tuple

import numpy as np
import pandas as pd
import patsy
from numpy.linalg import matrix_rank
from scipy import linalg

from .common import Vector, Matrix, is_vector, is_matrix, are_same_length
from .integration import NumericalIntegration


@dataclass
class Products:
    market_ids: Vector
    market_shares: Vector
    prices: Vector
    X1: Matrix
    X2: Matrix
    Z: Matrix

    def __post_init__(self):
        assert is_vector(self.market_ids)
        assert is_vector(self.market_shares)
        assert is_vector(self.prices)
        assert is_matrix(self.X1)
        assert is_matrix(self.X2)
        assert is_matrix(self.Z)
        assert are_same_length(self.market_ids, self.market_shares, self.prices, self.X1, self.X2, self.Z)

        _, self.K1 = self.X1.shape
        _, self.K2 = self.X2.shape

    @classmethod
    def from_formula(cls, linear_coefficients: str, random_coefficients: str, instruments: str, data: pd.DataFrame):
        try:
            data = data.sort_values(by="market_id").reset_index()
        except KeyError:
            raise KeyError("Product data must have a market_id field.")

        market_ids = np.asanyarray(data["market_id"])

        try:
            prices = np.asanyarray(data["price"])
        except KeyError:
            raise KeyError("Product data must have a price field.")

        try:
            market_shares = np.asanyarray(data["market_share"])
        except KeyError:
            raise KeyError("Product data must have a market_share field.")

        # Build matrices
        X1 = patsy.dmatrix(linear_coefficients, data, NA_action="raise", eval_env=2)
        X2 = patsy.dmatrix(random_coefficients, data, NA_action="raise", eval_env=2)

        # Sanity checks to stop doing something stupid
        if "market_share" in X1.design_info.column_names:
            raise ValueError("market_share cannot be included in X1.")
        if "market_share" in X2.design_info.column_names:
            raise ValueError("market_share cannot be included in X2.")
        if "price" not in X1.design_info.column_names or "price" not in X2.design_info.column_names:
            raise ValueError("price must be included in X1 or X2.")

        # Build instruments from exogenous variables (excluding price) and instruments
        w = patsy.dmatrix(instruments, data, NA_action="raise", eval_env=2)
        zd_terms = [term for term in itertools.chain(X1.design_info.terms, w.design_info.terms) if term.name() != "price"]
        Z = patsy.dmatrix(patsy.ModelDesc([], zd_terms), data, NA_action="raise", eval_env=2)

        # Rank condition for instruments
        z_x1 = Z.T @ X1
        if matrix_rank(z_x1) < min(*z_x1.shape):
            raise ValueError("Matrix Z' X₁ does not have full rank.")

        return cls(market_ids=market_ids, market_shares=market_shares, prices=prices, X1=X1, X2=X2, Z=Z)

    def split_markets(self) -> Iterator["Products"]:
        for arrays in _split_markets(self.market_ids, self.market_shares, self.prices, self.X1, self.X2, self.Z):
            yield Products(*arrays)


@dataclass
class Individuals:
    market_ids: Vector  # IDs that associate individuals with markets.
    weights: Optional[Vector]  # I
    nodes: Matrix  # I x D
    demographics: Matrix  # I x D

    def __post_init__(self):
        assert is_vector(self.market_ids)
        assert self.weights is None or is_vector(self.weights)
        assert is_matrix(self.nodes)
        assert is_matrix(self.demographics)
        if self.weights is None:
            assert are_same_length(self.nodes, self.demographics)
        else:
            assert are_same_length(self.weights, self.nodes, self.demographics)

        self.I, self.D = self.demographics.shape

    @classmethod
    def from_formula(cls, demographics_formula: str, demographics_data: pd.DataFrame,
                     products: Products, integration: NumericalIntegration,
                     seed: Optional[int] = None) -> "Individuals":

        try:
            demographics_data = demographics_data.sort_values(by="market_id").reset_index()
        except KeyError:
            raise KeyError("Demographics data must have a market_id field.")

        market_ids = np.asanyarray(demographics_data["market_id"])

        if set(np.unique(products.market_ids)) > set(np.unique(market_ids)):
            raise ValueError("There are product markets with no corresponding demographic data.")

        # Generate random taste shocks
        state = np.random.RandomState(seed=seed)
        market_ids, nodes, weights = integration.build_many(products.K2, np.unique(market_ids), state)

        # Randomly sample demographic data
        demographics_list = []
        for market_id in np.unique(market_ids):
            demographics = demographics_data[demographics_data["market_id"] == market_id]
            num_sampled = np.sum(market_ids == market_id)
            demographics_list.append(demographics.sample(num_sampled, replace=True))

        drawn_demographics = pd.concat(demographics_list)

        # Build from formula
        demographics = patsy.dmatrix(demographics_formula, drawn_demographics)

        # TODO: remove the intercept in a nicer way
        demographics = demographics[:, 1:]

        return cls(market_ids=market_ids, weights=weights, nodes=nodes, demographics=demographics)

    def split_markets(self) -> Iterator["Individuals"]:
        if self.weights is None:
            for market_ids, nodes, demographics in _split_markets(self.market_ids, self.nodes, self.demographics):
                yield Individuals(market_ids, None, nodes, demographics)
        else:
            for arrays in _split_markets(self.market_ids, self.weights, self.nodes, self.demographics):
                yield Individuals(*arrays)


def _split_markets(market_ids, *arrays_or_matrices) -> Iterator[Tuple[np.ndarray]]:
    # TODO: Check that market_ids array is sorted
    _, indices = np.unique(market_ids, return_index=True)

    arrays_or_matrices = itertools.chain([market_ids], arrays_or_matrices)
    it = zip(*map(lambda array: np.split(array, indices_or_sections=indices), arrays_or_matrices))

    # np.split leaves an empty array as the first item
    next(it)
    return it


def _remove_intercept(terms_list):
    pass