import itertools
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, NamedTuple, NewType

import numpy as np
import pandas as pd
import patsy
from numpy.linalg import matrix_rank

from .common import Vector, Matrix, is_vector, is_matrix, are_same_length
from .integration import NumericalIntegration


class ProductFormulation(NamedTuple):
    linear: str
    random: str
    instruments: str


DemographicsFormulation = NewType("DemographicsFormulation", str)


@dataclass
class Products:
    market_ids: Vector
    market_shares: Vector
    prices: Vector
    X1: Matrix
    X2: Matrix
    Z: Matrix
    # TODO: Use a structured array for data locality reasons?

    def __post_init__(self):
        assert is_vector(self.market_ids)
        assert is_vector(self.market_shares)
        assert is_vector(self.prices)
        assert is_matrix(self.X1)
        assert is_matrix(self.X2)
        assert is_matrix(self.Z)
        assert are_same_length(self.market_ids, self.market_shares, self.prices, self.X1, self.X2, self.Z)

        self.J, self.K1 = self.X1.shape
        _, self.K2 = self.X2.shape
        self.MD = self.Z.shape[1]

    @classmethod
    def from_formula(cls, product_formulation: ProductFormulation, data: pd.DataFrame):
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
        X1 = patsy.dmatrix(product_formulation.linear, data, NA_action="raise", eval_env=2)
        X2 = patsy.dmatrix(product_formulation.random, data, NA_action="raise", eval_env=2)

        # Sanity checks to stop doing something stupid
        if "market_share" in X1.design_info.column_names:
            raise ValueError("market_share cannot be included in X1.")
        if "market_share" in X2.design_info.column_names:
            raise ValueError("market_share cannot be included in X2.")
        if "price" not in X1.design_info.column_names or "price" not in X2.design_info.column_names:
            raise ValueError("price must be included in X1 or X2.")

        # Build instruments from exogenous variables (excluding price) and instruments
        w = patsy.dmatrix(product_formulation.instruments, data, NA_action="raise", eval_env=2)
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
    weights: Vector  # I
    nodes: Matrix  # I x D
    demographics: Optional[Matrix]  # I x D

    def __post_init__(self):
        assert is_vector(self.market_ids)
        assert self.weights is None or is_vector(self.weights)
        assert is_matrix(self.nodes)
        assert self.demographics is None or is_matrix(self.demographics)
        assert are_same_length(self.weights, self.nodes, self.demographics)

        self.I = len(self.weights)

        if self.demographics is not None:
            _, self.D = self.demographics.shape
        else:
            self.D = 0

    @classmethod
    def from_formula(cls, demographics_formulation: DemographicsFormulation, demographics_data: pd.DataFrame,
                     products: Products, integration: NumericalIntegration,
                     seed: Optional[int] = None) -> "Individuals":

        if (demographics_formulation is None and demographics_data is not None) or (demographics_formulation is not None and demographics_data is None):
            raise ValueError("Both demographics_formulation and demographics_data should be None or not None")

        if demographics_formulation is not None:
            try:
                demographics_data = demographics_data.sort_values(by="market_id").reset_index()
            except KeyError:
                raise KeyError("Demographics data must have a market_id field.")

            market_ids = np.asanyarray(demographics_data["market_id"])

            if set(np.unique(products.market_ids)) > set(np.unique(market_ids)):
                raise ValueError("There are product markets with no corresponding demographic data.")

        else:
            market_ids = np.asanyarray(products.market_ids)

        state = np.random.RandomState(seed=seed)

        # Generate random taste shocks
        market_ids, nodes, weights = integration.build_many(products.K2, np.unique(market_ids), state)

        if demographics_formulation is not None:
            demographics_list = []
            for market_id, num in zip(*np.unique(market_ids, return_counts=True)):
                demographics = demographics_data[demographics_data["market_id"] == market_id]
                if num == len(demographics):
                    demographics_list.append(demographics)
                else:
                    # Randomly sample demographic data
                    demographics_list.append(demographics.sample(num, replace=True, random_state=state))

            drawn_demographics = pd.concat(demographics_list)

            # Build from formula
            demographics = patsy.dmatrix(demographics_formulation, drawn_demographics)
        else:
            demographics = None

        return cls(market_ids=market_ids, weights=weights, nodes=nodes, demographics=demographics)

    def split_markets(self) -> Iterator["Individuals"]:
        if self.demographics is None:
            for market_ids, weights, nodes in _split_markets(self.market_ids, self.weights, self.nodes):
                yield Individuals(market_ids, weights, nodes, None)
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
