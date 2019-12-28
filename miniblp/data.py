import itertools
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, NamedTuple, NewType, List

import numpy as np
import pandas as pd
import patsy
from numpy.linalg import matrix_rank

from .common import Vector, Matrix, is_vector, is_matrix, are_same_length, NamedMatrix
from .integration import NumericalIntegration


class ProductFormulation:
    def __init__(self, linear: str, random: str, instruments: str):
        self._linear_terms = _parse_terms(linear)
        self._random_terms = _parse_terms(random)

        w_terms = _parse_terms(instruments)
        instrument_terms = [t for t in itertools.chain(self._linear_terms, w_terms) if t.name() != "price"]
        self._instrument_terms = list(dict.fromkeys(instrument_terms))  # remove duplicates

        # Sanity checks to stop doing something stupid
        linear_term_names = {term.name() for term in self._linear_terms}
        random_term_names = {term.name() for term in self._random_terms}

        if "market_share" in linear_term_names:
            raise ValueError("market_share cannot be included in X1.")
        if "market_share" in random_term_names:
            raise ValueError("market_share cannot be included in X2.")
        if "price" not in linear_term_names or "price" not in random_term_names:
            raise ValueError("price must be included in X1 or X2.")

    def build(self, product_data: pd.DataFrame) -> Tuple[NamedMatrix, NamedMatrix, NamedMatrix]:
        X1 = _build_matrix(self._linear_terms, product_data)
        X2 = _build_matrix(self._random_terms, product_data)
        Z = _build_matrix(self._instrument_terms, product_data)

        return X1, X2, Z


def _build_matrix(terms: List[patsy.desc.Term], data: pd.DataFrame) -> NamedMatrix:
    design_info = patsy.build.design_matrix_builders([terms], lambda: iter([data]), patsy.eval.EvalEnvironment([data]))[0]
    matrix = np.asarray(patsy.build.build_design_matrices([design_info], data, NA_action="raise")[0])
    return NamedMatrix(matrix, design_info.column_names)


def _parse_terms(formula: str) -> List[patsy.desc.Term]:
    description = patsy.highlevel.ModelDesc.from_formula(formula)
    if description.lhs_termlist:
        raise patsy.PatsyError("Formulae should not have left-hand sides.")
    return description.rhs_termlist


DemographicsFormulation = NewType("DemographicsFormulation", str)


@dataclass
class Products:
    market_ids: Vector
    market_shares: Vector
    prices: Vector
    X1: NamedMatrix
    X2: NamedMatrix
    Z: NamedMatrix
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
        X1, X2, Z = product_formulation.build(data)

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
    demographics: Optional[NamedMatrix]  # I x D

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

        state = np.random.RandomState(seed=seed)

        # Generate random taste shocks
        market_ids, nodes, weights = integration.build(products.K2, np.unique(products.market_ids), state)

        if demographics_formulation is not None:
            demographics_terms = _parse_terms(demographics_formulation)
            try:
                demographics_market_ids = np.asanyarray(demographics_data["market_id"])
            except KeyError:
                raise KeyError("Demographics data must have a market_id field.")

            if set(np.unique(products.market_ids)) > set(np.unique(demographics_market_ids)):
                raise ValueError("There are product markets with no corresponding demographic data.")

            demographics_list = []
            for market_id, num_individuals in zip(*np.unique(market_ids, return_counts=True)):
                demographics = demographics_data[demographics_data["market_id"] == market_id]
                if num_individuals == len(demographics):
                    demographics_list.append(demographics)
                else:
                    # Randomly sample demographic data
                    demographics_list.append(demographics.sample(num_individuals, replace=True, random_state=state))

            drawn_demographics = pd.concat(demographics_list)

            # Build from formula
            demographics = _build_matrix(demographics_terms, drawn_demographics)
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
