import os

import pandas as pd
import numpy as np
import miniblp

# Data from Nevo (2000) to solve the paper’s fake cereal problem
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
product_data = pd.read_csv(os.path.join(_THIS_DIR, "nevo_products.csv"))
individual_data = pd.read_csv(os.path.join(_THIS_DIR, "nevo_individuals.csv"))

product_formulation = miniblp.ProductFormulation(
    "0 + price + C(product_id)",
    "1 + price + sugar + mushy",
    " + ".join(f"demand_instrument{i}" for i in range(20))
)

integration = miniblp.integration.MonteCarloIntegration(ns=50)

problem = miniblp.Problem(
    product_formulation, product_data,
    integration=integration,
    seed=0
)


def test_sigma_ones(capsys):
    # squarem = miniblp.iteration.SQUAREMIteration()
    # iteration = miniblp.iteration.SimpleFixedPointIteration()
    optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)
    # optimisation = miniblp.optimisation.SciPyOptimisation("Nelder-Mead")
    iteration = miniblp.iteration.NewSimpleIteration()

    with capsys.disabled():
        result = problem.solve(sigma=np.ones((4, 4)), iteration=iteration, optimisation=optimisation)
        print(result)

    assert result.beta_estimates.loc["price", "estimate"] == -3.10E1