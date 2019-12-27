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

demographic_formulation = miniblp.DemographicsFormulation("0 + income + income_squared + age + child")
nevo_integration = miniblp.integration.PrecomputedIntegration(
    individual_data,
    nodes=[f"nodes{i}" for i in range(0, 4)],
    weights="weights"
)

nevo_problem = miniblp.Problem(
    product_formulation, product_data,
    demographic_formulation, individual_data,
    integration=nevo_integration
)


def test_sigma_ones(capsys):
    iteration = miniblp.iteration.SQUAREMIteration()
    optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)

    with capsys.disabled():
        result = problem.solve(sigma=np.ones((4, 4)), iteration=iteration, optimisation=optimisation)
        print(result)

    np.testing.assert_approx_equal(result.beta_estimates.loc["price", "estimate"], -3.10E1, significant=3)


def test_nevo(capsys):
    iteration = miniblp.iteration.SQUAREMIteration()
    optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)

    initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
    initial_pi = np.array([
        [5.4819, 0, 0.2037, 0],
        [15.8935, -1.2000, 0, 2.6342],
        [-0.2506, 0, 0.0511, 0],
        [1.2650, 0, -0.8091, 0]
    ])

    theta2 = miniblp.Theta2(nevo_problem, initial_sigma, initial_pi)
    assert len(theta2.unfixed) == 4 + 9
    assert len(theta2.fixed) == 6 + 7

    with capsys.disabled():
        print(nevo_problem)
        result = nevo_problem.solve(sigma=initial_sigma, pi=initial_pi, iteration=iteration, optimisation=optimisation, method="1s")
        print(result)

    np.testing.assert_approx_equal(result.beta_estimates.loc["price", "estimate"], -6.27E+01, significant=3)