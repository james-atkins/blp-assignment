import os

import pandas as pd
import numpy as np
try:
    import miniblp
except ImportError:
    # Bodge in case we are running this from the root directory
    import sys
    sys.path.append(".")
    import miniblp

if __name__ == "__main__":
    # Data from Nevo (2000) to solve the paper’s fake cereal problem
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    product_data = pd.read_csv(os.path.join(_THIS_DIR, "nevo_products.csv"))
    individual_data = pd.read_csv(os.path.join(_THIS_DIR, "nevo_individuals.csv"))

    iteration = miniblp.iteration.SQUAREMIteration()
    optimisation = miniblp.optimisation.BFGS(gtol=1e-10)

    product_formulation = miniblp.ProductFormulation(
        "0 + price + C(product_id)",
        "1 + price + sugar + mushy",
        " + ".join(f"demand_instrument{i}" for i in range(20))
    )

    demographic_formulation = miniblp.DemographicsFormulation("0 + income + income_squared + age + child")

    integration = miniblp.integration.PrecomputedIntegration(
        individual_data,
        nodes=[f"nodes{i}" for i in range(0, 4)],
        weights="weights"
    )

    problem = miniblp.Problem(
        product_formulation, product_data,
        demographic_formulation, individual_data,
        integration=integration
    )

    print(problem)

    initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
    initial_pi = np.array([
        [5.4819, 0, 0.2037, 0],
        [15.8935, -1.2000, 0, 2.6342],
        [-0.2506, 0, 0.0511, 0],
        [1.2650, 0, -0.8091, 0]
    ])

    result = problem.solve(sigma=initial_sigma, pi=initial_pi, iteration=iteration, optimisation=optimisation, method="1s")
    print(result)

    # Results are similar to those in the original paper with a price coefficient of 𝛼̂ =−62.7
    np.testing.assert_approx_equal(result.beta_estimates.loc["price", "estimate"], -6.27E+01, significant=3)