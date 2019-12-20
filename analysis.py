import pandas as pd
import numpy as np

import miniblp

apps = pd.read_csv("data/apps.csv")
demographics = pd.read_csv("data/demographics.csv")

integration = miniblp.integration.MonteCarloIntegration(50)
iteration = miniblp.iteration.SQUAREMIteration()
# iteration = miniblp.iteration.SimpleFixedPointIteration()

product_formulation = miniblp.ProductFormulation(
    "1 + price + average_score + in_app_purchases",
    "1 + price + average_score + in_app_purchases",
    "num_apps_category"
)

demographics_formulation = miniblp.DemographicsFormulation("0 + demographic1")

problem = miniblp.Problem(
    product_formulation, apps,
    # demographics_formulation, demographics,
    integration=integration, iteration=iteration,
    seed=12345
)

print(problem)

optimisation = miniblp.optimisation.SciPyOptimisation("Nelder-Mead")
# optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)

initial_sigma = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
initial_pi = np.array([[0.5], [0.5]])

initial_sigma = np.diag([0.5, 0.5, 0.5, 0.5])

result = problem.solve(sigma=np.ones((4,4)), optimisation=optimisation)
print(result)