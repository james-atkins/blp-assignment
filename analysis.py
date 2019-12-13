import pandas as pd
import numpy as np

import miniblp

apps = pd.read_csv("data/apps.csv")
demographics = pd.read_csv("data/demographics.csv")

integration = miniblp.integration.MonteCarloIntegration(200)
iteration = miniblp.iteration.PhasedToleranceIteration()
optimisation = miniblp.optimisation.SciPyOptimisation("Nelder-Mead")
# optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)

problem = miniblp.Problem("1 + price + average_score + in_app_purchases",
                          "0 + price",
                          "num_apps_category",
                          "0 + demographic1", apps, demographics,
                          integration, iteration)


result = problem.solve(np.array([[-0.5]]), np.array([[0.5]]), optimisation)
print(result)