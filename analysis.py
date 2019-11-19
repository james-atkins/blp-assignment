import pandas as pd
import numpy as np

import miniblp

apps = pd.read_csv("data/apps.csv")
demographics = pd.read_csv("data/demographics.csv")

integration = miniblp.integration.MonteCarloIntegration(200)
iteration = miniblp.iteration.PhasedToleranceIteration()

problem = miniblp.Problem("1 + price + average_score + in_app_purchases",
                          "0 + price",
                          "num_apps_category",
                          "demographic1", apps, demographics,
                          integration, iteration)

theta2 = miniblp.Theta2(np.eye(1), np.array([[1]]))


print(problem.solve(theta2))