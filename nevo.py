import pandas as pd
import numpy as np
import miniblp

product_data = pd.read_csv("nevo/nevo_products.csv")
agent_data = pd.read_csv("nevo/nevo_agents.csv")

# integration = miniblp.integration.PrecomputedIntegration(agent_data, ["nodes0", "nodes1", "nodes2", "nodes3"], "weights")
integration = miniblp.integration.MonteCarloIntegration(ns=200)
iteration = miniblp.iteration.PhasedToleranceIteration()
# optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)
optimisation = miniblp.optimisation.SciPyOptimisation("Nelder-Mead")

problem = miniblp.Problem("price + C(product_id)",
                          "1 + price + sugar + mushy",
                          " + ".join(f"demand_instrument{i}" for i in range(20)),
                          "0 + income + income_squared + age + child",
                          product_data,
                          agent_data,
                          integration,
                          iteration)

initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
initial_pi = np.array([
    [5.4819, 0, 0.2037, 0],
    [15.8935, -1.2000, 0, 2.6342],
    [-0.2506, 0, 0.0511, 0],
    [1.2650, 0, -0.8091, 0]
])

result = problem.solve(initial_sigma, initial_pi, optimisation)
print(result)
