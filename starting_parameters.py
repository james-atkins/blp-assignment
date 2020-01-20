""" Analysis of different starting parameters. """
import itertools

import numpy as np
import pandas as pd
import miniblp

RANDOM_SEED = 0
FIXED_POINT_TOLERANCE = 1e-14


def make_results_dataframe(results):
    return pd.DataFrame({"initial_sigma": [initial_sigma for (initial_sigma, _, _) in results],
                         "initial_pi": [initial_pi for (_, initial_pi, _) in results],
                       "converged": [result.optimisation_result.success for (_, _, result) in results],
                       "sigma": [result.optimisation_result.solution[()] for (_, _, result) in results],
                       "objective": [result.optimisation_result.objective for (_, _, result) in results],
                       "price": [result.beta_estimates.loc["price", "estimate"] for (_, _, result) in results]}). \
        sort_values(["objective"]).reset_index(drop=True)


if __name__ == "__main__":
    apps = pd.read_csv("data/apps.csv")
    demographics = pd.read_csv("data/demographics.csv")

    integration = miniblp.integration.GaussHermiteQuadrature(level=7)
    iteration = miniblp.iteration.SQUAREMIteration(tolerance=FIXED_POINT_TOLERANCE)
    optimisation = miniblp.optimisation.Powell(ftol=1e-50)

    product_formulation_1 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="num_apps_group"
    )

    demographics_formulation_1 = miniblp.DemographicsFormulation("0 + demographic1_z")

    problem_0 = miniblp.Problem(
        product_formulation, apps,
        integration=integration,
        seed=RANDOM_SEED
    )

    results = []
    for initial_sigma in np.arange(-5, 2, 0.5):
        print("Trying initial parameters:", initial_sigma)
        result = problem_0.solve(np.array([[initial_sigma]]), optimisation=optimisation, iteration=iteration,
                                 method="1s", parallel=True)
        print(result)
        results.append((initial_sigma, None, result))

    df = make_results_dataframe(results)
    df.to_csv("searches/0.csv", index=False)

    problem_1 = miniblp.Problem(
        product_formulation_1, apps,
        demographics_formulation_1, demographics,
        integration=integration,
        seed=RANDOM_SEED
    )

    results = []
    for initial_sigma, initial_pi in itertools.product(np.arange(-1, 1, 0.5), np.arange(-2, 2, 0.5)):
        print("Trying initial parameters:", initial_sigma, initial_pi)
        result = problem_1.solve(np.array([[initial_sigma]]), np.array([[initial_pi]]), optimisation=optimisation, iteration=iteration,
                                 method="1s", parallel=True)
        print(result)
        results.append((initial_sigma, initial_pi, result))

    df = make_results_dataframe(results)
    df.to_csv("searches/1.csv", index=False)

    product_formulation_4 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price + average_score",
        instruments="average_price_other_countries"
    )

    apps_hausman = apps.dropna()

    problem_4 = miniblp.Problem(
        product_formulation_4, apps_hausman,
        demographics_formulation_1, demographics,
        integration=integration, seed=RANDOM_SEED
    )

    initial_params = np.arange(-1, 1.5, 0.5)

    results = []
    for initial_sigma_1, initial_sigma_2, initial_pi_1, initial_pi_2 in itertools.product(initial_params, initial_params, initial_params, initial_params):
        initial_sigma = np.diag([initial_sigma_1, initial_sigma_2])
        initial_pi = np.array([[initial_pi_1], [initial_pi_2]])
        print("Trying initial parameters:", initial_sigma, initial_pi)
        result = problem_4.solve(initial_sigma, initial_pi, optimisation=optimisation,
                             iteration=iteration,
                             method="1s", parallel=True)
        print(result)
        results.append((initial_sigma, initial_pi, result))

    df = make_results_dataframe(results)
    df.to_csv("searches/4.csv", index=False)