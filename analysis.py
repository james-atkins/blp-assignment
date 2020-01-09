import itertools
import pickle

import numpy as np
import pandas as pd
import miniblp

RANDOM_SEED = 0

def zscore(x):
    """
     This scaling has an advantage of making coefficients interpretable (the response due to a 1-σ shift in x),
     and also to get rid of the scale issues discussed above.
    """
    return (x - x.mean()) / x.std()


if __name__ == "__main__":
    apps = pd.read_csv("data/apps.csv")
    demographics = pd.read_csv("data/demographics.csv")

    demographics_scaled = demographics.copy()
    demographics_scaled["demographic1"] = zscore(demographics_scaled["demographic1"])

    # integration = miniblp.integration.MonteCarloIntegration(100)
    integration = miniblp.integration.GaussHermiteQuadrature(level=9)
    iteration = miniblp.iteration.SQUAREMIteration(tolerance=1e-14)
    # iteration = miniblp.iteration.SimpleFixedPointIteration(tolerance=1e-14)
    optimisation = miniblp.optimisation.NelderMead(fatol=1e-24)
    # optimisation = miniblp.optimisation.LBFGSB(ftol=1e-24)
    # optimisation = miniblp.optimisation.BFGS(gtol=1e-20)

    # Part 0

    # Part 1

    demographics_formulation_1 = miniblp.DemographicsFormulation("0 + demographic1")

    product_formulation_1 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="num_apps_group"
    )

    problem_1 = miniblp.Problem(
        product_formulation_1, apps,
        demographics_formulation_1, demographics_scaled,
        integration=integration,
        seed=RANDOM_SEED
    )

    print(problem_1)

    initial_sigma = np.array([[0.5]])
    initial_pi = np.array([[0.5]])

    result_1 = problem_1.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration,
                               parallel=False)
    print(result_1)

    exit()

    # with open("part_1.pickle", "wb") as f:
    #     for initial_sigma, initial_pi in itertools.product(np.arange(0, 5, 0.5), np.arange(-5, 5, 0.5)):
    #         print("Trying initial parameters:", initial_sigma, initial_pi)
    #         result = problem_1.solve(np.array([[initial_sigma]]), np.array([[initial_pi]]), optimisation=optimisation, iteration=iteration,
    #                                  method="1s", parallel=True)
    #         print(result)
    #         results.append((initial_sigma, initial_pi, result))
    #         pickle.dump(results, f)
    # exit()

    # Part 2 - dummy variables for the genre

    product_formulation_2 = miniblp.ProductFormulation(
        linear="0 + C(nest) + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="num_apps_group"
    )

    problem_2 = miniblp.Problem(
        product_formulation_2, apps,
        demographics_formulation_1, demographics_scaled,
        integration=integration,
        seed=RANDOM_SEED
    )

    # result_2 = problem_2.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration,
    #                            parallel=False)
    # print(result_2)

    # Part 3 - Hausman instrument

    # For some apps the Hausman instrument cannot be calculated as the app is only in one country
    apps_hausman = apps.dropna()

    product_formulation_3 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="average_price_other_countries"
    )

    problem_3 = miniblp.Problem(
        product_formulation_3, apps_hausman,
        demographics_formulation_1, demographics_scaled,
        integration=integration, seed=RANDOM_SEED
    )

    result_3 = problem_3.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration, parallel=False)
    print(result_3)