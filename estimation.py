import numpy as np
import pandas as pd
import miniblp

RANDOM_SEED = 0
FIXED_POINT_TOLERANCE = 1e-14
PARALLEL = False

integration = miniblp.integration.GaussHermiteQuadrature(level=7)
iteration = miniblp.iteration.SQUAREMIteration(tolerance=FIXED_POINT_TOLERANCE)

if __name__ == "__main__":
    apps = pd.read_csv("data/apps.csv")
    demographics = pd.read_csv("data/demographics.csv")

    optimisation = miniblp.optimisation.NelderMead(fatol=1e-24)
    optimisation = miniblp.optimisation.Powell(ftol=1e-24)

    # Part 1 - baseline model

    product_formulation_1 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="num_apps_group"
    )

    demographics_formulation_1 = miniblp.DemographicsFormulation("0 + demographic1_z")

    problem_1 = miniblp.Problem(
        product_formulation_1, apps,
        demographics_formulation_1, demographics,
        integration=integration,
        seed=RANDOM_SEED
    )

    print(problem_1)

    initial_sigma = np.array([[-0.5]])
    initial_pi = np.array([[-0.5]])

    result_1 = problem_1.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration,
                               parallel=PARALLEL)
    print(result_1)
    result_1.to_latex("output/estimation1.tex", caption="Baseline model", label="tab:estimation1")

    # Part 2 - dummy variables for the genre

    product_formulation_2 = miniblp.ProductFormulation(
        linear="0 + C(nest) + price + average_score + in_app_purchases",
        random="0 + price",
        instruments="num_apps_group"
    )

    problem_2 = miniblp.Problem(
        product_formulation_2, apps,
        demographics_formulation_1, demographics,
        integration=integration,
        seed=RANDOM_SEED
    )

    result_2 = problem_2.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration,
                               parallel=PARALLEL)
    print(result_2)
    result_2.to_latex("output/estimation2.tex", caption="Model with genre dummy variables", label="tab:estimation2")

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
        demographics_formulation_1, demographics,
        integration=integration, seed=RANDOM_SEED
    )

    result_3 = problem_3.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration, parallel=PARALLEL)
    result_3.to_latex("output/estimation3.tex", caption="Baseline model with Hausman instrument", label="tab:estimation3")
    print(result_3)

    # Part 4 - average score having random coefficients

    product_formulation_4 = miniblp.ProductFormulation(
        linear="1 + price + average_score + in_app_purchases",
        random="0 + price + average_score",
        instruments="average_price_other_countries"
    )

    problem_4 = miniblp.Problem(
        product_formulation_4, apps_hausman,
        demographics_formulation_1, demographics,
        integration=integration, seed=RANDOM_SEED
    )

    print(problem_4)

    initial_sigma = np.diag([-1.0, -1.0])
    initial_pi = np.array([[1.0], [-1.0]])

    result_4 = problem_4.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration, parallel=PARALLEL)
    result_4.to_latex("output/estimation4.tex", caption="Model with average score having random coefficients", label="tab:estimation4")
    print(result_4)