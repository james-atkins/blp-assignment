from typing import Tuple

import numpy as np
import pandas as pd
import miniblp
import pyblp


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    apps = pd.read_csv("data/dataset_computational_noise.csv")

    # Markets are identified by (country, year) pairs
    apps["market_id"] = apps["country"] + " " + apps["year"].apply(str)

    # Fix the download numbers are these are sometimes computed incorrectly
    apps["downloads"] = 20 * apps["nofreviews"]
    apps.loc[apps["downloads"] < apps["lowerbound"], "downloads"] = apps["lowerbound"]
    apps.loc[apps["upperbound"] < apps["downloads"], "downloads"] = apps["upperbound"]

    apps = apps[["market_id", "app_id", "app", "price", "downloads", "averagescore", "inapppurchases", "newgroup", "nest"]].\
        rename(columns={"averagescore": "average_score", "inapppurchases": "in_app_purchases", "newgroup": "group"}).\
        dropna()

    # Compute the market share
    group = apps.groupby("market_id")
    apps["market_share"] = apps["downloads"] / group["downloads"].transform("sum")

    # Remove the outside good
    apps = apps[apps["group"] != "OUTSIDE"].reset_index(drop=True)

    # Compute the number of apps per category
    num_apps_category = apps.groupby(["market_id", "group"], as_index=False).\
        count()[["market_id", "group", "app_id"]].\
        rename(columns={"app_id": "num_apps_category"})

    apps = pd.merge(apps, num_apps_category, on=["market_id", "group"])

    # The demographic data is unfortunately in one of the silliest formats ever to grace the good
    # people of this earth. Convert to a dataframe.
    # "It contains data on 2 different demographics, with 500 observation each, for a given year
    # and country. These data are organized in a 36x1000 matrix, having markets (defined as a
    # country-year pair) along the rows, and the observations for the two demographics along the
    # columns."
    raw_demographics = np.loadtxt("data/demogr_apps.csv", delimiter=",")
    demographic1, demographic2 = np.split(raw_demographics, 2, axis=1)
    demographics = pd.DataFrame({"market_id": np.repeat(apps["market_id"].unique(), 500),
                                 "demographic1": demographic1.reshape(-1),
                                 "demographic2": demographic2.reshape(-1)},
                                copy=True)

    return apps, demographics

if __name__ == "__main__":
    apps, demographics = load_data()
    apps_pyblp = apps.rename(columns={"price": "prices",
                                      "market_share": "shares",
                                      "market_id": "market_ids",
                                      "num_apps_category": "demand_instruments0"})
    demographics_pyblp = demographics.rename(columns={"market_id": "market_ids"})

    integration = pyblp.Integration("monte_carlo", size=200, seed=0)
    bfgs = pyblp.Optimization("bfgs", {"gtol": 1e-10})

    product_formulations = (
        pyblp.Formulation("1 + prices + average_score + in_app_purchases"),
        pyblp.Formulation("1 + prices")
    )

    agent_formulation = pyblp.Formulation("0 + demographic1")

    problem = pyblp.Problem(
        product_formulations,
        apps_pyblp,
        agent_formulation,
        demographics_pyblp,
        integration=integration
    )

    print(problem)

    initial_sigma = np.array([[0.5, 0.0],
                              [0.5, 0.5]])
    initial_pi = np.array([[0.5], [0.5]])

    result = problem.solve(initial_sigma, initial_pi, check_optimality="gradient")

    print(result)


    # integration = miniblp.integration.MonteCarloIntegration(300)
    #
    # product_formulation = miniblp.ProductFormulation(
    #     "1 + price + average_score + in_app_purchases",
    #     "1 + price",
    #     "num_apps_category"
    # )
    #
    # demographics_formulation = miniblp.DemographicsFormulation("0 + demographic1")
    #
    # problem = miniblp.Problem(
    #     product_formulation, apps,
    #     demographics_formulation, demographics,
    #     integration=integration, seed=12345
    # )
    #
    # print(problem)
    #
    # iteration = miniblp.iteration.SQUAREMIteration()
    # optimisation = miniblp.optimisation.SciPyOptimisation("Nelder-Mead")
    # # optimisation = miniblp.optimisation.SciPyOptimisation("BFGS", gtol=1e-10)
    #
    # initial_sigma = np.array([[0.5, 0.0],
    #                           [0.5, 0.5]])
    # initial_pi = np.array([[0.5], [0.5]])
    #
    # result = problem.solve(initial_sigma, initial_pi, optimisation=optimisation, iteration=iteration, parallel=False)
    # print(result)