import pandas as pd
import numpy as np

import miniblp

apps = pd.read_csv("raw_data/dataset_computational_noise.csv")
apps["market_id"] = apps["country"] + " " + apps["year"].apply(str)

# demographics = pd.read_csv("raw_data/demographics.csv") \
#     .rename(columns={"Year ": "year", "Country": "country", "GDP per capita": "gdp_per_capita", "Age ": "age"})
# demographics["market_id"] = demographics["country"] + " " + demographics["year"].apply(str)

# The demographic data is unfortunately in one of the silliest formats ever to grace the good people of this earth.
# "It contains data on 2 different demographics, with 500 observation each, for a given year and country.
# These data are organized in a 36x1000 matrix, having markets (defined as a country-year pair) along the rows, and the
# observations for the two demographics along the columns."
demographics_raw = np.genfromtxt("raw_data/demogr_apps.csv", delimiter=",")
demographic1, demographic2 = np.hsplit(demographics_raw, 2)

demographics = pd.DataFrame({"market_id": np.repeat(apps["market_id"].unique(), 500),
              "demographic1": demographic1.ravel(),
              "demographic2": demographic2.ravel()})

def compute_market_share(group):
    return group["estimateddownloads"] / group["estimateddownloads"].sum()


apps["market_share"] = apps.groupby("market_id", as_index=False).apply(compute_market_share).rename(
    "market_share").reset_index(level=0, drop=True)
num_apps_category = apps.groupby(by=["market_id", "newgroup"]).count()["rank"].reset_index().rename(
    columns={"rank": "num_apps_category"})
apps = pd.merge(apps, num_apps_category, on=("market_id", "newgroup"))
apps = apps.dropna()
apps = apps[apps["estimateddownloads"] > 0]

# Calculate the instrument

integration = miniblp.integration.MonteCarloIntegration(200)
iteration = miniblp.iteration.SimpleFixedPointIteration()

problem = miniblp.Problem("1 + price + averagescore + inapppurchases",
                          "price",
                          "num_apps_category",
                          "demographic1", apps, demographics,
                          integration, iteration)

theta2 = miniblp.Theta2(np.eye(2), np.array([[1], [1]]))

# theta2 = miniblp.Theta2(np.zeros(shape=(2, 2)), np.zeros(shape=(2, 1)))

print(problem._compute_delta(theta2))