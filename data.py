import pandas as pd
import numpy as np

apps = pd.read_csv("raw_data/dataset_computational_noise.csv")

# Markets are identified by (country, year) pairs
apps["market_id"] = apps["country"] + " " + apps["year"].apply(str)

# Fix the download numbers are these are sometimes computed incorrectly
apps["downloads"] = 20 * apps["nofreviews"]
apps.loc[apps["downloads"] < apps["lowerbound"], "downloads"] = apps["lowerbound"]
apps.loc[apps["upperbound"] < apps["downloads"], "downloads"] = apps["upperbound"]

apps = apps[["market_id", "app_id", "app", "downloads", "averagescore", "inapppurchases", "newgroup", "nest"]].\
    rename(columns={"averagescore": "average_score", "inapppurchases": "in_app_purchases", "newgroup": "group"}).\
    dropna()

# Compute the market share
group = apps.groupby("market_id")
apps["market_share"] = apps["downloads"] / group["downloads"].transform("sum")

# Remove the outside good
apps = apps[apps["group"] != "OUTSIDE"].reset_index(drop=True)

# Compute the number of apps per category
group = apps.groupby(["market_id", "group"])

# Compute the number of apps per category


# The demographic data is unfortunately in one of the silliest formats ever to grace the good
# people of this earth. Convert to a dataframe.
# "It contains data on 2 different demographics, with 500 observation each, for a given year
# and country. These data are organized in a 36x1000 matrix, having markets (defined as a
# country-year pair) along the rows, and the observations for the two demographics along the
# columns."
raw_demographics = np.loadtxt("raw_data/demogr_apps.csv", delimiter=",")
demographic1, demographic2 = np.split(raw_demographics, 2, axis=1)
