library(dplyr)
library(readr)
library(tidyr)

raw_apps <- read_csv("raw_data/dataset_computational_noise.csv")

apps <- raw_apps %>%
  mutate(market_id = paste(country, year),
         downloads = 20 * nofreviews,
         downloads = if_else(downloads > upperbound, upperbound, downloads),
         downloads = if_else(downloads < lowerbound, lowerbound, downloads)) %>%
  select(market_id,
         app_id,
         app,
         downloads,
         price,
         average_score = averagescore,
         in_app_purchases = inapppurchases,
         group = newgroup,
         nest) %>%
  drop_na() %>%
  group_by(market_id) %>%
  mutate(market_share = downloads / sum(downloads)) %>%
  ungroup() %>% 
  filter(group != "OUTSIDE")

num_apps_category <- apps %>%
  count(market_id, group, name = "num_apps_category")

apps <- left_join(apps, num_apps_category, by = c("market_id", "group"))

# The demographic data is unfortunately in one of the silliest formats ever to grace the good
# people of this earth. Convert to a dataframe.
# "It contains data on 2 different demographics, with 500 observation each, for a given year
# and country. These data are organized in a 36x1000 matrix, having markets (defined as a
# country-year pair) along the rows, and the observations for the two demographics along the
# columns."
raw_demographics <- as.matrix(read_csv("raw_data/demogr_apps.csv", col_names = FALSE))

demographics <- tibble(
  market_id = rep(unique(apps$market_id), each = 500),
  demographic1 = as.vector(raw_demographics[,1:500]),
  demographic2 = as.vector(raw_demographics[,501:ncol(raw_demographics)])
)

write_csv(apps, "data/apps.csv")
write_csv(demographics, "data/demographics.csv")
