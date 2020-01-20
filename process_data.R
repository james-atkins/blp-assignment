library(dplyr)
library(readr)

### APPS - IMPORT AND TIDYING

raw_apps <- read_csv("raw_data/dataset_computational_noise.csv")

# A number of apps have multiple (market_id, app_id) entries, when this should be a
# unique key. Perhaps this is caused by errors during the scraping process?
duplicate_apps <- raw_apps %>%
  count(app_id, country, year) %>%
  filter(n > 1)

apps <- raw_apps %>%
    # Markets are identified by (country, year) pairs
    mutate(market_id = paste(country, year)) %>%
    
    # Select only the columns we need and rename them
    select(market_id,
           country,
           year,
           app_id,
           app_name = app,
           developer,
           downloads = new_est,
           price,
           group = newgroup,
           in_app_purchases = inapppurchases,
           average_score = averagescore,
           nest) %>%
    
    # Get rid of apps with NAs in row (error in scraping process?)
    tidyr::drop_na() %>% 
    
    # Get rid of duplicate apps (see above)
    distinct(app_id, market_id, .keep_all = TRUE) %>%
    
    # Compute the market share and remove the outside good
    group_by(market_id) %>%
    mutate(market_share = downloads / sum(downloads)) %>%
    ungroup() %>%
    filter(group != "OUTSIDE")


# The share of the outside good
outside_good <- apps %>%
  group_by(country, year) %>%
  summarise(outside_good = 1 - sum(market_share))


### INSTRUMENT CALCULATIONS

# A valid instrument is everything that influences prices but it is not endogenously
# determined.

# The number of apps in a group
apps <- left_join(apps,
                  count(apps, market_id, group, name = "num_apps_group"),
                  by = c("market_id", "group"))
  

# Hausman Instrument
# Use prices in other markets as instruments
# E.g. prices in UK, US, France etc as instrument for price in Italy
# Pick up common costs but invalid if common demand shocks
apps <- apps %>%
  group_by(year, app_id) %>%
  mutate(average_price_other_countries = (sum(price) - price)/(n() - 1)) %>%
  ungroup()


### DEMOGRAPHICS - IMPORT AND TIDYING

# "It contains data on 2 different demographics, with 500 observation each, for a given year
# and country. These data are organized in a 36x1000 matrix, having markets (defined as a
# country-year pair) along the rows, and the observations for the two demographics along the
# columns."

raw_demographics <- as.matrix(read_csv("raw_data/demogr_apps.csv", col_names = FALSE))

z_score <- function(x) (x - mean(x)) / sd(x)

# !!! Assume that country-year pairs appear in the same order as they do in the apps dataset !!!
# i.e. Spain 2013, UK 2013, ...
demographics <- apps %>%
  distinct(market_id, country, year) %>% slice(rep(1:n(), each = 500)) %>%
  mutate(demographic1 = as.vector(raw_demographics[,1:500]),
         demographic2 = as.vector(raw_demographics[,501:1000]),
         demographic1_z = z_score(demographic1),
         demographic2_z = z_score(demographic2))


### EXPORT

write_csv(apps, "data/apps.csv")
write_csv(demographics, "data/demographics.csv")
