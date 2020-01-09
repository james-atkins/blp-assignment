library(dplyr)
library(ggplot2)
library(cowplot)
library(xtable)

options(xtable.booktabs = TRUE,
        xtable.caption.placement = "top",
        xtable.include.rownames = FALSE)

ggsave <- function(filename, plot = ggplot2::last_plot(), width = 9, height = NA, units = "in") {
  cowplot::ggsave2(filename, plot, width = width, height = height, units = units)
}

source("process_data.R")

apps %>%
  group_by(app_id) %>%
  summarise(num_markets = length(unique(market_id))) %>%
  count(num_markets) %>%
  ggplot(aes(x = num_markets)) +
    geom_col(aes(y = n)) +
    labs(x = "Number of markets", y = "Unique apps") +
    theme_minimal_hgrid()

ggsave("output/unique_apps_per_num_markets.png")

# !!! THIS CODE IS WRONG !!!
# apps %>%
#   group_by(app_id) %>%
#   summarise(num_countries = length(unique(country)),
#             num_years = length(unique(year))) %>%
#   group_by(num_countries, num_years) %>%
#   summarise(n = n()) %>%
#   count(num_countries, num_years) %>%
#   tidyr::pivot_wider(names_from = num_years, values_from = n) %>% 
#   xtable(caption = "Apps in number of ",
#          label = "tab:apps_per_num_countries_years",
#          align = c("rowname_not_used", "c", "r", "r", "r")) %>%
#   print(file = "output/apps_per_num_countries_years.tex",
#         add.to.row=list(
#           pos = list(0, 0),
#           command = c("& \\multicolumn{3}{c}{Years} \\\\ \\cmidrule{2-4}","Countries & 1 & 2 & 3\\\\\n")
#         ),
#         include.colnames = FALSE)


# Table of outside good shares per country and year
outside_good %>%
  mutate(outside_good = scales::percent(outside_good, accuracy = 0.01)) %>%
  tidyr::pivot_wider(names_from = year, values_from = outside_good) %>%
  rename(Country = country) %>%
  xtable(caption = "Share of outside good", label = "tab:share_outside_good") %>%
  print(file = "output/share_outside_good.tex")

apps %>%
  filter(country == "United States", year == 2019) %>%
  top_n(50, market_share) %>%
  ggplot(aes(y = market_share, x = reorder(app_name, market_share))) +
  geom_col(aes(fill = group)) +
  # ggtitle("Top 50 Apps in the United States in 2019") +
  labs(y = "Market Share", fill = "Category") +
  theme_minimal_hgrid() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

ggsave("output/top_50_us_2019.png")

kernel_density_plot <- function(demographic) {
  demographic <- enquo(demographic)
  
  ggplot(demographics, aes(x = !!demographic, colour = as.factor(year))) +
    geom_line(stat="density") +
    facet_wrap(~ country) +
    scale_x_continuous(labels = scales::comma) +
    labs(x = NULL, colour = NULL) +
    theme_cowplot() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.line.y = element_blank(),
          axis.line.x = element_blank(),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
          legend.position = "bottom")
}

kernel_density_plot(demographic1)
ggsave("output/kernel_density_demographic1.png")

kernel_density_plot(demographic2)
ggsave("output/kernel_density_demographic2.png")
