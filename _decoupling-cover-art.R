library(RcppRoll)
library(covidcast)
library(tidyverse)
library(modeltools)
shape <- 3.5
scale <- 4
window_max <- 49
kernel <- dgamma(seq(window_max), shape = shape, scale = scale)
kernel <- kernel / sum(kernel) * window_max
cases <- covidcast_signal("jhu-csse", "confirmed_incidence_num", geo_type = "state")
hosp <- covidcast_signal("hhs", "confirmed_admissions_covid_1d", geo_type = "state")

cases <- cases %>%
  filter(time_value > "2020-06-01", geo_value != "pr") %>%
  group_by(geo_value) %>%
  arrange(time_value) %>%
  mutate(kvalue = roll_sumr(value, n = window_max, weights = kernel)) %>%
  rename(cases = value)
hosp <- hosp %>%
  filter(time_value > "2020-06-01", geo_value != "pr") %>%
  rename(hosp = value)

comb <- left_join(
  cases %>% select(geo_value, time_value, cases, kvalue),
  hosp %>% select(geo_value, time_value, hosp))

# hosp_regression = function(x, ...) {
#   n = nrow(x)
#   return(tryCatch(suppressWarnings(suppressMessages({
#     # Fit a regression on the most recent w days
#     lm_obj = glm(log(hosp+1) ~ offset(log(kvalue+1)), data = x, family = "quasipoisson")
#     # Return the fitted lm object and prediction
#     list(lm_obj = lm_obj, intercept = coef(lm_obj)[1])
#   })),
#   error = function(e) return(list(lm_obj = NA, intercept = NA))))
# }

# comb <- slide_by_geo(comb, slide_fun = hosp_regression, n = 28,
#                      col_type = "list", col_name = "reg_out")

comb <- comb %>%
  # rowwise() %>%
  # mutate(intercept = reg_out$intercept) %>%
  group_by(geo_value) %>%
  mutate(roll_avg = roll_meanr(hosp / kvalue, n=28, na.rm = TRUE)) %>%
  group_by(time_value) %>%
  mutate(avg = mean(roll_avg, na.rm = TRUE), med = median(roll_avg, na.rm = TRUE),
         tot = sum(cases, na.rm = TRUE) / 1000000)

vacc <- read_csv("~/Downloads/COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv") %>%
  group_by(Location) %>%
  #summarise(vaxed = max(Administered_Dose1_Recip_12PlusPop_Pct, na.rm = TRUE)) %>%
  summarise(vaxed = max(Series_Complete_Pop_Pct, na.rm = TRUE)) %>%
  mutate(geo_value = tolower(Location)) %>%
  select(-Location)

comb <- left_join(comb, vacc)

  # mutate(avg = mean(intercept, na.rm = TRUE), med = median(intercept, na.rm = TRUE),
  #        tot = sum(cases, na.rm = TRUE) / 1000000)
us <- comb %>%
  group_by(time_value) %>%
  summarise(kvalue = sum(kvalue, na.rm = TRUE),
            hosp = sum(hosp, na.rm = TRUE)) %>%
  mutate(roll_avg = roll_meanr(hosp / kvalue, n=28, na.rm = TRUE))

start_day = "2020-10-01"
line_cols = viridisLite::plasma(6, begin = .1)
ggplot(comb %>%
         filter(time_value > start_day, !(geo_value %in% c("as", "mp", "vi", "gu"))) %>%
         group_by(time_value) %>%
         mutate(cvax = cut(vaxed, c(0,40,45,50,55,60,100))),
       aes(time_value)) +
  # annotate("rect", xmin = Sys.Date() - 21, xmax = Sys.Date(),
  #          ymin = 0, ymax = Inf, fill = "goldenrod1", alpha = .4) +
  geom_line(aes(y = roll_avg, color = cvax, group = geo_value),
            alpha = 1) +
  theme_bw(base_family = "Times") +
  scale_color_manual(values = line_cols) +
  # scale_color_viridis_b(
  #   begin = .1, option = "C",
  #   breaks = c(0,40,50,60,70,100)) +
  guides(color = guide_legend(title.position = "top", nrow=1,
                              override.aes = list(size = 2))) +
  geom_ribbon(aes(ymin = 0, ymax = tot/300), fill = "cornflowerblue", alpha = .2) +
  scale_y_log10(breaks = c(1e-1,1e-2,1e-3,1e-4, 1e-5), labels = scales::label_percent(.01)) + #,
  #            labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  #coord_cartesian(ylim = c(1e-2, 1e-4)) +
  scale_x_date(expand = expansion(0), date_breaks = "2 months", date_labels = "%b %Y") +
  labs(
    y = "",
    x = "",
    colour = "% of population fully vaccinated") +
  geom_line(data = us %>% filter(time_value > start_day),
            aes(y = roll_avg), color = "black", size = 1.25) +
  theme(legend.position = "none", plot.margin = margin(t=5, r = 10))

ggsave("cover.png", width = 7, height = 6)
