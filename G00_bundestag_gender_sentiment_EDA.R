# Here, I set up the bundestag dataframe with necessary information for gender
# analysis and perform visualization

#Test

# This is just for data wrangling

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(purrr)
library(qs)
library(reshape2)

# Load quanteda related packages

library(quanteda)
library(spacyr)
library(quanteda.corpora)
library(quanteda.dictionaries)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)
library(quanteda.sentiment)
library(genderdata)
library(gender)



# Load data

Corp_Bundestag_V2 <- readRDS("Corp_Bundestag_V2.rds")

# Filter to more than 50 terms and remove independent MPs
# Additionally, filter for year >= 2000

filtered.bundestag <- Corp_Bundestag_V2 %>%
  dplyr::filter(terms > 50) %>%
  dplyr::filter(!is.na(party)) %>%
  dplyr::filter(party != "independent") %>%
  dplyr::mutate(Year = year(date)) %>%
  dplyr::filter(Year >= 2000)

## Fill in gender information

# The gender() function only takes first names in as its argument

names <- sub("\\ .*", "", filtered.bundestag$speaker)
filtered.bundestag$first_name <- names


gender.pred.kantrowitz <- gender(names, method = c("kantrowitz")) %>%
  drop_na() %>%
  unique()
gender.pred.ssa <- gender(names, method = c("ssa")) %>%
  unique() %>%
  dplyr::select(name, gender)
gender.pred.ipums <- gender(names, method = c("ipums")) %>%
  unique() %>%
  dplyr::select(name, gender)
gender.pred.napp <- gender(names, method = c("napp")) %>%
  unique() %>%
  dplyr::select(name, gender)

gender.pred <- rbind(gender.pred.ssa, gender.pred.ipums, gender.pred.napp, gender.pred.kantrowitz)

gender.pred.unique <- gender.pred %>%
  distinct(name, .keep_all = TRUE)

corp.unique <- unique(names)

filtered.bundestag.name <- filtered.bundestag %>%
  dplyr::left_join(., gender.pred.unique, by = c("first_name" = "name")) %>%
  drop_na(gender)

# Create corpus

c.fbn <- corpus(
  filtered.bundestag.name,
  text_field = "text"      
)

# Calculate sentiment

toks <- tokens_tolower(tokens(c.fbn, remove_punct = TRUE, remove_numbers = TRUE))


toks_lsd <- tokens_lookup(toks, dictionary = data_dictionary_Rauh, 
                          exclusive = TRUE,  
                          nested_scope = "dictionary")


df_lsd <- dfm(toks_lsd)
df_sentiment <- convert(df_lsd, to = "data.frame")

df_merged <- cbind(df_sentiment, docvars(c.fbn)) %>%
  dplyr::mutate(sentiment = log((positive+1)/(negative+1)))

# Save sentiment calculation

qsave(df_merged, "bundestag_gender_sentiment_df.qs")

# Group by variables of interest

party_year <- df_merged %>%
  group_by(party, Year) %>%
  filter(n() >= 2) %>%
  summarise(
    total_docs = n(),
    negative = sum(negative, na.rm = TRUE),
    positive = sum(positive, na.rm = TRUE),
    female = sum(gender == "female"),
    negative_rate = negative / total_docs,
    positive_rate = positive / total_docs,
    net_sentiment_rate = (positive - negative) / total_docs,
    female_prop = female / total_docs,
    sentiment = median(sentiment),
    .groups = 'drop'
  )

# Create output dir for plots

dir.create(file.path("./", "gender_EDA"), showWarnings = FALSE)


# Create plots

library(reshape2)

generate_sentiment_eda <- function(data, parliament_name = "Parliament") {
  
  # Create folder for saving gender_EDA
  if (!dir.exists("gender_EDA")) dir.create("gender_EDA")
  
  ## 1. Sentiment vs Female Proportion
  p1 <- ggplot(data, aes(x = female_prop, y = net_sentiment_rate, color = party)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE) +
    theme_minimal() +
    labs(title = paste(parliament_name, "Sentiment vs Female Proportion by Party"),
         x = "Female proportion", y = "Net sentiment rate")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_sentiment_vs_female_prop.pdf"),
         plot = p1, width = 8, height = 5)
  
  ## 2. Sentiment vs Female Proportion faceted by Year
  p2 <- ggplot(data, aes(x = female_prop, y = net_sentiment_rate, color = party)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    facet_wrap(~ Year) +
    theme_minimal() +
    labs(title = paste(parliament_name, "Sentiment vs Female Proportion by Year"),
         x = "Female proportion", y = "Net sentiment rate")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_sentiment_vs_female_prop_by_year.pdf"),
         plot = p2, width = 12, height = 8)
  
  ## 3. Female Proportion over Time by Party
  p3 <- ggplot(data, aes(x = Year, y = female_prop, color = party)) +
    geom_line(size = 1) +
    geom_point() +
    theme_minimal() +
    labs(title = paste(parliament_name, "Female proportion over time by Party"),
         y = "Female proportion")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_female_prop_over_time.pdf"),
         plot = p3, width = 10, height = 6)
  
  ## 4. Sentiment over Time by Female Proportion Quartile
  data_quartile <- data %>% mutate(female_group = ntile(female_prop, 4))
  
  p4 <- ggplot(data_quartile, aes(x = Year, y = net_sentiment_rate, color = factor(female_group))) +
    geom_line(aes(group = interaction(party, female_group))) +
    geom_point() +
    theme_minimal() +
    labs(title = paste(parliament_name, "Sentiment over time by Female proportion quartile"),
         color = "Female proportion quartile")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_sentiment_by_female_quartile.pdf"),
         plot = p4, width = 10, height = 6)
  
  ## 5. Sentiment trends by Party and Female Category (High/Low)
  data_cat <- data %>% mutate(female_cat = ifelse(female_prop >= median(female_prop), "High", "Low"))
  
  p5 <- ggplot(data_cat, aes(x = Year, y = net_sentiment_rate, color = female_cat)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ party) +
    theme_minimal() +
    labs(title = paste(parliament_name, "Sentiment trends by female representation"),
         color = "Female proportion category")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_sentiment_trends_by_female_cat.pdf"),
         plot = p5, width = 12, height = 8)
  
  ## 6. Correlation Heatmap
  cor_vars <- data %>% select(sentiment, female_prop, net_)
  cor_mat <- cor(cor_vars, use = "pairwise.complete.obs")
  
  p6 <- ggplot(melt(cor_mat), aes(Var1, Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 2)), color = "white") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    theme_minimal() +
    labs(title = paste(parliament_name, "Correlation matrix"))
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_correlation_matrix.pdf"),
         plot = p6, width = 6, height = 6)
  
  ## 7. Bubble plot: Sentiment vs Female Proportion, size = total_docs
  p7 <- ggplot(data, aes(x = female_prop, y = net_sentiment_rate, size = total_docs, color = party)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(title = paste(parliament_name, "Sentiment vs Female Representation"),
         x = "Female proportion", y = "Net sentiment rate", size = "Total speeches")
  
  ggsave(filename = paste0("gender_EDA/", parliament_name, "_bubble_sentiment_vs_female.pdf"),
         plot = p7, width = 10, height = 6)
  
  message("EDA gender_EDA generated and saved in the 'gender_EDA/' folder.")
}

generate_sentiment_eda(party_year, "Bundestag")
