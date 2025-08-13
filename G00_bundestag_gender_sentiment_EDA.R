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
  dplyr::filter(party != "Independent") %>%
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
