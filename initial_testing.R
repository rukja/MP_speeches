#Test

# This is just for initial analysis

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)

# Load quanteda related packages

library(quanteda)
library(spacyr)
library(quanteda.corpora)
library(quanteda.dictionaries)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)
library(quanteda.sentiment)



# Load data

Corp_Bundestag_V2 <- readRDS("Corp_Bundestag_V2.rds")

# Filter to 2000s onwards

recent.bundestag <- Corp_Bundestag_V2 %>%
  dplyr::filter(terms > 50) %>%
  dplyr::filter(!is.na(party)) %>%
  dplyr::mutate(Year = year(date))

c.recent.bundestag <- corpus(
  recent.bundestag,
  text_field = "text"      
)



# what can i do
# Step 1: Basic preprocessing
toks <- tokens_tolower(tokens(c.recent.bundestag, remove_punct = TRUE, remove_numbers = TRUE))

# Step 2: Apply dictionary lookup with exclusive = TRUE to get ONLY sentiment categories
toks_lsd <- tokens_lookup(toks, dictionary = data_dictionary_Rauh, 
                          exclusive = TRUE,  # This is key - only keeps dictionary matches
                          nested_scope = "dictionary")

# Step 3: Create dfm directly (no need for tokens_compound after lookup)
df_lsd <- dfm(toks_lsd)
df_sentiment <- convert(df_lsd, to = "data.frame")

# Add the docvars (including party) to your sentiment data
df_merged <- cbind(df_sentiment, docvars(c.recent.bundestag)) %>%
  dplyr::mutate(sentiment = log((positive+1)/(negative+1)))

# Convert to data frame

party_year_sentiment <- df_merged %>%
  group_by(party, Year) %>%
  filter(n() >= 2) %>%
  summarise(
    total_docs = n(),
    negative = sum(negative, na.rm = TRUE),
    positive = sum(positive, na.rm = TRUE),
    negative_rate = negative / total_docs,
    positive_rate = positive / total_docs,
    net_sentiment_rate = (positive - negative) / total_docs,
    sentiment = median(sentiment),
    .groups = 'drop'
  )

ggplot(party_year_sentiment, aes(x = Year, y = sentiment, color = party)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Sentiment by party over time",
       y = "Median sentiment for party for year")
