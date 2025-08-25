# Here, I set up the nz dataframe with necessary information for gender
# analysis and perform visualization
# This is just for data wrangling

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(purrr)
library(qs)
library(reshape2)
library(data.table)

# Load external and related packages

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
library(manifestoR)


# Load data

Corp_nz_V2 <- readRDS("nz_files/Corp_NZHoR_V2.rds")

# The above is accessed from here: 
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN

# Filter to more than 50 terms and remove independent 
# Additionally, filter for year >= 2000

filtered.nz <- Corp_nz_V2 %>%
  dplyr::filter(terms > 50) %>%
  dplyr::filter(!is.na(party)) %>%
  dplyr::mutate(Year = year(date)) %>%
  dplyr::filter(Year >= 2000)

## Fill in gender information

# The gender() function only takes first names in as its argument

names <- sub("\\ .*", "", filtered.nz$speaker)
filtered.nz$first_name <- names


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

filtered.nz.name <- filtered.nz %>%
  dplyr::left_join(., gender.pred.unique, by = c("first_name" = "name")) %>%
  drop_na(gender)

qsave(filtered.nz, "nz_files/filtered_nz.qs")

# Create corpus

c.fbn <- corpus(
  filtered.nz.name,
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

qsave(df_merged, "nz_files/nz_gender_sentiment_df.qs")
df_merged <- qread("nz_files/nz_gender_sentiment_df.qs")


# Begin plotting (non mpds data)

gender_map <- c("female" = "blue", "male" = "red")
party_map <- c("National" = "darkblue",
               "NZ First" = "black",
               "Labour" = "indianred2",
               "ACT" = "darkgoldenrod1",
               "Green" = "green",
               "Maori" = "darkred",
               "Mana" = "grey",
               "United Future" = "mediumpurple1",
               "Alliance" = "mediumspringgreen",
               "Progressive" = "lightsteelblue1")

chair_map <- c("TRUE" = "burlywood3",
               "FALSE" = "darkseagreen3")

status_map <- c("MinC" = "violetred1",
                "MajC" = "tomato1",
                "opp" = "darkslategray2")

df.merged <- df_merged %>%
  dplyr::mutate(female = ifelse(gender == "female", 1, 0)) %>%
  dplyr::filter(party != "Independent")

###===### Proportion of all unique speakers who were women  ====

group.by.name.party <- df.merged %>%
  distinct(party, speaker, .keep_all = TRUE) %>%
  group_by(party) %>%
  summarise(
    total = n(),
    prop_female = sum(female) / total
  )

ggplot(group.by.name.party, aes(x = party, y = prop_female, fill = party)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = party_map) +
  ggtitle("Proportion of all unique speakers who were women (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/prop_unique_speakers_who_were_women_NZ.pdf", width = 9.5, height = 11)

###===### Proportion of all unique speakers who were women per party per year  ====

group.by.party.year <- df.merged %>%
  distinct(party, speaker, Year, .keep_all = TRUE) %>%
  group_by(party, Year) %>%
  summarise(
    total = n(),
    prop_female = sum(female) / total
  )

ggplot(group.by.party.year, aes(x = Year, y = prop_female, color = party)) +
  geom_line() +
  scale_color_manual(values = party_map) +
  ggtitle("Proportion of all unique speakers who were women (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/party_prop_unique_speakers_who_were_women_NZ.pdf", width = 9.5, height = 11)

###===### Violinplot of sentiment per party ====

ggplot(df.merged, aes(x = party, y = sentiment, fill = party)) +
  geom_violin() +
  scale_fill_manual(values = party_map) +
  ggtitle("Distribution of speech sentiments by party (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/dist_speech_sentiment_party_NZ.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment per party, grouped by year ====

ggplot(df.merged, aes(x = as.factor(Year), y = sentiment, fill = party)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = party_map) +
  xlab("Year") + 
  ggtitle("Distribution of speech sentiments by party over time (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/dist_speech_sentiment_party_time_NZ.pdf", width = 9.5, height = 11)

###===### Boxplot of terms per party ====

ggplot(df.merged, aes(x = party, y = terms, fill = party)) +
  geom_boxplot() +
  scale_fill_manual(values = party_map) +
  xlab("Year") + 
  ggtitle("Distribution of terms spoken by party (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/dist_terms_party_NZ.pdf", width = 9.5, height = 11)


###===### Boxplot of terms per party per gender ====

ggplot(df.merged, aes(x = party, y = terms, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  ggtitle("Distribution of terms spoken by party by sex (NZ)") +
  theme_minimal()

ggsave("./gender_EDA_nz/dist_terms_party_sex_NZ.pdf", width = 9.5, height = 11)

###===### Boxplot of terms per party per gender over time ====

ggplot(df.merged, aes(x = as.factor(Year), y = terms, fill = gender)) +
  geom_boxplot() +
  xlab("Year") +
  scale_fill_manual(values = gender_map, name = "sex") +
  facet_wrap(~party, scales = "free") +
  ggtitle("Distribution of terms spoken by party by sex over time (NZ)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))

ggsave("./gender_EDA_nz/dist_terms_time_sex_NZ.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment by sex ====

ggplot(df.merged, aes(x = gender, y = sentiment, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  xlab("sex") + 
  ggtitle("Distribution of sentiment by speaker sex (NZ)") +
  theme_minimal() 

ggsave("./gender_EDA_nz/dist_sent_sex_NZ.pdf", width = 9.5, height = 11)


###===### Boxplot of sentiment by sex by party ====

ggplot(df.merged, aes(x = party, y = sentiment, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  ggtitle("Distribution of sentiment by speaker sex by party (NZ)") +
  theme_minimal() 

ggsave("./gender_EDA_nz/dist_sent_sex_party_NZ.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment by sex by party, facets ====

ggplot(df.merged, aes(x = party, y = sentiment, fill = party)) +
  geom_boxplot() +
  scale_fill_manual(values = party_map) +
  facet_wrap(~gender) + 
  ggtitle("Distribution of sentiment by speaker sex by party, faceted by sex (NZ)") +
  theme_minimal() 

ggsave("./gender_EDA_nz/dist_sent_sex_party_facet_NZ.pdf", width = 9.5, height = 11)

# PARTY STATUS ANALYSIS

###===### READ IN PARTY INFO FILE =====

EJPR.orig <- read_csv("nz_files/data.csv")

###===### CLEAN PARTY INFO FILE =====


EJPR.cleaned <- EJPR.orig %>%
  filter(Year >= 2000) %>%
  dplyr::distinct(Party, Year, .keep_all = TRUE) %>%
  group_by(Party, Year) %>%
  summarize(
    tot_minist = sum(per_pos)
  )

EJPR.cleaned.wide <- EJPR.cleaned %>%
  pivot_wider(names_from = Party, values_from = tot_minist, values_fill = 0) 

EJPR.coalition <- EJPR.cleaned.wide %>%
  pivot_longer(-Year, names_to = "Party", values_to = "Share") %>%
  group_by(Year) %>%
  mutate(
    MaxShare = max(Share, na.rm = TRUE),
    Status = case_when(
      Share == 0 | is.na(Share) ~ "opp",
      Share == MaxShare         ~ "MajC",
      Share > 0                 ~ "MinC"
    )
  ) %>%
  ungroup() %>%
  dplyr::select(c(Year, Party, Status))


###===### MERGE =====
df_merged <- df.merged
dt_speech <- as.data.table(df_merged)
dt_coalition <- as.data.table(EJPR.coalition)

setnames(dt_speech, "party", "Party")
setkey(dt_coalition, Party, Year)
setkey(dt_speech, Party, Year)
dt_merged <- dt_coalition[dt_speech, roll = TRUE]
dt_merged[is.na(Status), Status := "opp"]

###===### ADD INFO ABOUT STATUS =====

df_merged <- df_merged %>% rename(Party = party)

df_merged <- dt_merged %>%
  dplyr::select(c(Party, speaker, Year, doc_id, Status)) %>%
  inner_join(df_merged, ., by = c("Party", "speaker", "doc_id", "Year"))

###===### SAVE =====

qsave(df_merged, "nz_files/nz_gender_sentiment_df_status.qs")

###===### PLOT =====

df_merged <- dt_merged

###===### SENTIMENT SCORE BY STATUS ======

ggplot(df_merged, aes(x = Status, y = sentiment, fill = Status)) +
  geom_violin() +
  scale_fill_manual(values = status_map) + 
  theme_minimal() +
  labs(
    x = "Government status",
    y = "Sentiment",
    title = "Sentiment by government status (NZ)"
  )

ggsave("./gender_EDA_nz/sentiment_gov_status.pdf", width = 9.5, height = 11)

###===### SENTIMENT SCORE BY STATUS BY PARTY======

ggplot(df_merged, aes(x = Status, y = sentiment, fill = Party)) +
  geom_violin() +
  scale_fill_manual(values = party_map) + 
  theme_minimal() +
  labs(
    x = "Government status",
    y = "Sentiment",
    title = "Sentiment by government status for each party (NZ)"
  )

ggsave("./gender_EDA_nz/sentiment_party_gov_status.pdf", width = 9.5, height = 11)



###===### SENTIMENT SCORE BY PARTY BY STATUS ======

ggplot(df_merged, aes(x = Party, y = sentiment, fill = Status)) +
  geom_violin() +
  scale_fill_manual(values = status_map) + 
  theme_minimal() +
  labs(
    x = "Party",
    y = "Sentiment",
    title = "Sentiment by party colored by government status (NZ)"
  )

ggsave("./gender_EDA_nz/sentiment_party_gov_status_alt.pdf", width = 9.5, height = 11)

###===### SENTIMENT SCORE FACETED BY GOVERNMENT STATUS, COLORED BY SEX ======
ggplot(df_merged, aes(x = Party, y = sentiment, fill = gender)) +
  geom_violin() +
  scale_fill_manual(values = gender_map, name = "Sex") + 
  theme_minimal() +
  facet_wrap(~Status, scales = "free") + 
  labs(
    x = "Party",
    y = "Sentiment",
    title = "Sentiment by party, government status, and speaker sex (NZ)"
  )

ggsave("./gender_EDA_nz/sentiment_party_gov_status_sex.pdf", width = 9.5, height = 11)

## DONE
q <- qread("./nz_files/filtered_nz.qs")
write.csv(q, "./nz_files/nz_filtered.csv")
