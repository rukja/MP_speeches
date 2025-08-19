# Here, I set up the bundestag dataframe with necessary information for gender
# analysis and perform visualization
# This is just for data wrangling

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(purrr)
library(qs)
library(reshape2)

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

Corp_Bundestag_V2 <- readRDS("Corp_Bundestag_V2.rds")

# The above is accessed from here: 
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN

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
df_merged <- qread("bundestag_files/bundestag_gender_sentiment_df.qs")

# Load manifesto information
mp_setapikey("manifesto_apikey.txt")
mpds <- mp_maindataset() %>%
  dplyr::filter(countryname == "Germany") %>%
  dplyr::filter(date > 199410) %>% # Obtain data from 1994 onwards for merging
  mutate(prop_seats = absseat/totseats) %>%
  mutate(party_merge = 
           case_when(partyabbrev == "90/Greens" ~ "GRUENE",
                     partyabbrev == "PDS" ~ "PDS/LINKE",
                     partyabbrev == "L-PDS" ~ "PDS/LINKE",
                     partyabbrev == "LINKE" ~ "PDS/LINKE",
                     TRUE ~ partyabbrev)) %>%
  dplyr::filter(party_merge != "") %>%
  dplyr::filter(absseat > 0) %>%
  mutate(year_election = as.numeric(substr(date, 1, 4))) %>%
  dplyr::select(-c(date, party))

# Merge with df merge

df_merged.ye <- df_merged %>%
  mutate(year_election =
           case_when(Year <= 2001 ~ 1998,
                     Year <= 2004 ~ 2002,
                     Year <= 2008 ~ 2005,
                     Year <= 2012 ~ 2009,
                     Year <= 2016 ~ 2013,
                     Year <= 2020 ~ 2017,
                     TRUE ~ 2021
           )) %>%
  mutate(year_election = ifelse(party == "FDP" & year_election == 2013, 2009, year_election))


df.merged.mpds <- inner_join(df_merged.ye, mpds,
                             by = c("party" = "party_merge", "year_election"))

# Begin plotting (non mpds data)

gender_map <- c("female" = "blue", "male" = "red")
party_map <- c("GRUENE" = "forestgreen",
               "AfD" = "black",
               "CDU/CSU" = "deepskyblue4",
               "FDP" = "darkgoldenrod1",
               "PDS/LINKE" = "deeppink2",
               "SPD" = "darkred")

chair_map <- c("TRUE" = "burlywood3",
               "FALSE" = "darkseagreen3")

df.merged.mpds <- df.merged.mpds %>%
  dplyr::mutate(female = ifelse(gender == "female", 1, 0))

###===### Proportion of all unique speakers who were women  ====

group.by.name.party <- df.merged.mpds %>%
  distinct(party, speaker, .keep_all = TRUE) %>%
  group_by(party) %>%
  summarise(
    total = n(),
    prop_female = sum(female) / total
  )

ggplot(group.by.name.party, aes(x = party, y = prop_female, fill = party)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = party_map) +
  ggtitle("Proportion of all unique speakers who were women (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/prop_unique_speakers_who_were_women_germany.pdf", width = 9.5, height = 11)

###===### Proportion of all unique speakers who were women per party per year  ====
  
group.by.party.year <- df.merged.mpds %>%
  distinct(party, speaker, Year, .keep_all = TRUE) %>%
  group_by(party, Year) %>%
  summarise(
    total = n(),
    prop_female = sum(female) / total
  )

ggplot(group.by.party.year, aes(x = Year, y = prop_female, color = party)) +
  geom_line() +
  scale_color_manual(values = party_map) +
  ggtitle("Proportion of all unique speakers who were women (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/party_prop_unique_speakers_who_were_women_germany.pdf", width = 9.5, height = 11)

###===### Violinplot of sentiment per party ====

ggplot(df.merged.mpds, aes(x = party, y = sentiment, fill = party)) +
  geom_violin() +
  scale_fill_manual(values = party_map) +
  ggtitle("Distribution of speech sentiments by party (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/dist_speech_sentiment_party_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment per party, grouped by year ====

ggplot(df.merged.mpds, aes(x = as.factor(Year), y = sentiment, fill = party)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = party_map) +
  xlab("Year") + 
  ggtitle("Distribution of speech sentiments by party over time (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/dist_speech_sentiment_party_time_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of terms per party ====

ggplot(df.merged.mpds, aes(x = party, y = terms, fill = party)) +
  geom_boxplot() +
  scale_fill_manual(values = party_map) +
  xlab("Year") + 
  ggtitle("Distribution of terms spoken by party (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/dist_terms_party_germany.pdf", width = 9.5, height = 11)


###===### Boxplot of terms per party per gender ====

ggplot(df.merged.mpds, aes(x = party, y = terms, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  ggtitle("Distribution of terms spoken by party by sex (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/dist_terms_party_sex_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of terms per party per gender over time ====

ggplot(df.merged.mpds, aes(x = as.factor(Year), y = terms, fill = gender)) +
  geom_boxplot() +
  xlab("Year") +
  scale_fill_manual(values = gender_map, name = "sex") +
  facet_wrap(~party, scales = "free") +
  ggtitle("Distribution of terms spoken by party by sex over time (Germany)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))

ggsave("./gender_EDA/dist_terms_time_sex_germany.pdf", width = 9.5, height = 11)


###===### Boxplot of sentiment based on chairs ====

ggplot(df.merged.mpds, aes(x = chair, y = sentiment, fill = chair)) +
  geom_boxplot() +
  scale_fill_manual(values = chair_map) +
  ggtitle("Distribution of sentiment by speaker position (Germany)") +
  theme_minimal() 

ggsave("./gender_EDA/dist_sent_chair_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of terms based on chairs ====

ggplot(df.merged.mpds, aes(x = chair, y = terms, fill = chair)) +
  geom_boxplot() +
  scale_fill_manual(values = chair_map) +
  ggtitle("Distribution of terms spoken by speaker position (Germany)") +
  theme_minimal()

ggsave("./gender_EDA/dist_terms_chair_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment by sex ====

ggplot(df.merged.mpds, aes(x = gender, y = sentiment, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  xlab("sex") + 
  ggtitle("Distribution of sentiment by speaker sex (Germany)") +
  theme_minimal() 

ggsave("./gender_EDA/dist_sent_sex_germany.pdf", width = 9.5, height = 11)


###===### Boxplot of sentiment by sex by party ====

ggplot(df.merged.mpds, aes(x = party, y = sentiment, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = gender_map, name = "sex") +
  ggtitle("Distribution of sentiment by speaker sex by party (Germany)") +
  theme_minimal() 

ggsave("./gender_EDA/dist_sent_sex_party_germany.pdf", width = 9.5, height = 11)

###===### Boxplot of sentiment by sex by party, facets ====

ggplot(df.merged.mpds, aes(x = party, y = sentiment, fill = party)) +
  geom_boxplot() +
  scale_fill_manual(values = party_map) +
  facet_wrap(~gender) + 
  ggtitle("Distribution of sentiment by speaker sex by party, faceted by sex (Germany)") +
  theme_minimal() 

ggsave("./gender_EDA/dist_sent_sex_party_facet_germany.pdf", width = 9.5, height = 11)

# Integrate mpds data ====

###===### Relationship between seats and sentiment ====



df_avg <- df.merged.mpds %>%
  group_by(party, year_election, prop_seats) %>%
  summarize(median_sentiment = median(sentiment), .groups = "drop")


ggplot(df_avg, aes(x = as.numeric(prop_seats), y = median_sentiment)) +
  geom_point() +
  ggtitle("Median Sentiment vs. Proportion of Seats (Germany)") +
  xlab("Proportion of Seats") +
  ylab("Median Sentiment") +
  theme_minimal()

ggsave("./gender_EDA/sentvsseatsprop_germany.pdf", width = 9.5, height = 11)

###===### Relationship between seats and sentiment, grouped by party ====


ggplot(df_avg, aes(x = as.numeric(prop_seats), y = median_sentiment, color = party)) +
  geom_point() +
  geom_line() + 
  scale_color_manual(values = party_map) +
  ggtitle("Median Sentiment vs. Proportion of Seats (Germany)") +
  xlab("Proportion of Seats") +
  ylab("Median Sentiment") +
  theme_minimal()

ggsave("./gender_EDA/party_sentvsseatsprop_germany.pdf", width = 9.5, height = 11)

# Set up perceptions df ====

df_perc <- df_merged.ye %>%
  dplyr::mutate(female = ifelse(gender == "female", 1, 0)) %>%
  group_by(party, year_election) %>%
  summarize(total = n_distinct(speaker, party),
            median_sentiment = median(sentiment),
            prop_female = sum(female[!duplicated(speaker)]) / total,
            .groups = "drop") %>%
  inner_join(., mpds,
             by = c("party" = "party_merge", "year_election")) %>%
  select(where(~ !all(. == 0, na.rm = TRUE))) %>%
  select(-matches("_\\d+$"))

###===### Relationship between perception scores and sentiment, grouped by party ====

df_long <- df_perc %>%
  select(party, median_sentiment, starts_with("per"), rile, markeco, welfare, intpeace) %>%
  pivot_longer(
    cols = -c(party, median_sentiment),
    names_to = "variable",
    values_to = "value"
  )


ggplot(df_long, aes(x = value, y = median_sentiment, color = party)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +  
  scale_color_manual(values = party_map) + 
  theme_minimal() +
  theme(axis.text.x = element_text(size = 8)) + 
  labs(
    x = element_blank(),
    y = "Median sentiment",
    title = "Scatter plots of median sentiment vs. manifesto variables (Germany)"
  )

ggsave("./gender_EDA/manifesto_party_sentiment_germany.pdf", width = 11, height = 9.5)

###===### Relationship between perception scores and female proportion, grouped by party ====
df_long <- df_perc %>%
  select(party, prop_female, starts_with("per"), rile, markeco, welfare, intpeace) %>%
  pivot_longer(
    cols = -c(party, prop_female),
    names_to = "variable",
    values_to = "value"
  )


ggplot(df_long, aes(x = value, y = prop_female, color = party)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +  
  scale_color_manual(values = party_map) + 
  theme_minimal() +
  theme(axis.text.x = element_text(size = 8)) + 
  labs(
    x = element_blank(),
    y = "Proportion of speakers who were female",
    title = "Scatter plots of female proportion vs. manifesto variables (Germany)"
  )

ggsave("./gender_EDA/manifesto_party_female_germany.pdf", width = 11, height = 9.5)

###===### SAVE =====

qsave(df.merged.mpds, "bundestag_mpds_gender_sentiment_df.qs")
df_merged.mpds <- qread("bundestag_files/bundestag_mpds_gender_sentiment_df.qs")

###===### READ IN PARTY INFO FILE =====

EJPR.orig <- read_csv("bundestag_files/data.csv")

###===### CLEAN PARTY INFO FILE =====

EJPR.cleaned <- EJPR.orig %>%
  dplyr::filter(Year >= 2000) %>%
  dplyr::distinct(Party, Year, .keep_all = TRUE) %>%
  dplyr::mutate(Party = ifelse(Party == "CSU", "CDU/CSU", Party)) %>%
  group_by(Party, Year) %>%
  summarize(
    tot_minist = sum(per_pos)
  )

# Template code

library(dplyr)

# Sample data
data <- tibble(
  year = c(2020, 2021, 2022, 2024, 2025),
  value = c(10, 15, 20, 25, 30)
)

target_year <- 2023 # The year you are looking for

# Check if target_year exists in the data
if (target_year %in% data$year) {
  result <- data %>% filter(year == target_year)
} else {
  # If target_year is not found, get the most recent year's data
  result <- data %>% 
    arrange(desc(year)) %>% 
    slice(1) # Select the first row (most recent)
}

print(result)


#Two dataframes, one where year is present and the other with party, year, and % of positions
#For row in df 2
# Subset to that year in df 1
# If the value for the party is NA, then assign absent
# elif the value for the party is 0, then assign Min
# Else
# If the 

