# Here, I set up the tk dataframe with sentiment
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

Corp_TweedeKamer_V2 <- readRDS("dutch_files/Corp_TweedeKamer_V2.rds")

# The above is accessed from here: 
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN

# Filter to more than 50 terms and remove independent MPs
# Additionally, filter for year >= 2000

filtered.tk <- Corp_TweedeKamer_V2 %>%
  dplyr::filter(terms > 50) %>%
  dplyr::filter(!is.na(party)) %>%
  dplyr::filter(party != "independent") %>%
  dplyr::mutate(Year = year(date)) %>%
  dplyr::filter(Year >= 2000)

# save TK

qsave(filtered.tk, "dutch_files/tk_df.qs")

