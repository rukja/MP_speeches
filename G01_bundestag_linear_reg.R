# Here, I perform linear regression analysis for exploring the relationship between
# mpds, sex, party, and sentiment

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(purrr)
library(qs)
library(reshape2)

# Load external and related packages


# Read in data =====
df.merged.mpds <- qread("bundestag_files/bundestag_mpds_gender_sentiment_df.qs")
df.merged <- qread("bundestag_files/bundestag_gender_sentiment_df.qs")