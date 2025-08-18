# Here, I perform linear regression analysis for exploring the relationship between
# mpds, sex, party, and sentiment

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(purrr)
library(qs)
library(reshape2)
library(car)

# Load external and related packages

# Leaps
library(leaps)
library(robustbase) # For robust regression

# I will compare models using AIC



# Read in data =====
df.merged.mpds <- qread("bundestag_files/bundestag_mpds_gender_sentiment_df.qs")
df.merged <- qread("bundestag_files/bundestag_gender_sentiment_df.qs")


# Linear regression has assumptions ======

hist(df.merged$sentiment)

qqPlot(df.merged$sentiment)

ks.test(df.merged$sentiment, "pnorm", 1, 2) 

summary(df.merged$sentiment)

#=====# Perform analysis without mpds variables =================

## Data is not normally distributed; sample size is large, however
## We use robust regression  =======

df.merged$party <- as.factor(df.merged$party)
df.merged$gender <- as.factor(df.merged$gender)
df.merged$year <- as.factor(df.merged$Year)
robust_model <- glmrob(sentiment ~ party + gender + gender:party, family = "gaussian", data = df.merged)
summary(robust_model)
robust_residuals <- residuals(robust_model)

## I want to compare with glm, and with lm ======


basic_model <- lm(sentiment ~ party + gender + gender:party, data = df.merged)
summary(basic_model)
basic_residuals <- residuals(basic_model)

general_model <- glm(sentiment ~ party + gender + gender:party, data = df.merged)
summary(general_model)
general_residuals <- residuals(general_model)

## Plot residuals =====

residual.df <- as.data.frame(cbind(robust_residuals, basic_residuals, general_residuals))
residual.df <- residual.df %>%
  mutate(id = row_number()) %>%
  melt(., id = "id")

residual.plot <- ggplot(residual.df, aes(x=id, y=value)) +
  geom_point() +
  ggtitle("Relationship between regression type and residual behavior") +
  xlab("Observation in dataset") +
  ylab("Residual value") + facet_wrap(~variable)




