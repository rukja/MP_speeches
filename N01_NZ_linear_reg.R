# Here, I perform linear regression analysis for exploring the relationship between
#sex, Party, and sentiment

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
library(emmeans)


# Read in data =====
df.merged <- qread("nz_files/nz_gender_sentiment_df_status.qs")


# Linear regression has assumptions ======

hist(df.merged$sentiment)

qqPlot(df.merged$sentiment)

ks.test(df.merged$sentiment, "pnorm", 1, 2) 

summary(df.merged$sentiment)

#=====# Perform analysis without mpds variables =================

## Data is not normally distributed; sample size is large, however
## We use robust regression  =======

df.merged$Party <- as.factor(df.merged$Party)
df.merged$gender <- as.factor(df.merged$gender)
df.merged$year <- as.factor(df.merged$Year)
df.merged$Status <- as.factor(df.merged$Status)

df.merged$Party  <- relevel(df.merged$Party, ref = "Green")
df.merged$Status <- relevel(df.merged$Status, ref = "opp")
df.merged$gender <- relevel(df.merged$gender, ref = "male")

form <- as.numeric(sentiment) ~ Party + gender + gender:Party + Status + Party:Status


robust_model <- glmrob(form, family = "gaussian", data = df.merged)
summary(robust_model)


## I want to compare with glm, and with lm ======


basic_model <- lm(form, data = df.merged)
summary(basic_model)


general_model <- glm(form, data = df.merged)
summary(general_model)


par(mfrow=c(2,2))

plot(general_model)

# The model does not fit well 


