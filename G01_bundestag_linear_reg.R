# Here, I perform linear regression analysis for exploring the relationship between
# mpds, sex, Party, and sentiment

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

df.merged$Party <- as.factor(df.merged$Party)
df.merged$gender <- as.factor(df.merged$gender)
df.merged$year <- as.factor(df.merged$Year)
df.merged$Status <- as.factor(df.merged$Status)

df.merged$Party  <- relevel(df.merged$Party, ref = "PDS/LINKE")
df.merged$Status <- relevel(df.merged$Status, ref = "opp")
df.merged$gender <- relevel(df.merged$gender, ref = "male")

form <- sentiment ~ Party + gender + gender:Party + Status + Party:Status


robust_model <- glmrob(form, family = "gaussian", data = df.merged)
summary(robust_model)
robust_residuals <- residuals(robust_model)

## I want to compare with glm, and with lm ======


basic_model <- lm(form, data = df.merged)
summary(basic_model)
basic_residuals <- residuals(basic_model)

general_model <- glm(form, data = df.merged)
summary(general_model)
general_residuals <- residuals(general_model)

## Conduct emmeans analysis =====

emm.g.sp <- emmeans(general_model, ~ Status | Party + gender)
pairs(emm.g.sp)

emm.g.ps <- emmeans(general_model, ~ gender | Party + Status)
pairs(emm.g.ps)

emm.simple <- emmeans(general_model, ~ Status | Party )
pairs(emm.simple)

emm.gender <- emmeans(general_model, ~ gender | Party)
pairs(emm.gender)


RG5 <- ref_grid(general_model, cov.reduce = FALSE)


emmip(RG5, Party ~ gender | Status, style = "factor")
emmip(RG5, Party ~ Status, style = "factor")

emm_df <- as.data.frame(emm.simple)


ggplot(emm_df, aes(x = emmean, y = Status, color = Party)) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL),
                 position = position_dodge(width = 0.5), height = 0.2) +
  labs(x = "Estimated Sentiment", y = "Coalition Status") +
  theme_minimal()

emmip(emm.simple, sentimentstyle = "Factor")
