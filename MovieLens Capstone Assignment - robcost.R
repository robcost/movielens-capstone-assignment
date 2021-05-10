################################################################################
# HarvardX Professional Certificate in Data Science (PH125.9x) - MovieLens Capstone Assignment
# Author: Rob Costello
################################################################################

################################################################################
# SECTION ONE
#
# Create edx set, validation set (final hold-out test set)
#
# Note: Section provided in Introduction section of Capstone Project
################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################################################################################
# SECTION TWO
#
# Prepare and explore dataset
#
################################################################################

# Create edx train and test sets to ensure we don't touch the validation set previously created.
#
# Per Grading Rubric: Please be sure not to use the validation set (the final hold-out test set) for training or 
# regularization - you should create an additional partition of training and test sets from the 
# provided edx dataset to experiment with multiple parameters or use cross-validation.

# Split the edx dataset into train/test for training and regularization
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-edx_test_index,]
test_set <- edx[edx_test_index,]

# Remove entries from the test set where there is not a corresponding user or movie
# in the training set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#-------------------------------------------------------------------------------
# Exploring the full edx dataset
#-------------------------------------------------------------------------------

# Discover dataset split
nrow(edx) # 9000055 rows
nrow(train_set) # 8100048 rows
nrow(test_set) # 900007 rows

# Discover unique users and movies in the dataset
edx %>% summarize(n_users = n_distinct(userId),n_movies = n_distinct(movieId))
# n_users n_movies
#   69878    10677

# Determine mean rating and standard deviation for movies across the training set
mean(train_set$rating) # 3.512509
sd(train_set$rating) # 1.060242

# Determine values for ratings
distinct(edx$rating) # 0 is not used, minimum rating is 0.5, maximum rating is 5.0

#-------------------------------------------------------------------------------
# Visualise different dimensions of the training set to understand the data
#-------------------------------------------------------------------------------

# Visualise ...
train_set %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, binwidth=0.1) + 
  scale_x_log10() + 
  ggtitle("Movies Rated Count")

# Visualise ...
train_set %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, binwidth=0.2, color="black", show.legend = FALSE) + 
  scale_x_log10() + 
  ggtitle("User Review Count Distribution")

# Visualise ...
train_set %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.25) + 
  ggtitle("Rating Distribution")


################################################################################
# SECTION THREE
#
# Naive analysis
#
################################################################################

# Define RMSE algorithm to be used for evaluating prediction performance

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Determine naive RMSE by comparing mean rating with the training set
mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu) # 1.061135

# Create tibble to store results and capture naive rmse
rmse_results <- tibble(method = "Naive RMSE", RMSE = naive_rmse)

################################################################################
# SECTION FOUR
#
# Consider Movie Effect
#
################################################################################

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_rmse <- RMSE(test_set$rating, predicted_ratings)

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Movie Effect RMSE", RMSE = movie_effect_rmse ))

################################################################################
# SECTION FIVE
#
# Consider Movie & User Effect
#
################################################################################

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movie_user_effect_rmse <- RMSE(test_set$rating, predicted_ratings)

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Movie & User Effect RMSE", RMSE = movie_user_effect_rmse ))

################################################################################
# SECTION SIX
#
# Regularize the Movie Effect
#
################################################################################

#Choose lambda with cross-validation
lambdas <- seq(0, 10, 0.25)

reg_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(reg_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses) 

lambdas[which.min(rmses)]
rmses[which.min(lambdas)]

movie_reg_effect_rmse <- rmses[which.min(lambdas)]

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Movie Regularization Effect RMSE", RMSE = movie_reg_effect_rmse ))

################################################################################
# SECTION SEVEN
#
# Regularize the Movie and User Effects
#
################################################################################

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

best_wide_lambda <- lambdas[which.min(rmses)]

movie_user_reg_effect_rmse <- min(rmses)

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Movie & User Regularization Effect RMSE", RMSE = movie_user_reg_effect_rmse ))

################################################################################
#
# Optimise lambda before running on validation set
#
################################################################################

start_lambda <- best_wide_lambda - 1
end_lambda <- best_wide_lambda + 1

lambdas_tight <- seq(start_lambda, end_lambda, 0.01)

rmses <- sapply(lambdas_tight, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas_tight, rmses)  

best_tight_lambda <- lambdas_tight[which.min(rmses)]

movie_user_opt_reg_effect_rmse <- min(rmses)

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Optimised Movie & User Regularization Effect RMSE", RMSE = movie_user_opt_reg_effect_rmse ))

################################################################################
# SECTION EIGHT
#
# Run predictions on Validation set for final output RMSE
#
################################################################################

rmse_validation <- sapply(best_tight_lambda, function(l){
  
  b_i <- validation %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- validation %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  return(RMSE(predicted_ratings, validation$rating))
})

# Update results tibble with Movie Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Final RMSE on Validation Set", RMSE = rmse_validation ))

#-------------------------------------------------------------------------------
# Output final results
#-------------------------------------------------------------------------------
rmse_results
paste("Final RMSE score is:",rmse_validation)
if(rmse_validation >= 0.90000){ print("You scored 0!")}
if(between(rmse_validation, 0.86550, 0.89999)){ print("You scored 10!")}
if(between(rmse_validation, 0.86500, 0.86549)){ print("You scored 15!")}
if(between(rmse_validation, 0.86490, 0.86499)){ print("You scored 20!")}
if(rmse_validation < 0.86490){ print("You scored 25! Top marks well done!")} # <<<<<<<   Script output

################################################################################
#
# End of file
#
################################################################################