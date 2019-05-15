if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(gridExtra)
library(Hmisc)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
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

# Explore data set "edx"
head(edx)
dim(edx)

# Identify missing values
sum(is.na(edx))
sum(is.na(validation))

# Identify number of unique users, movies, and genres
edx %>% 
  summarise(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_genres = n_distinct (genres))

# Compute and plot average rating across users, movies and genres
U1 <- edx %>% 
  group_by(userId) %>% 
  summarise(avg = mean(rating)) %>% 
  ggplot(aes(avg)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Average rating across users")

M1 <- edx %>% 
  group_by(movieId) %>% 
  summarise(avg = mean(rating)) %>% 
  ggplot(aes(avg)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Average rating across movies")

G1 <- edx %>% 
  group_by(genres) %>% 
  summarise(avg = mean(rating)) %>% 
  ggplot(aes(avg)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Average rating across genres")

grid.arrange(U1, M1, G1, nrow = 3, newpage = TRUE)

# Compute and plot number of ratings across users, movies and genres
U2 <- edx %>% 
  group_by(userId) %>% 
  summarise(n = n()) %>% 
  filter (n <= 1000) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Number of ratings across users")

M2 <- edx %>% 
  group_by(movieId) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Number of ratings across movies")

G2 <- edx %>% 
  group_by(genres) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Number of ratings across genres")

grid.arrange(U2, M2, G2, nrow = 3, newpage = TRUE)

left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  group_by(title) %>%
  summarise (b_i = mean(b_i), residual = mean(residual), n = n())

mistakes %>% arrange(desc(b_i)) %>% 
  slice (1:10) %>% knitr::kable(caption = "10 best movies")
mistakes %>% arrange(b_i) %>% 
  slice (1:10) %>% knitr::kable(caption = "10 worst movies")

# Write a function that computes the RMSE for vectors of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Build a first model called "Just the average"
# Model assumes the same rating for all movies and users with all the differences explained by random variation
mu_hat <- mean(edx$rating)
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

rmse_results <- tibble(method = "Just The Average", RMSE = naive_rmse)

# Modeling movie effects
# Augment previous model by adding average ranking for movie i
# Least square estimate b_i is the average of rating - mu for each movie i
mu <- mean(edx$rating)

movie_avgs <- validation %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))

# Create predicted ratings
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Generate RMSE, create and print result table for movie effect model
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",  
                                 RMSE = model_1_rmse))

rmse_results

# Modeling genre effects
# Augment movie effect model by adding average ranking for genres
genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu - b_i))

# Create predicted ratings
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_g) %>%
  pull(pred)

# Impute predicted ratings by using mean method because of missing values ("NA")
new_predicted_ratings <- impute(predicted_ratings, mean)

# Generate RMSE, create and print result table for movie & genre effects model
model_2_rmse <- RMSE(new_predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + Genre Effect Model",  
                                 RMSE = model_2_rmse))
rmse_results

# Modeling user effects
# Augment movie effect model by adding average ranking for users
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Create predicted ratings
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Impute predicted ratings by using mean method because of missing values ("NA")
new_predicted_ratings <- impute(predicted_ratings, mean)

# Generate RMSE, create and print result table for movie & user effects model
model_3_rmse <- RMSE(new_predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effect Model",  
                                 RMSE = model_3_rmse))

rmse_results

# Regularization of movie & user effects
# Explore mistakes in first model by using movie effects
mistakes <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  group_by(title) %>%
  summarise (b_i = mean(b_i), residual = mean(residual), n = n())

mistakes %>% arrange(desc(b_i)) %>% 
  slice (1:10) %>% knitr::kable(caption = "10 best movies")
mistakes %>% arrange(b_i) %>% 
  slice (1:10) %>% knitr::kable(caption = "10 worst movies")


# Construct regularized movie and user effect model to control total variability of effects
# Penalty terms: Use cross-validation to choose tuning parameter lambda 
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Show optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda


# Generate RMSE, create and print result table for regularized movie & user effect model
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",  
                                 RMSE = min(rmses)))
rmse_results %>% knitr::kable()
# The final model achieves a RMSE of 0.86


