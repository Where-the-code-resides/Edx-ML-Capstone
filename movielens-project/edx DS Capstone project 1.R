if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
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


#Add year of release and year of rating to dataset
edx <- edx %>% mutate(release_year = as.numeric(str_sub(title, start = -5, end = -2)),
               year_rated = year(as_datetime(timestamp)))

validation <- validation %>% mutate(release_year = as.numeric(str_sub(title, start = -5, end = -2)),
                                    year_rated = year(as_datetime(timestamp)))


#define mean
mu <- mean(edx$rating)

#Before creating any models, we will assess how the 
#explanatory variables affect the rating a user gives a film

#release year

edx %>%
  group_by(release_year) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(release_year, mean_rating)) +
  geom_point() +
  geom_smooth()

#year of rating

edx %>%
  group_by(year_rated) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(year_rated, mean_rating)) +
  geom_point() +
  geom_smooth()

#genres - first split the genres

set.seed(2, sample.kind = "Rounding")
edx_by_genre <- edx %>%
  slice(sample(1:nrow(edx),100000)) %>% # can't handle entire data set
  separate_rows(genres, sep = "\\|")    # still takes a minute or two to run

edx_by_genre %>%
  group_by(genres) %>%
  summarize(n_movies = n_distinct(movieId),
            n_ratings = n(),
            mean_rating = mean(rating),
            se = sd(rating)/sqrt(n()),
            lower = mean_rating - se*qt(0.975, n()),
            upper = mean_rating + se*qt(0.975, n())) %>%
  mutate(genres = reorder(genres, mean_rating)) %>%
  ggplot(aes(genres, mean_rating)) +
  geom_point(aes(size = n_ratings)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_hline(yintercept = mu, color = "red", linetype = "dashed") +
  geom_errorbar(aes(ymin = upper, ymax = lower)) +
  ylim(3.2, 4.25) +
  ggtitle("Mean Rating for Each Genre") +
  ylab("Mean Rating") +
  xlab("Genre") +
  labs(size = "No. of ratings")

#between users
edx %>%
  group_by(userId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram()

#between movies
edx %>%
  group_by(movieId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating))+
  geom_histogram(bins = 60)

#We are using the RMSE to determine the quality of our model
#The smaller the RMSE, the better our model is performing
#However, a RMSE too small may be a sign of overfitting, so we would want to avoid this

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#We will split edx dataset into a training and test set 

set.seed(1, sample.kind="Rounding")
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx %>% slice(-edx_test_index)
edx_temp <- edx %>% slice(edx_test_index)

# make sure userId and movieId in test set are also in train set
edx_test <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# add rows removed from test set back into train set
removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)

#start with the most basic model - the mean
pred_1 <- mean(edx_train$rating)

rmse_1 <- RMSE(pred_1, edx_test$rating)

#Store RMSE in a dataframe (will add our other RMSE results here)

rmse_results <- data.frame(Method = "Just the Average",
                           RMSE = rmse_1)


#Next model - account for movie bias
#We saw that different movies are rating differently
# i.e. some movies are more popular than others

movie_bias <- edx_train %>%
  group_by(movieId) %>%
  summarise(b_m = mean(rating - pred_1))

pred_2 <- pred_1 + edx_test %>%
  left_join(movie_bias, by = "movieId") %>%
  .$b_m

rmse_2 <- RMSE(pred_2, edx_test$rating)
rmse_2

rmse_results <- rbind(rmse_results,data.frame(Method = "Movie Bias", RMSE = rmse_2))

#Account for user bias as well as movie bias

user_bias <- edx_train %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - pred_1 - b_m))

pred_3 <- edx_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(pred = pred_1 + b_m + b_u) %>%
  .$pred

#Some of the ratings exceed 5 or are below 0.5 - amend these
pred_3[pred_3< 0.5] <- 0.5
pred_3[pred_3 > 5] <- 5

rmse_3 <- RMSE(pred_3, edx_test$rating)
rmse_results <- rbind(rmse_results,data.frame(Method = "Movie & User Bias", RMSE = rmse_3))

#To achieve greater accuracy, we will create
#A regularised model accounting for movie and user bias 
#We will use 5-fold Cross-validation to choose the ideal lambda

#regularised movie bias


lambdas <- seq(1,15, 0.5)
k = 5

set.seed(1995, sample.kind = "Rounding")
indexes <- createFolds(1:nrow(edx_train), k = k)

#create matrix to store rmse results from cv

rmses <- matrix(nrow = k, ncol = length(lambdas))

for (i in 1:k) {
  
  # define train/test set for cv
  cv_train <- edx_train %>% slice(indexes[[i]])
  cv_temp  <- edx_train %>% slice(-indexes[[i]])
  
  # make sure movieId in test set is also in train set
  cv_test <- cv_temp %>%
    semi_join(cv_train, by = "movieId") 
  
  # add removed terms back into train set
  removed <- cv_temp %>% anti_join(cv_test)
  cv_train <- rbind(cv_train, removed)
  
  # define mean rating of cv train set
  cv_mu <- mean(cv_train$rating)
  
  # the sum element of the regularised regression loss term
  just_the_sum <- cv_train %>%
    group_by(movieId) %>%
    summarize(s = sum(rating - cv_mu), n_m = n())
  
  # for each lambda, obtain predictions and rmse value
  rmses[i,] <- sapply(lambdas, function(l){
    
    predicted_ratings <- cv_test %>%
      left_join(just_the_sum, by = "movieId") %>%
      mutate(b_m_reg = s/(n_m + l)) %>%
      mutate(pred = cv_mu + b_m_reg) %>%
      .$pred
    
    return(RMSE(cv_test$rating, predicted_ratings))
  })
  rm(cv_train, cv_temp, cv_test, just_the_sum, removed, cv_mu)
}

sapply(1:5,function(k) {
  ind <- which.min(rmses[k,])
  lambdas[ind]
})

#identify optimal lambda
optm_lambda <- lambdas[which.min(colMeans(rmses))]
optm_lambda

#train model using this lambda

movie_bias_reg <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_m_reg = sum(rating - pred_1)/(n() + optm_lambda))


pred_4 <- edx_test %>%
  left_join(movie_bias_reg, by = "movieId") %>%
    mutate(pred = pred_1 + b_m_reg) %>%
    .$pred

rmse_4 <- RMSE(pred_4, edx_test$rating)
rmse_results <- rbind(rmse_results,data.frame(Method = "Movie Bias regularised", RMSE = rmse_4))

#next model: add regularised user bias (keep lambda for movie bias fixed)

rmses_2 <- matrix(nrow = k, ncol = length(lambdas))

k <- 
for (i in 1:k) {
  
  # define train/test set for cv
  cv_train <- edx_train %>% slice(indexes[[i]])
  cv_temp  <- edx_train %>% slice(-indexes[[i]])
  
  # make sure movieId and userId in test set is also in train set
  cv_test <- cv_temp %>%
    semi_join(cv_train, by = "movieId") %>%
    semi_join(cv_train, by = "userId")
  
  # add removed terms back into train set
  removed <- cv_temp %>% anti_join(cv_test)
  cv_train <- rbind(cv_train, removed)
  
  # define mean rating of cv train set
  cv_mu <- mean(cv_train$rating)
  
  # define resularised movie bias for cv train set
  cv_b_m_reg <- cv_train %>%
    group_by(movieId) %>%
    summarize(b_m_reg = sum(rating - cv_mu)/(n() + optm_lambda)) %>%
    data.frame()
  
  # for each lambda, obtain predictions and rmse value
  rmses_2[i,] <- sapply(lambdas, function(l){
    
    # regularised user bias
    cv_b_u_reg <- cv_train %>%
      left_join(cv_b_m_reg, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u_reg = sum(rating - b_m_reg - cv_mu)/(n()+l)) %>%
      data.frame()
    
    predicted_ratings <- cv_test %>%
      left_join(cv_b_m_reg, by = "movieId") %>%
      left_join(cv_b_u_reg, by = "userId") %>%
      mutate(pred = cv_mu + b_m_reg + b_u_reg) %>%
      .$pred
    
    return(RMSE(cv_test$rating, predicted_ratings))
  })
  # remove unnecessary variables
  rm(cv_train, cv_temp, cv_test, removed, cv_b_m_reg, cv_mu)
}
rm(i, ind, k) # remove unnecessary variables

optm_lambda_2 <- lambdas[which.min(colMeans(rmses_2))]
optm_lambda_2 #5

#build model using regularised movie and user bias
# regularised user bias
user_bias_reg <- edx_train %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_m_reg - pred_1)/(n()+optm_lambda_2)) %>%
  data.frame()

pred_5 <- edx_test %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  mutate(pred = pred_1 + b_m_reg + b_u_reg) %>%
  .$pred

rmse_5 <- RMSE(pred_5, edx_test$rating)
rmse_results <- rbind(rmse_results,data.frame(Method = "Movie & User Bias regularised", RMSE = rmse_5))


#test on validation dataset

mu <- mean(edx$rating)
movie_bias_reg_final <- edx %>%
  group_by(movieId) %>%
  summarize(b_m_reg = sum(rating - mu)/(n() + optm_lambda))

user_bias_reg_final <- edx %>%
  left_join(movie_bias_reg_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_m_reg - mu)/(n()+optm_lambda_2)) %>%
  data.frame()

pred_6 <- validation %>%
  left_join(movie_bias_reg_final, by = "movieId") %>%
  left_join(user_bias_reg_final, by = "userId") %>%
  mutate(pred = mu + b_m_reg + b_u_reg) %>%
  .$pred

RMSE(pred_6, validation$rating) #0.8648432 < 0.86490
