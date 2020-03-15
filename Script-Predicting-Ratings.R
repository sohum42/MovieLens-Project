################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

################################
# Analysis
################################

summary(edx)

#Get counts of each rating (0.5,..., 4.5, 5.0)
rating_counts = edx %>%
  group_by(rating) %>%
  summarize(rate_cnts = n()) %>%
  mutate(rating = as.factor(rating)) %>%
  ggplot(data = ., aes(rating, rate_cnts)) +
  geom_bar(stat = "identity")
rating_counts

#Get average rating of movies with at least 25,000 ratings
movie_cnts = edx %>%
  group_by(movieId, title) %>%
  summarize(rate_cnts = n(), avg_rating = mean(rating)) %>%
  arrange(-rate_cnts) %>%
  filter(rate_cnts >= 25000) %>%
  ggplot(data = ., aes(title, avg_rating)) +
  geom_bar(aes(fill = "avg_rating"), stat = "identity") + 
  ggtitle("Average Rating by Movie (min. 25K ratings)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) + 
  labs(y = "Average Rating", x = "Title")
movie_cnts

#Get average rating of users with at least 3000 ratings
user_cnts = edx %>%
  group_by(userId) %>%
  summarize(rate_cnts = n(), avg_rating = mean(rating), 
            se_rating = sd(rating)/sqrt(n())) %>%
  mutate(userId = as.factor(userId)) %>%
  arrange(-rate_cnts) %>%
  filter(rate_cnts >= 3000) %>%
  ggplot(data = ., aes(userId, avg_rating)) +
  geom_bar(aes(fill = "avg_rating"), stat = "identity") +
  geom_errorbar(aes(ymin = avg_rating-2*se_rating, 
                    ymax = avg_rating+2*se_rating)) + 
  ggtitle("Average Rating by User (min. 3000 ratings)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) + 
  labs(y = "Average Rating", x = "User ID")
user_cnts

#Average Ratings by Genre (min. 100,000 ratings)
avg_genre_rating = edx %>%
  group_by(genres) %>%
  summarize(cnts = n(), avg_rating = mean(rating), 
            se_rating = sd(rating)/sqrt(n())) %>%
  filter(cnts >= 100000)

#Create Error Bar Plot using Mean and SE from above
ggplot(avg_genre_rating, aes(reorder(genres, -avg_rating), 
                             avg_rating)) +
  geom_point() + geom_errorbar(aes(ymin = avg_rating-2*se_rating,
                                   ymax = avg_rating+2*se_rating)) +
  ggtitle("Average Rating by Genre (min. 100K ratings)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) + 
  labs(y = "Average Rating", x = "Genre")

#Average Ratings by Year
avg_yr_rating = edx %>%
  mutate(year = year(as.Date.POSIXct(timestamp))) %>%
  group_by(year) %>%
  summarize(cnts = n(), avg_rating = mean(rating))

ggplot(avg_yr_rating, aes(year, avg_rating)) +
  geom_bar(fill = "blue", stat="identity") + 
  ggtitle("Average Rating by Year") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) + 
  labs(y = "Average Rating", x = "Year")

################################
# Prediction Methods
################################

#Split edx into training and test set
#Follows same steps as when separating edx and validation set
set.seed(1, sample.kind = "Rounding")
test_index = createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_edx = edx[-test_index,]
temp_edx = edx[test_index,]

test_edx = temp_edx %>%
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

removed = anti_join(temp_edx, test_edx)
train_edx = rbind(train_edx, removed)

rm(removed, temp_edx)

##Simple model based on overall average of all ratings
ovrl_avg = mean(train_edx$rating)

#Predict all movies to have the average rating
simple_RMSE = RMSE(test_edx$rating, ovrl_avg)
simple_RMSE 

##Model based on movie effects
movie_avg = train_edx %>%
  group_by(movieId) %>%
  summarize(movie_i = mean(rating - ovrl_avg))

#Predict based on the average rating for each movie
movie_based_pred = test_edx %>%
  left_join(movie_avg, by = 'movieId') %>%
  mutate(pred = ovrl_avg + movie_i) %>%
  pull(pred)

movie_RMSE = RMSE(test_edx$rating, movie_based_pred)
movie_RMSE


##Model based on movie and user effects
um_avg = train_edx %>%
  left_join(movie_avg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_i = mean(rating - ovrl_avg - movie_i))

#Predict based on the average rating for each user and movie
um_pred = test_edx %>%
  left_join(movie_avg, by = 'movieId') %>%
  left_join(um_avg, by = 'userId') %>%
  mutate(pred_um = ovrl_avg + movie_i + user_i) %>%
  pull(pred_um)

um_RMSE = RMSE(test_edx$rating, um_pred)
um_RMSE

#Predicted Ratings above 5 or below 0.5
sum(um_pred > 5) #4311 ratings predicted above 5.0
sum(um_pred < 0.5) #229 ratings predicted below 0

#Predict based on the average rating for each user and movie with adjustment for unacceptable ratings.
um_adj_pred = test_edx %>%
  left_join(movie_avg, by = 'movieId') %>%
  left_join(um_avg, by = 'userId') %>%
  mutate(pred_um = ovrl_avg + movie_i + user_i) %>%
  mutate(pred_um_adj = ifelse(pred_um > 5, 5.0, 
                              ifelse(pred_um < 0.5, 0.5, pred_um))) %>%
  pull(pred_um_adj)

um_rmse_adj = RMSE(test_edx$rating, um_adj_pred)
um_rmse_adj


##Model based on movie, user, and genre effects
umg_avg = train_edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(um_avg, by = "userId") %>%
  group_by(genres) %>%
  summarize(genre_i = mean(rating - ovrl_avg - movie_i - user_i))

#Predict based on the average rating for each user, movie, and genre with adjustment
umg_pred_adj = test_edx %>%
  left_join(movie_avg, by = 'movieId') %>%
  left_join(um_avg, by = 'userId') %>%
  left_join(umg_avg, by = 'genres') %>%
  mutate(pred_umg = ovrl_avg + movie_i + user_i + genre_i) %>%
  mutate(pred_umg_adj = ifelse(pred_umg > 5, 5.0, 
                               ifelse(pred_umg < 0.5, 0.5, pred_umg))) %>%
  pull(pred_umg_adj)

umg_RMSE_adj = RMSE(test_edx$rating, umg_pred_adj)
umg_RMSE_adj 


##Model based on movie, user, genre, and year effects
umgy_avg = train_edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(um_avg, by = "userId") %>% 
  left_join(umg_avg, by = "genres") %>%
  mutate(year = year(as.Date.POSIXct(timestamp))) %>%
  group_by(year) %>%
  summarize(year_i = mean(rating - ovrl_avg - movie_i - user_i -                       genre_i))

#Predict based on the average rating for each user, movie, genre, and year
umgy_pred = test_edx %>%
  mutate(year = year(as.Date.POSIXct(timestamp))) %>%
  left_join(movie_avg, by = 'movieId') %>%
  left_join(um_avg, by = 'userId') %>%
  left_join(umg_avg, by = 'genres') %>%
  left_join(umgy_avg, by = 'year') %>%
  mutate(pred_umgy = ovrl_avg + movie_i + user_i + genre_i +                          year_i) %>%
  mutate(pred_umgy_adj = ifelse(pred_umgy > 5, 5.0, 
                                ifelse(pred_umgy < 0.5, 0.5, pred_umgy))) %>%
  pull(pred_umgy_adj)

umgy_RMSE_adj = RMSE(test_edx$rating, umgy_pred)
umgy_RMSE_adj #0.8653532

################################
# Regularization Methods
################################

#Regularized user and movie effects model
lambdas <- seq(0, 10, 0.5) #try lambdas between 0 and 10 (0.5 increments)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_edx$rating)
  
  b_i <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred_um_reg = mu + b_i + b_u) %>%
    mutate(pred_um_reg_adj = ifelse(pred_um_reg > 5, 5.0, 
                                    ifelse(pred_um_reg < 0.5, 0.5, 
                                           pred_um_reg))) %>% 
    pull(pred_um_reg_adj)
  
  return(RMSE(predicted_ratings, test_edx$rating))
})

qplot(lambdas, rmses) #plot to show the lambdas against RMSE values
min_lambda_um_reg = lambdas[which.min(rmses)] 
min_lambda_um_reg #lambda = 4.5 provides lowest RMSE
min_rmse_um_reg = min(rmses)
min_rmse_um_reg #0.8651262


#Regularized user, movie, genre, and year effects model
lambdas <- seq(0, 10, 0.5)

rmses_umgy <- sapply(lambdas, function(l){
  
  mu <- mean(train_edx$rating)
  
  b_i <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- train_edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_y <- train_edx %>%
    mutate(year = year(as.Date.POSIXct(timestamp))) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  
  predicted_ratings <- test_edx %>% 
    mutate(year = year(as.Date.POSIXct(timestamp))) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred_umgy_reg = mu + b_i + b_u + b_g + b_y) %>%
    mutate(pred_umgy_reg_adj = ifelse(pred_umgy_reg > 5, 5.0, 
                                      ifelse(pred_umgy_reg < 0.5, 0.5, 
                                             pred_umgy_reg))) %>%
    pull(pred_umgy_reg_adj)
  
  return(RMSE(predicted_ratings, test_edx$rating))
})

min_lambda_umgy_reg = lambdas[which.min(rmses_umgy)] 
min_lambda_umgy_reg #lambda = 4.5 provides lowest RMSE
min_rmse_umgy_reg = min(rmses_umgy)
min_rmse_umgy_reg #0.8647711


################################
# Final Results
################################

#Use the regularized user, movie, genre, and year model
#Build the model using edx set. Test the model on the validation set.
lambda <- 4.5 #Selected from model in previous section

rmses_umgy_edx_fn <- function(l){
  # This function constructs the regularized model on the edx set 
  # given input lambda (l) and calculates RMSE on the validation set
  
  mu <- mean(edx$rating) #overall average rating
  
  #effect of movie i
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #effect of user u
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #effect of genre g
  b_g <- train_edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  #effect of year y
  b_y <- edx %>%
    mutate(year = year(as.Date.POSIXct(timestamp))) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  
  #predict the ratings for the validation set
  predicted_ratings <- validation %>% 
    mutate(year = year(as.Date.POSIXct(timestamp))) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred_umgy_reg = mu + b_i + b_u + b_g + b_y) %>%
    mutate(pred_umgy_reg_adj = ifelse(pred_umgy_reg > 5, 5.0, 
                                      ifelse(pred_umgy_reg < 0.5, 0.5, 
                                             pred_umgy_reg))) %>%
    pull(pred_umgy_reg_adj)
  
  ratings_and_RMSE_list = list(head(predicted_ratings, 50), 
                               RMSE(predicted_ratings, validation$rating))
  return(ratings_and_RMSE_list)
}

#Return the ratings and RMSE on the validation set using lambda = 4.5
rmses_umgy_edx_fn(lambda) #0.8642996
