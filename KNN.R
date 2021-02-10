# Helper packages
install.packages("dplyr")
install.packages("ggplot2")
install.packages("rsample")
install.packages("recipes")
install.packages("caret")
install.packages("AmesHousing")
install.packages("modeldata")
install.packages("purrr")
install.packages("e1071")
install.packages("tibble")
install.packages("stringr")


# Helper packages
library(dplyr) # for data wrangling
library(ggplot2) # for awesome graphics
library(rsample) # for data splitting
library(recipes) # for feature engineering
# Modeling packages
library(caret) # for training KNN models
library(AmesHousing)
library(modeldata)
library(purrr)
library(e1071)
library(tibble)
library(stringr)


ames <- AmesHousing::make_ames()
# Print dimensions
dim(ames)
## [1] 2930 81
# Peek at response variable
head(ames$Sale_Price)

set.seed(123)
split <- initial_split(ames, prop = 0.7,
                       strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)
# Create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))
# Tune a knn model using grid search
knn_fit <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

knn_fit
ggplot(knn_fit)
attrition <- rsample::attrition
data("attrition")

attrit <- attrition %>%
  mutate_if(is.ordered, factor, ordered = FALSE)
set.seed(123) # for reproducibility
churn_split <- initial_split(attrit, prop = 0.7,
                             strata = "Attrition")
churn_train <- training(churn_split)
# Import MNIST training data
mnist <- dslabs::read_mnist()
names(mnist)

two_houses <- ames_train %>%
  select(Gr_Liv_Area, Year_Built) %>%
  sample_n(2)

two_houses
dist(two_houses, method = "euclidean")
dist(two_houses, method = "manhattan")

home1 <- ames_train[ames_train$Bedroom_AbvGr == 4 & ames_train$Year_Built == 2008, ]
home1 <- home1[1:1, c("Bedroom_AbvGr", "Year_Built")]
home2 <- ames_train[ames_train$Bedroom_AbvGr == 2 & ames_train$Year_Built == 2008, ]
home2 <- home2[1:1, c("Bedroom_AbvGr", "Year_Built")]
home3 <- ames_train[ames_train$Bedroom_AbvGr == 3 & ames_train$Year_Built == 1998, ]
home3 <- home3[1:1, c("Bedroom_AbvGr", "Year_Built")]
home1
home2
home3
features <- c("Bedroom_AbvGr", "Year_Built")
dist(rbind(home1[,features], home2[,features]))
dist(rbind(home1[,features], home3[,features]))



ames <- AmesHousing::make_ames()

ames_std <- as.data.frame(ames%>%
            select_if(is.numeric) %>% # sadece numerik kolonlarý seç
            scale()) # standartize et

home1 <- ames[423,c("Bedroom_AbvGr", "Year_Built")]
home1_std <- ames_std[423,c("Bedroom_AbvGr", "Year_Built")]

home2 <- ames[424,c("Bedroom_AbvGr", "Year_Built")]
home2_std <- ames_std[424,c("Bedroom_AbvGr", "Year_Built")]

home3 <- ames[6,c("Bedroom_AbvGr", "Year_Built")]
home3_std <- ames_std[6,c("Bedroom_AbvGr", "Year_Built")]
home1_std
home2_std
home3_std



dist(rbind(home1_std[,features], home2_std[,features]))

dist(rbind(home1_std[,features], home3_std[,features]))


blueprint <- recipe(Attrition ~ ., data = churn_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(contains("Satisfaction")) %>%
  step_integer(WorkLifeBalance) %>%
  step_integer(JobInvolvement) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())  

# Create a resampling method
cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

hyper_grid <- expand.grid(
  k = floor(seq(1, nrow(churn_train)/3, length.out = 20))
)

knn_grid <- train(
  blueprint,
  data = churn_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "ROC"
)
ggplot(knn_grid)

set.seed(123)
index <- sample(nrow(mnist$train$images), size = 10000)
mnist_x <- mnist$train$images[index, ]
mnist_y <- factor(mnist$train$labels[index])


mnist_x %>%
  as.data.frame() %>%
  map_df(sd) %>%
  gather(feature, sd) %>%
  ggplot(aes(sd)) +
  geom_histogram(binwidth = 1)


# Rename features
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))
# Remove near zero variance features manually
nzv <- nearZeroVar(mnist_x)
index <- setdiff(1:ncol(mnist_x), nzv)
mnist_x <- mnist_x[, index]


cv <- trainControl(
  method = "LGOCV",
  p = 0.7,
  number = 1,
  savePredictions = TRUE
)
# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))
# Execute grid search
knn_mnist <- train(
  mnist_x,
  mnist_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)
ggplot(knn_mnist)

cm <- confusionMatrix(knn_mnist$pred$pred, knn_mnist$pred$obs)
cm$byClass[, c(1:2, 11)] # sensitivity, specificity, & accuracy


vi <- varImp(knn_mnist)
vi


imp <- vi$importance %>%
  rownames_to_column(var = 'feature') %>%
  gather(response, imp, -feature) %>%
  group_by(feature) %>%
  summarize(imp = median(imp))

edges <- tibble(
  feature = paste0("V", nzv),
  imp = 0
)
# Combine and plot
imp <- rbind(imp, edges) %>%
  mutate(ID = as.numeric(str_extract(feature, "\\d+"))) %>%
  arrange(ID)
image(matrix(imp$imp, 28, 28), col = gray(seq(0, 1, 0.05)),
      xaxt="n", yaxt="n")

set.seed(9)
good <- knn_mnist$pred %>%
  filter(pred == obs) %>%
  sample_n(4)
# Get a few inaccurate predictions
set.seed(9)
bad <- knn_mnist$pred %>%
  filter(pred != obs) %>%
  sample_n(4)
combine <- bind_rows(good, bad)

set.seed(123)
index <- sample(nrow(mnist$train$images), 10000)
X <- mnist$train$images[index,]
# Plot results
par(mfrow = c(4, 2), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(combine)), 4, 2, byrow = FALSE))
for(i in seq_len(nrow(combine))) {
  image(matrix(X[combine$rowIndex[i],], 28, 28)[, 28:1],
        col = gray(seq(0, 1, 0.05)),
        main = paste("Actual:", combine$obs[i], " ",
                      "Predicted:", combine$pred[i]),
        xaxt="n", yaxt="n")
}