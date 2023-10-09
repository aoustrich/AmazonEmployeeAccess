library(tidymodels)
library(tidyverse)
library(vroom)
library(ggmosaic)

train <- vroom("./train.csv")
test <- vroom("./test.csv")

train$ACTION <- as.factor(train$ACTION)

# recipe
recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

prepped <- prep(recipe)
bakedSet <- bake(prepped, new_data = train)

# EDA
# ggplot(data=train) + geom_mosaic(aes(x=product(MGR_ID), fill=ACTION))

# Make function to predict and export
predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,new_data=test)  %>%
    mutate(id=row_number()) %>% 
    rename(Action = .pred_1) %>% 
    select(3,2)
  vroom_write(x, file=fileName,delim=',')
}

# Logistic Regression -----------------------------------------------------

my_mod <- logistic_reg() %>% #Type of mode
  set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(recipe) %>%
add_model(my_mod) %>%
fit(data = train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=test,
                              type="prob") # "class" or "prob" (see doc) %>% 

submission <- amazon_predictions %>% 
            mutate(id=row_number()) %>% 
              rename(Action = .pred_1) %>% 
              select(3,2)
vroom_write(submission, file = "submission1.csv", delim=',')














