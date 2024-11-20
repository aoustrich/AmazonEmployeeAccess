library(tidyverse)
library(tidyverse)
library(embed)
library(bonsai) # boosted trees & bart
library(lightgbm) # boosted trees

train <- vroom::vroom("data/train.csv")
test <- vroom::vroom("data/test.csv")

train$ACTION <- as.factor(train$ACTION)

# Make function to get predictions, prep for kaggle, and export the data
predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,new_data=test) %>%
    bind_cols(., test) %>% 
    select(datetime, .pred) %>% 
    rename(count=.pred) %>% 
    mutate(
      count=exp(count),
      count=pmax(1, count)
    ) %>% 
    mutate(datetime=as.character(format(datetime)))
  
  vroom::vroom_write(x, file=fileName,delim=',')
}

# Recipe
recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())


# Boosted Trees -----------------------------------------------------------
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boostWF <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(recipe)

#   tuning grid
boostGrid <- grid_regular(tree_depth(),
                          trees(),
                          learn_rate(),
                          levels = 20)
# folds
boostFolds <- vfold_cv(train, v=10, repeats=1)


boostResultsCV <- boostWF %>% 
  tune_grid(resamples=boostFolds,
            grid=boostGrid,
            metrics=metric_set(roc_auc))

#   find best tune
boostBestTune <- boostResultsCV %>%
  select_best("roc_auc")

#   finalize the Workflow & fit it
boostFinalWF <- boostWF %>%
  finalize_workflow(boostBestTune) %>%
  fit(data=train)

#   predict and export
predict_export(boostFinalWF,"lightGBM_1.csv")