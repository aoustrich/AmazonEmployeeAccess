library(tidyverse)
library(tidyverse)
library(embed)
library(dbarts)

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

# Model

bartModel <- parsnip::bart(mode = "regression",
                           engine='dbarts',trees = tune())

bartWF <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(bartModel) 

#   tuning grid
grid <- grid_regular(trees(),
                          levels = 20)
# folds
folds <- vfold_cv(train, v=10, repeats=1)


bartResultsCV <- bartWF %>% 
  tune_grid(resamples=folds,
            grid=grid,
            metrics=metric_set(roc_auc))

#   find best tune
bartBestTune <- bartResultsCV %>%
  select_best("roc_auc")

#   finalize the Workflow & fit it
bartFinalWF <- bartWF %>%
  finalize_workflow(bartBestTune) %>%
  fit(data=train)

## ADD TUNING GRID ?? 
# metrics=metric_set(roc_auc)

predict_export(bartFinalWF,"bart_1.csv")