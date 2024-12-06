library(tidyverse)
library(tidymodels)
library(embed)
library(dbarts)
library(vroom)

train <- vroom::vroom("data/train.csv")
test <- vroom::vroom("data/test.csv")

train$ACTION <- as.factor(train$ACTION)

# Make function to predict and export
predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  
  preds <- workflowName %>%
    predict(new_data = test, type="prob")
  
  submission <- preds %>% 
    mutate(id=row_number()) %>% 
    rename(Action = .pred_1) %>% 
    select(3,2)
  
  directory = "./TA_submissions/"
  path = paste0(directory,fileName)
  
  vroom_write(submission, file = path, delim=',')
  
  
}

# Recipe
recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
 # step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Model

bartModel <- parsnip::bart(mode = "classification",
                           engine='dbarts',trees = tune())

bartWF <- workflow() %>% 
  add_recipe(recipe) %>%
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
  select_best(metric = "roc_auc")

#   finalize the Workflow & fit it
bartFinalWF <- bartWF %>%
  finalize_workflow(bartBestTune) %>%
  fit(data=train)

## ADD TUNING GRID ?? 
# metrics=metric_set(roc_auc)

predict_export(bartFinalWF,"bart_1.csv")
