library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(kernlab)


# Read in Data and prep export function ----------------------------------

train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)


predict_export <- function(workflowName, fileName){
  preds <- workflowName %>%
    predict(new_data = test, type="prob")
  
  submission <- preds %>% 
    mutate(id=row_number()) %>% 
    rename(Action = .pred_1) %>% 
    select(3,2)
  
  directory = "./submissions/"
  path = paste0(directory,fileName)
  
  vroom_write(submission, file = path, delim=',')
}

# Recipe ------------------------------------------------------------------

recipe.pca <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.85) 

num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
doParallel::registerDoParallel(cl)

prepped.pca <- prep(recipe.pca)
bakedSetPCA <- bake(prepped.pca, new_data = train)

stopCluster(cl)


# Build Model and Workflow -----------------------------------------------

svmLinear <- svm_linear(cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svmWF <- workflow() %>% 
  add_recipe(recipe.pca) %>% 
  add_model(svmLinear)

# Cross Validation --------------------------------------------------------

## Create tuning grid and folds
svmGrid <- grid_regular(cost(),levels = 10)
svmFolds <- vfold_cv(train, v=5, repeats=1)

## Run cross validation
svmCVstart <- proc.time()
svmCVresults <- svmWF %>%
  tune_grid(resamples=svmFolds,
            grid=svmGrid,
            metrics=metric_set(roc_auc))
proc.time()-svmCVstart


# Fitting the best model --------------------------------------------------

## Select the best tuned model
svmBestTune <- svmCVresults %>% 
  select_best("roc_auc")

# finalize workflow and fit
svmFinal <- svmWF %>% 
  finalize_workflow(svmBestTune) %>% 
  fit(train)


# Export Data -------------------------------------------------------------
predict_export(svmFinal,"svmLinear.csv")

