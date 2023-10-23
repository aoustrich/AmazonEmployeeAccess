library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(parallel)
library(embed) #used for target encoding
library(discrim) # for naive bayes engine
library(naivebayes) # naive bayes
library(kknn)

train <- vroom("train.csv")
test <- vroom("test.csv")

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
  
  directory = "./submissions/"
  path = paste0(directory,fileName)
  
  vroom_write(submission, file = path, delim=',')
  
}

num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

recipe.pca <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.95) # threshold is kind of like the R^2 of the components

## Parallel to prep and bake recipe

prepped.pca <- prep(recipe.pca)
bakedSetPCA <- bake(prepped.pca, new_data = train)


#### Re run the Naive Bayes model with the new recipe
naiveModelPCA <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

naivePCA_WF <- workflow() %>%
  add_recipe(recipe.pca) %>%
  add_model(naiveModelPCA)

## Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Parallel


## Run the CV ~ about 3.5 minutes
naiveBayes_CV_resultsPCA <- naivePCA_WF %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
naiveBestTunePCA <- naiveBayes_CV_resultsPCA %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
naiveFinalWFPCA <-
  naivePCA_WF %>%
  finalize_workflow(naiveBestTunePCA) %>%
  fit(data=train)

## Predict
predict_export(naiveFinalWFPCA,"naiveBayesPCA.csv")


stopCluster(cl)