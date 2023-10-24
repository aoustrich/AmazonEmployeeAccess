library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(parallel)
library(embed)
library(ranger)
library(doParallel)
library(rstanarm)

startTime <- proc.time()

train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)

recipe.b <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_bayes(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 20
cl <- makePSOCKcluster(num_cores)
doParallel::registerDoParallel(cl)


## Prep and Bake
prepStart <- proc.time()
prepped.b <- prep(recipe.b)
bakedSetB <- bake(prepped.b, new_data = train)

print("Time to bake:")
proc.time()-prepStart

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


# Classification Forest ---------------------------------------------------
randForestModel <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees=500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

forestWF <- workflow() %>% 
  add_recipe(recipe.b) %>%
  add_model(randForestModel)

# create tuning grid
forest_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(bakedSetB)-1))),
                                   min_n(),
                                   levels = 100)

# split data for cross validation
rfolds <- vfold_cv(train, v = , repeats=1)

cvStart <- proc.time()

# run cross validation
treeCVResults <- forestWF %>% 
  tune_grid(resamples = rfolds,
            grid = forest_tuning_grid,
            metrics=metric_set(roc_auc)) 

print("CV time:")
proc.time()-cvStart

# select best model
best_tuneForest <- treeCVResults %>% 
  select_best("roc_auc")

# finalize workflow
finalForestWF <- 
  forestWF %>% 
  finalize_workflow(best_tuneForest) %>% 
  fit(data=train)

# predict and export
predict_export(finalForestWF,"ClassificationForest.csv")

stopCluster(cl)

print("total time: ")
proc.time()-startTime
