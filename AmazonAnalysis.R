library(tidymodels)
library(tidyverse)
library(vroom)
# library(ggmosaic)
library(glmnet)
library(parallel)
library(embed) #used for target encoding
library(discrim) # for naive bayes engine
library(naivebayes) # naive bayes
library(kknn)
library(kernlab) #for svm

#workingdirectory <- getwd()
#setwd(workingdirectory)

train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)



# Recipes -----------------------------------------------------------------
# recipe
recipe <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())
# step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))

prepped <- prep(recipe)
bakedSet <- bake(prepped, new_data = train)

# make a new recipe to do target encoding
recipe.t <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())


## Parallel
doParallel::registerDoParallel(4)
## Prep and Bake
prepped.t <- prep(recipe.t)
bakedSet <- bake(prepped.t, new_data = train)

# Make a new recipe that uses PCA

recipe.pca <- recipe(ACTION ~ ., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.85) # threshold is kind of like the R^2 of the components

## Parallel to prep and bake recipe
doParallel::registerDoParallel(4)
prepped.pca <- prep(recipe.pca)
bakedSetPCA <- bake(prepped.pca, new_data = train)
# with threshold=0.85 we have 50 variables which seems like a good amount

# EDA
# ggplot(data=train) + geom_mosaic(aes(x=product(MGR_ID), fill=ACTION))

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

# Logistic Regression -----------------------------------------------------
logisticMod <- logistic_reg() %>% #Type of mode
  set_engine("glm")

logistic_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(logisticMod) %>%
  fit(data = train) # Fit the workflow

amazon_predictions <- predict(logistic_workflow,
                              new_data=test,
                              type="prob") # "class" or "prob" (see doc) %>% 

submission <- amazon_predictions %>% 
            mutate(id=row_number()) %>% 
              rename(Action = .pred_1) %>% 
              select(3,2)
vroom_write(submission, file = "submission1.csv", delim=',')



# Penalized Logistic Regression -------------------------------------------
## Parallel
doParallel::registerDoParallel(4)

penalizedMod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penalizedWF <- workflow() %>%
  add_recipe(recipe.t) %>%
  add_model(penalizedMod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 15) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 15, repeats=1)


## Parallel
doParallel::registerDoParallel(4)

## Run the CV ~ about 3 minutes
penalizedCV_results <- penalizedWF %>%
    tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- penalizedCV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  penalizedWF %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
predict_export(final_wf,"penalized2.csv")




# Decision Trees and Forests ----------------------------------------------

# still to do



# Naive Bayes -------------------------------------------------------------
naiveModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

naiveWF <- workflow() %>%
  add_recipe(recipe.t) %>%
  add_model(naiveModel)

## Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Parallel
doParallel::registerDoParallel(4)

## Run the CV ~ about 2.5 minutes
naiveBayes_CV_results <- naiveWF %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
naiveBestTune <- naiveBayes_CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
naiveFinalWF <-
  naiveWF %>%
  finalize_workflow(naiveBestTune) %>%
  fit(data=train)

## Predict
predict_export(naiveFinalWF,"naiveBayes.csv")


# KNN ---------------------------------------------------------------------

## Set up model
knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(recipe.t) %>%
  add_model(knn_model)

## set up grid of tuning values
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 10)

## set up k-fold CV
knn_folds <- vfold_cv(train, v = 10, repeats=1)

## Parallel
doParallel::registerDoParallel(4)

## Run the CV - about 3.5 minutes
knn_CVresults <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc)) 

## find best tuning parameters
KNNbestTune <- knn_CVresults %>%
  select_best("roc_auc")

## Finalize workflow  
finalKNN_wf <- knn_wf %>%
  finalize_workflow(KNNbestTune) %>%
  fit(data=train)

## Predict
predict_export(finalKNN_wf,"knn.csv")

# PCA ---------------------------------------------------------------------

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
doParallel::registerDoParallel(4)

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


# SVM ---------------------------------------------------------------------
# Build models - Running Radial Only

# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

# Create Workflow
svmRadialWF <- workflow() %>% 
  add_recipe(recipe.pca) %>% 
  add_model(svmRadial)

# Create tuning grid and folds
svmRadialGrid <- grid_regular(rbf_sigma(),
                             cost(),
                             levels = 10)
svmFolds <- vfold_cv(train, v=5,repeats=1)

doParallel::registerDoParallel(4)
# Run the CV ~ about  minutes
svmCVstart <- proc.time()
svmRadial_CVresults <- svmRadialWF %>%
  tune_grid(resamples=svmFolds,
            grid=svmRadialGrid,
            metrics=metric_set(roc_auc))
proc.time()-svmCVstart

# select the best tune
svmRadialBestTune <- svmRadial_CVresults %>% 
  select_best("roc_auc")

# finalize workflow and fit
svmRadiaFinal <- svmRadialWF %>% 
  finalize_workflow(svmRadialBestTune) %>% 
  fit(train)

# 






