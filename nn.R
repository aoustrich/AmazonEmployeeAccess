library(tidyverse)
library(tidyverse)
# library(embed)


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
# recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())

nn_recipe <- recipe(ACTION ~ ., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% ## Turn color to factor then dummy encode color
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 100, #or 100 or 250
                activation="relu") %>%
  set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")
########### Keras: https://stackoverflow.com/questions/44611325/r-keras-package-error-python-module-tensorflow-contrib-keras-python-keras-was-n
###########  did all but step 4 from the answer install_github


# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50) %>%  #or 100 or 250
#                 # activation="relu") %>%
#   set_engine("nnet") %>% #verbose = 0 prints off less
#   set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(),
                            levels=10)

nn_folds <- vfold_cv(train, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
  tune_grid(grid=nn_tuneGrid,
            resamples=nn_folds,
            metrics=metric_set(roc_auc))

#   find best tune
nn_bestTune <- tuned_nn %>%
  select_best("roc_auc")

#   finalize the Workflow & fit it
final_nn_wf <- nn_wf %>%
  finalize_workflow(nn_bestTune) %>%
  fit(data=train)

#   predict and export
predict_export(final_nn_wf,"nn_Keras.csv")
