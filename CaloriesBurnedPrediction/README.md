# Calories Burned Prediction with Python

Machine learning project on Calories burned prediction with Python.

## Description

+ `src/create_folds.py`
  + Creates and saves the CSV file with fold information.
  + Technique used for Cross-validation
+ `notebooks/explore_data.ipynb`
  + Explore the data in the dataset. We found that 'Duration', 'Heart_Rate', 'Body_Temp' are relevant columns.
+ `src/train.py`
  + Train the model and calculate the RMSE (Root Mean Squared Error) for the model to evaluate and compare between different models.
+ `src/model_dispatcher.py`
  + Defines the models which can be used to train and evaluate.
  + If new model needs to be added, this is the only place to be changed.
+ `src/config.py`
  + Defines the variables like `Training_file`, `Model_output_dir`, etc.
+ `src/run.sh`
  + Shell script to run the model for each fold together.

## Conclusion

+ `Duration` column is enough to train and evaluate the model
+ __RMSE__
  + `Linear Regression` = 18.4
  + `Decision Tree` = 16.6
  + `Random Forest` = 16.6