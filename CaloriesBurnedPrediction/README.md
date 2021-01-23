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

+ __Metrics__
  
  COLUMNS | MODEL | RMSE | R2
  --- | --- | --- | ---
  ['Duration'] | Linear regression | 18.5 | 0.91
  ['Duration'] | Lasso regression | 18.5 | 0.91
  ['Duration'] | Decision tree | 16.6 | 0.92
  ['Duration'] | Random forest | 16.6 | 0.92
  ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Linear regression | 11.6 | 0.966
  ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Lasso regression | 12.1 | 0.963
  ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Decision tree | 7.3 | 0.986
  ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Random forest | 4.8 | 0.994
  ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Linear regression | 11.6 | 0.966
  ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Lasso regression | 12.1 | 0.963
  ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Decision tree | 5.6 | 0.992
  ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'] | Random forest | 3.0 | 0.998
