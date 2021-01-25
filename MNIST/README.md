# Digit recognizer project with Python

In this project, we take pixel values of images of size 28 x 28 pixels with label that is an actual digit of the image in a CSV format.

Prepare different models to take input of 784 features and categorize the input into one of the 10 digits _(0 - 9)_.

## Description

+ `src/create_folds.py`
  + Creates and saves the CSV file with fold information.
  + Technique used for Cross-validation
+ `src/train.py`
  + Train the model and calculate the __Accuracy score__ for the model to evaluate and compare between different models.
+ `src/model_dispatcher.py`
  + Defines the models which can be used to train and evaluate.
  + If new model needs to be added, this is the only place to be changed.
+ `src/config.py`
  + Defines the variables like `Training_file`, `Model_output_dir`, etc.
+ `src/run.sh`
  + Shell script to run the model for each fold together.

## Conclusion

+ __Metrics__
  
  MODEL | Accuracy (%)
  --- | ---
  Decision tree (Gini) | 84.7
  Decision tree (Entropy) | 86
  Random forest | 96.2
