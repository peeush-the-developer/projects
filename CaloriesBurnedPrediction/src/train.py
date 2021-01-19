import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

import os
import config

def run(fold):
    # Read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into train and validation
    # Training data is where kfold != fold value. Reset the index too
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    # Validation data is where kfold == fold value. Reset the index too
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    # As we know from our 'explore_data.ipynb' that only 'Duration', 'Heart_Rate' and 'Body_Temp' columns are relevant.
    # For simplicity, we start with 'Duration' column to predict 'Calories'
    x_train = df_train[['Duration']].values
    y_train = df_train.Calories.values

    # Similarly for validation
    x_val = df_val[['Duration']].values
    y_val = df_val.Calories.values

    # Initialize LinearRegression model
    lr = LinearRegression()

    # Fit the model on training data
    lr.fit(x_train, y_train)

    # Predict on validation data
    preds = lr.predict(x_val)

    # Calculate the RMSE for the model
    rmse = np.sqrt(mean_absolute_error(y_val, preds))
    print(f'Fold={fold}, RMSE={rmse:3f}')

    # save the model
    joblib.dump(
        lr, 
        os.path.join(config.MODEL_OUTPUT, f'lr_{fold}.bin'))

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)