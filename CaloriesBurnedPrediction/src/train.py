import joblib
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

import config
import model_dispatcher

def run(fold, model_name, use_duration = False):
    # Read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    if not use_duration:
        # 'Gender' column: Since it has 2 values only, we can use LabelEncoder that converts categories 
        # into numeric by mapping 'Male':0, 'Female':1 or vice-versa
        lbl_enc = LabelEncoder()
        df.loc[:, 'Gender'] = lbl_enc.fit_transform(df.Gender.values)

    # Split the data into train and validation
    # Training data is where kfold != fold value. Reset the index too
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    # Validation data is where kfold == fold value. Reset the index too
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    # Initialize the model
    model = model_dispatcher.models[model_name]

    if not use_duration:
        # As we know from our 'explore_data.ipynb' that only 'Duration', 'Heart_Rate' and 'Body_Temp' columns are relevant.
        # Let's take all columns to be able to get important features from SelectFromModel
        x_train = df_train.drop(['Calories','kfold'], axis=1).values
        y_train = df_train.Calories.values

        # Col_names
        col_names = df_train.drop(['Calories','kfold'], axis=1).columns.tolist()

        # Similarly for validation
        x_val = df_val.drop(['Calories','kfold'], axis=1).values
        y_val = df_val.Calories.values
        
        # Select from the model
        sfm = SelectFromModel(estimator=model)
        x_train_transformed = sfm.fit_transform(x_train, y_train)
        x_val_transformed = sfm.transform(x_val)

        # See which features were selected
        support = sfm.get_support()
        sel_col_names = [x for x, y in zip(col_names, support) if y==True]
        if len(sel_col_names) == 1 and sel_col_names[0] == 'Duration':
            pass
        else:
            print(sel_col_names)

        # Fit the model on training data
        model.fit(x_train_transformed, y_train)

        # Predict on validation data
        preds = model.predict(x_val_transformed)
    else:
        x_train = df_train[['Duration']].values
        y_train = df_train.Calories.values

        x_val = df_val[['Duration']].values
        y_val = df_val.Calories.values
        
        # Fit the model on training data
        model.fit(x_train, y_train)

        # Predict on validation data
        preds = model.predict(x_val)

    # Calculate the RMSE for the model
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    # Calculate the R2 score for the model
    r2 = r2_score(y_val, preds)
    
    print(f'Model={model_name}, Fold={fold} => RMSE={rmse:.3f}, R2={r2:.3f}')

    # save the model
    joblib.dump(
        model, 
        os.path.join(config.MODEL_OUTPUT, f'{model_name}_{fold}.bin'))

if __name__ == "__main__":

    # Initialize the argument parser
    ap = ArgumentParser()

    # Add arguments that we expect to parse from CLI
    ap.add_argument('--fold', type=int, required=True, help='Fold value to run the training script')
    ap.add_argument('--model', type=str, required=True, help='Model name to run the training script')

    # Parse the arguments received
    args = ap.parse_args()

    run(fold=args.fold, model_name=args.model, use_duration=args.model == 'lr')