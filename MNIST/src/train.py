import pandas as pd
import os
import joblib
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score

import config
import model_dispatcher

def run(fold, model_name):
    # Load the data from CSV file (with fold)
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into train and val
    train = df.loc[df['kfold'] != fold, :]
    val = df.loc[df['kfold'] == fold, :]

    # Get features and label out of train/val sets
    x_train = train.drop(['label', 'kfold'], axis=1).values
    y_train = train.label.values

    x_val = val.drop(['label', 'kfold'], axis=1).values
    y_val = val.label.values

    # Initialize model
    model = model_dispatcher.MODELS[model_name]

    # Fit the model and make predictions
    model.fit(x_train, y_train)
    preds = model.predict(x_val)

    # Evaluate the model using accuracy score
    acc = accuracy_score(y_val, preds)

    print(f'Model={model_name}, Fold={fold}, Acc={acc:.3f}')

    # Save the model
    joblib.dump(
        model, 
        os.path.join(config.MODELS_OUTPUT_DIR, f'{model_name}_{fold}_{acc:.3f}.bin'))

if __name__ == '__main__':
    # Initialize the Argument parser
    ap = ArgumentParser()

    # Add arguments to be read from CLI
    ap.add_argument('--fold', type=int, required=True, help='Fold value to train the model')
    ap.add_argument('--model', type=str, required=True, help='Model name to train the model')
    args = ap.parse_args()

    # Run the training with provided arguments
    run(args.fold, args.model)
