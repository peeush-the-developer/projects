from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

import config

if __name__ == '__main__':
    # Load training file into dataframe
    df = pd.read_csv(config.TRAINING_FILE)

    # Create new column 'kfold'
    df['kfold'] = -1

    # Randomize the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize the kfold instance
    kf = KFold(n_splits=5)

    # Assign kfold column fold value
    for fold_, (train_, val_) in enumerate(kf.split(df)):
        df.loc[val_, 'kfold'] = fold_
    
    # save the dataframe to new file 'train_folds.csv
    df.to_csv(config.TRAINING_FOLDS_FILE, index=False)
