import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import config

if __name__ == '__main__':
    train = pd.read_csv(config.TRAINING_FILE_RAW)

    # Add kfold column in the dataframe
    train['kfold'] = -1

    # reshuffle the data
    train = train.sample(frac=1).reset_index(drop=True)

    # Initialize the KFold
    kf = KFold(n_splits = 5)

    # Assign fold value in kfold column
    for fold_, (train_, val_) in enumerate(kf.split(X=train)):
        train.loc[val_, 'kfold'] = fold_
    
    train.to_csv(config.TRAINING_FILE, index=False)