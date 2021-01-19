from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    calories_df = pd.read_csv('../input/calories.csv')
    exercise_df = pd.read_csv('../input/exercise.csv')
    df = pd.merge(calories_df, exercise_df, on='User_ID')
    df.drop('User_ID', axis=1, inplace=True)

    # Create a new column for fold information
    df['kfold'] = -1

    # Randomize the rows of the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize the kfold class
    kf = KFold(n_splits=5)

    # Fill the kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    # Save the new Csv with fold information
    df.to_csv('../input/train_folds.csv', index=False)
