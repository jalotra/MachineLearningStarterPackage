import pandas as pd
import argparse 
from sklearn import model_selection


def folds_creator(train_file_name, random_state):
    df = pd.read_csv("input/" + str(train_file_name))
    # Create a new column in the input data matrix 
    # named kfolds 
    df["kfold"] = -1

    # Now shuffle the data presnt 
    df = df.sample(frac = 1) 
    # Reset the indexes and drop the index
    df = df.reset_index(drop= True)

    # Now create the folds object
    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)

    # NOW CREATE THE FOLDS 
    for fold, (train_idx, val_idx) in enumerate(kf.split(X= df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold 
    
    # Now save the new Kfolded Df
    df.to_csv("input/train_folds.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action = "store",type = str, dest = "filename")
    args = parser.parse_args()
    folds_creator(args.filename, random_state = 40)






 
