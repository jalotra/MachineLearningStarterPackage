import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

# FOLD MAPPING 
FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    # Create the TRAINING AND TESTING DATAFRAMES
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    # Get the train df if and only if it is present in the Fold mapping  
    # Like if FOLD = 2
    # Then I will have the testing data folds 0,1,3,4  
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)

    # The validation df is used to collect the values only for FOLD
    # Like FOLD == 2 
    # Then its values will act as the testing data for rest of  Folds the classifier is trained on  
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    # ytrain and yvalid contains actual labels 
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    # drop the id, target and kfold as I AM TRAINING FELLAS 
    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    # vALID_df must contain all the columns in the TRAIN_df  
    valid_df = valid_df[train_df.columns]

    # NOW I HAVE TO LABEL ENCODE THE SIMILAR VALUES IN SAME COLUMNS 
    # like if the column1 usually called the datapoint1 
    # like list = [paris, paris, india, spain]
    # Then the le = preprocessing.LabelEncoder().transform(list)
    # Will return array([0, 0, 1, 2])
    label_encoders = {}
    for c in train_df.columns:
        # Create the lbl object of the LabelEncoder CLass
        lbl = preprocessing.LabelEncoder()
        # For the column c lets encode all the data in train_df and valid_df and df_test
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        # Change the labels to encoded labels
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())

        # For each column save the encoded labels in the dict label_encoders 
        label_encoders[c] = lbl
    
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))


    # DUMPING EVERYTHING 
    # DUMP THE label_encoders
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    # DUMP the classifier itself
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    # DUMP the train_df all columns 
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
