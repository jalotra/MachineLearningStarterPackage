from sklearn import ensemble


# Define the models here 
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=20, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=20, n_jobs=-1, verbose=2),
}