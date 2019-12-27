export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

export MODEL=$1

# TO BE USED WHENEVER I AM GOING TO TRAIN THE CLASSIFIER 
FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train



# TO BE USED WHENEVER I AM PREDICTING 
# python -m src.predict