#Ignore Warnings
import warnings
warnings.filterwarnings(action='ignore')

'''Import Scripts'''
from etl import *

'''Data Tools'''
import numpy as np
import pandas as pd

'''For loading models'''
from joblib import load
import pickle

RANDOM_SEED = 420420

def prepare_data(use_pca=False):
    '''
    Function to prepare data for evaluation
    If PCA was implemented in training set use_pca = True.
    '''
    #Set Paths
    EVAL_PATH = "/data"

    #Prep Data
    print("Preparing Data")
    
    #Read In Feature List
    feature_id_map_eval =  load('/model/feature_dict.pickle')
    
    #Create feature data frame
    df_eval_set = create_feature_df(feature_id_map_eval, path=EVAL_PATH)

    #Join gold standard
    gs = pd.read_csv(EVAL_PATH + "/goldstandard.csv")
    df_gold_standard = gs.set_index('person_id')
    df_merged_eval_set = df_eval_set.join(df_gold_standard)

    #Initalize Data Sets
    X_set =  df_merged_eval_set.loc[:, df_merged_eval_set .columns != 'status']
    Y_set = df_merged_eval_set.status

    #Load PCA for transformation
    if use_pca:
        pca =  load('/model/pca.pickle')
        X_set = pca.transform(X_set)

    X_set = np.array(X_set)
    Y_set = np.array(Y_set)

    return X_set

def prediction(X_test):
    clf =  load('/model/baseline.joblib')
    Y_pred = clf.predict_proba(X_test)[:,1]
    output_prob = pd.DataFrame(Y_pred,columns = ['score'])
    output_prob.index.name ='person_id'
    output_prob.reset_index(inplace=True)
    output_prob = output_prob.fillna(0)
    output_prob.to_csv('/output/predictions.csv', index = False)
    print("Inferring stage finished", flush = True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Performs inference and generates predictions to: /output/predictions.csv')
    parser.add_argument('--use_pca', type=bool, required=False, help='Set to True to expect PCA before model training')
    args_dict = vars(parser.parse_args())
    use_pca = args_dict['use_pca']

    X = prepare_data(use_pca=use_pca)
    prediction(X)