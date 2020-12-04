#Ignore Warnings
import warnings
warnings.filterwarnings(action='ignore')

'''Import Scripts'''
from etl import *

'''Data Tools'''
import numpy as np
import pandas as pd

'''Modeling Tools'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.pipeline import Pipeline

'''For saving models'''
from joblib import dump
import pickle

RANDOM_SEED = 420420


def prepare_data():
    #Set Paths
    TRAIN_PATH = "/data"

    #Prep Data
    print("Preparing Data")
    concept_feature_id_map_train_set = get_concept_feature_id_map(specific_path=TRAIN_PATH, specific_concept_id_list=None, include_parsed_values=True)

    #Save list of features for use in eval set
    dump(concept_feature_id_map_train_set, '/model/feature_dict.pickle')
        
    #Create feature data frames
    df_train_set = create_feature_df(concept_feature_id_map_train_set, path=TRAIN_PATH)

    #Join gold standard
    gs = pd.read_csv(TRAIN_PATH + "/goldstandard.csv")
    df_gold_standard = gs.set_index('person_id')
    df_merged_train_set = df_train_set.join(df_gold_standard)

    #Initalize Data Sets
    X_set =  df_merged_train_set.loc[:, df_merged_train_set .columns != 'status']
    Y_set = df_merged_train_set.status

    X_set = np.array(X_set)
    Y_set = np.array(Y_set)

    return X_set, Y_set

def logit_model (X_train, Y_train, model_type ='None'):
    '''
    Choose model_type when calling function:
    'LR' for Logistic Regression
    'SVM' for support vector machine
    'RF' for Random Forest
    Implementing gridsearchcv pipeline for hyperparameter tuning. 
    '''

    #Tuning models for F1 score
    model_score = 'f1'

    if model_type == 'LR':
        model_pipe = Pipeline([('clf', LogisticRegression(random_state=RANDOM_SEED))])
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model_grid_params = [{'clf__penalty': ['l1', 'l2'],
                       'clf__C': c_range,
                       'clf__solver': ['liblinear'],
                       'clf__class_weight': (None, 'balanced')}]

        print()
        print("Start Model Training - Logistic Regression")
        print()

    elif model_type == 'SVM':
        model_pipe = Pipeline([('svm', SVC(random_state=RANDOM_SEED))])
        c_range = [1, 10, 100, 1000]
        model_grid_params = [{'svm__kernel': ['rbf'],
                        'svm__gamma': [1e-3, 1e-4],
                        'svm__C': c_range},
                        {'svm__kernel': ['linear'], 
                        'svm__C': [1, 10, 100, 1000]}]

        print() 
        print("Start Model Training - Support Vector Machine")
        print()

    elif model_type == 'RF':
        model_pipe = Pipeline([('rf', RandomForestClassifier(random_state=RANDOM_SEED))])
        max_depths = np.linspace(1, 32, 32, endpoint=True)

        model_grid_params = [{'rf__bootstrap': [True, False],
                        'rf__max_depth': max_depths,
                        'rf__max_features': ['auto', 'sqrt', 'log2'],
                        'rf__min_samples_leaf': [3, 4, 5],
                        'rf__min_samples_split': [8, 10, 12],
                        'rf__n_estimators': [100, 200, 300, 500],
                        'rf__criterion': ['gini', 'entropy']}]
        print() 
        print("Start Model Training - Random Forest")
        print()

        
    model_grid = GridSearchCV(estimator=model_pipe, 
                    param_grid=model_grid_params, 
                    scoring=model_score, 
                    verbose = 0, 
                    n_jobs=-1,
                    cv=10) 

    model_clf = model_grid.fit(X_train, Y_train)


    #Save Model
    dump(model_clf, '/model/baseline.joblib')
    print("Training stage finished", flush = True)

   
if __name__ == "__main__":
    X, Y = prepare_data()
    logit_model(X, Y, model_type ='LR')