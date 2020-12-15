import datetime
import pandas as pd
import numpy as np
from datetime import datetime
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
'''for saving models'''
from joblib import dump
from simple_etl import *
import sklearn.decomposition as skdc
import sklearn.pipeline as skpl
import pickle as pk

docker = 1
TRAIN_PATH = '../synthetic_data/training/' # should be "/data/" for docker.
MODEL_PATH = '../model/' # should be "/model/" for docker.

if docker == 1:
    TRAIN_PATH = '/data/'
    MODEL_PATH = '/model/'

def logit_model(predictors):
    '''
    apply logistic regression models for selected demographics features and use GridSearchCV to optimize parameters
    '''
    X = predictors.drop(['person_id'], axis = 1)
    #features = X.columns.values
    gs = pd.read_csv(TRAIN_PATH+'goldstandard.csv')
    result = predictors.merge(gs,on = ['person_id'], how ='left')
    result.fillna(0,inplace = True)
    X = np.array(X)
    Y = np.array(result[['status']]).ravel()
    pca = skdc.PCA(n_components = 0.8)
    X = pca.fit_transform(X)
    pk.dump(pca, open(MODEL_PATH + "pca.pkl","wb"))
    clf = LogisticRegressionCV(cv = 10, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
    max_iter = 100, verbose = 0, n_jobs = None)
    model = clf.fit(X,Y)
    dump(model, MODEL_PATH + 'baseline.joblib')
    print("Training stage finished", flush = True)

if __name__ == '__main__':

    # ETL and preprocessing. Note that it should read data from /app, which is the volume of the docker image.
    #covid_measurement = add_COVID_measurement_date()
    #predictors = add_demographic_data(covid_measurement)

    predictors = get_features_from_list()

    # save predictor to a list
    save_ids(MODEL_PATH + 'predictor_ids.txt', predictors.columns.values[1:])

    # train the model and save it to the /model folder.
    logit_model(predictors)