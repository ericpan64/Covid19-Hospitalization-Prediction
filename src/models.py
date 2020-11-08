'''Import Scripts'''
from plots import *
from etl import *

'''Data Tools'''
import numpy as np
from numpy import mean
import pandas as pd

'''Modeling Tools'''
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.pipeline import Pipeline

'''For saving models'''
import csv
from joblib import dump

import imblearn.under_sampling as under

RANDOM_SEED = 420420


def display_metrics(classifierName,Y_true,Y_pred):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc, auc_, precision, recall, f1score = classification_metrics(Y_true,Y_pred)
    print(("Accuracy: "+str(acc)))
    print(("AUC: "+str(auc_)))
    print(("Precision: "+str(precision)))
    print(("Recall: "+str(recall)))
    print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")

def output_classification_report(Y_true, Y_pred, classifier):

    report = classification_report(Y_true, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv ('../classification_reports/{0}_classification_report.csv'.format(classifier), index = False, header=True)
    return

def write_params_to_csv(param_dict,score,classifier='None'):
    a_file = open('../best_params/{0}.csv'.format(classifier), "w")
    a_dict = param_dict
    writer = csv.writer(a_file)
    
    for key, value in a_dict.items():
        writer.writerow([key, value]) 
    writer.writerow(["score",score]) 
    a_file.close()
    return

def logistic_regression_pred(X_train, Y_train, X_test, Y_test, classifier_title='None'): #, X_test):
    """
    input: X_train, Y_train and X_test
    output: Y_pred
    Implementing gridsearchcv pipeline for hyperparameter tuning. 
    """
    pipe_lr = Pipeline([('clf', LogisticRegression(random_state=RANDOM_SEED))])
    c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                       'clf__C': c_range,
                       'clf__solver': ['liblinear'],
                       'clf__class_weight': (None, 'balanced')}]
    
    print("Logistic Regression")
    print()

    scores = ['roc_auc']#['recall','roc_auc', 'f1'] # Need to revisit recall
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        lr_grid = GridSearchCV(estimator=pipe_lr, 
                      param_grid=grid_params_lr, 
                      scoring=score, 
                      verbose = 0, 
                      n_jobs=-1,
                      cv=10) 
    
        clf_lr = lr_grid.fit(X_train, Y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf_lr.best_params_)
        print()
        print("Best {0} score found on development set:".format(score))
        print()
        print(clf_lr.best_score_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_lr.cv_results_['mean_test_score']
        stds = clf_lr.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_lr.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
        print()
        #Save Params to CSV
        write_params_to_csv(clf_lr.best_params_,clf_lr.best_score_,classifier='Logistic_Regression_{0}'.format(score))
        
        #Save Model
        dump(clf_lr, '../models/{0}_{1}.joblib'.format(classifier_title,score))

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        Y_true, Y_pred = Y_test, clf_lr.predict(X_test)

        #Plot AURC Plot
        plot_aurc_plot(Y_true, Y_pred, title='Logistic_Regression_ROC_Curve_{0}'.format(score))
        output_classification_report(Y_true, Y_pred, 'Logistic_Regression_{0}'.format(score))
        print(classification_report(Y_true, Y_pred))
        
        # display_metrics('Logistic_Regression_{0}'.format(score),Y_true,Y_pred)
        print()
    
    print("Logistic Regression Training stage finished", flush = True)

    return clf_lr

def classification_metrics(Y_true, Y_pred):
    """
    #input: Y_pred,Y_true
    #output: accuracy, auc, precision, recall, f1-score
    """
    #TODO: Figure out how to save the values below
    acc = accuracy_score(Y_true,Y_pred)
    auc_ = roc_auc_score(Y_true,Y_pred)
    precision = precision_score(Y_true,Y_pred)
    recall = recall_score(Y_true,Y_pred)
    f1score = f1_score(Y_true,Y_pred)
    return acc,auc_,precision,recall,f1score

def main():
    #Train
    DATA_PATH = "../data/DREAM_data"
    TRAIN_PATH = DATA_PATH + '/training'

    #Prep Data
    pid_list_train = get_unique_pid_list(path=TRAIN_PATH)
    X_set = create_feature_df(pid_list_train, path=TRAIN_PATH, impute_strategy=0.0, use_multivariate_impute=False)
    gs = pd.read_csv(TRAIN_PATH + "/goldstandard.csv")
    Y_set = gs.drop(['person_id'], axis = 1)
    X_set = np.array(X_set)
    Y_set = np.array(Y_set).ravel()

    #Undersampling
    UnderSampling = under.ClusterCentroids(sampling_strategy={1:100, 0:200}, random_state=83, voting='hard')
    x_resampled, y_resampled = UnderSampling.fit_resample(X_set, Y_set)

    X_train, X_test, Y_train, Y_test = train_test_split(x_resampled, y_resampled, test_size=0.8, random_state=RANDOM_SEED)

    clf_lr = logistic_regression_pred(X_train,Y_train,X_test,Y_test,'Baseline_Logistic_Regression')

if __name__ == "__main__":
    main()