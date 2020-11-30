#Ignore Warnings
import warnings
warnings.filterwarnings(action='ignore')

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.pipeline import Pipeline

'''For saving models'''
import csv
from joblib import dump

import imblearn.under_sampling as under
from imblearn.over_sampling import SMOTE

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
    '''
    Function to save classification report to CSV
    '''

    report = classification_report(Y_true, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv ('../classification_reports/{0}_classification_report.csv'.format(classifier), index = False, header=True)
    return

def write_params_to_csv(param_dict,score,classifier='None'):
    '''
    Function to save model parameters to CSV
    '''
    a_file = open('../best_params/{0}.csv'.format(classifier), "w")
    a_dict = param_dict
    writer = csv.writer(a_file)
    
    for key, value in a_dict.items():
        writer.writerow([key, value]) 
    writer.writerow(["score",score]) 
    a_file.close()
    return

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

def prepare_data(implement_undersampling=None, implement_oversampling=None):
    #Set Paths
    DATA_PATH = "../data/DREAM_data"
    TRAIN_PATH = DATA_PATH + '/training'

    #Prep Data
    print("Preparing Data")
    concept_feature_id_map_train_set, corr_series = get_highest_corr_concept_feature_id_map_and_corr_series(specific_path=None, keep_first_n=None, use_parsed_values=True, agg_imp_config=DEFAULT_AGG_IMP_CONFIG)

    #Create feature data frames
    df_train_set = create_feature_df(concept_feature_id_map_train_set, path=TRAIN_PATH)

    #Join gold standard to imputed and normalized data frames
    gs = pd.read_csv(TRAIN_PATH + "/goldstandard.csv")
    df_gold_standard = gs.set_index('person_id')
    df_merged_train_set = df_train_set.join(df_gold_standard)

    #Initalize Data Sets
    X_set =  df_merged_train_set.loc[:, df_merged_train_set .columns != 'status']
    Y_set = df_merged_train_set.status

    #Implement PCA (Identifying Components Responsible for 90% of data variance)
    pca = PCA(n_components=.90) 
    pca.fit(X_set)
    X_set = pca.transform(X_set)

    if implement_undersampling == True:
        #Undersampling Data
        UnderSampling = under.ClusterCentroids(sampling_strategy={1:100, 0:300}, random_state=RANDOM_SEED, voting='hard')
        X_set, Y_set = UnderSampling.fit_resample(X_set, Y_set)
    
    elif implement_oversampling == True:
        #SMOTE Oversampling
        X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=RANDOM_SEED)

        oversample = SMOTE()
        X_train_up, Y_train_up = oversample.fit_resample(X_train, Y_train)

        return X_train_up, X_test, Y_train_up, Y_test

    #Split Data for Training/Eval 80/20 Split
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=RANDOM_SEED)

    return X_train, X_test, Y_train, Y_test

def modeling (X_train, Y_train, X_test, Y_test, model_type ='None', classifier_title='None'):
    '''
    Choose model_type when calling function:
    'LR' for Logistic Regression
    'SVM' for support vector machine
    'RF' for Random Forest

    input: X_train, Y_train and X_test, Y_test
    output: Y_pred
    Implementing gridsearchcv pipeline for hyperparameter tuning. 
    '''

    if model_type == 'LR':
        model_pipe = Pipeline([('clf', LogisticRegression(random_state=RANDOM_SEED))])
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model_grid_params = [{'clf__penalty': ['l1', 'l2'],
                       'clf__C': c_range,
                       'clf__solver': ['liblinear'],
                       'clf__class_weight': (None, 'balanced')}]

        model_scores = ['recall','roc_auc', 'f1']

        print()
        print("Logistic Regression")
        print()

    elif model_type == 'SVM':
        model_pipe = Pipeline([('svm', SVC(random_state=RANDOM_SEED))])
        c_range = [1, 10, 100, 1000]
        model_grid_params = [{'svm__kernel': ['rbf'],
                        'svm__gamma': [1e-3, 1e-4],
                        'svm__C': c_range},
                        {'svm__kernel': ['linear'], 
                        'svm__C': [1, 10, 100, 1000]}]

        model_scores = ['recall','roc_auc', 'f1']

        print() 
        print("Support Vector Machine")
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

        model_scores = ['recall','roc_auc', 'f1']

        print() 
        print("Random Forest")
        print()


    for score in model_scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        model_grid = GridSearchCV(estimator=model_pipe, 
                        param_grid=model_grid_params, 
                        scoring=score, 
                        verbose = 0, 
                        n_jobs=-1,
                        cv=10) 

        model_clf = model_grid.fit(X_train, Y_train)

        print("Best parameters set found on development set:")
        print()
        print(model_clf.best_params_)
        print()
        print("Best {0} score found on development set:".format(score))
        print()
        print(model_clf.best_score_)
        print()
        print("Grid scores on development set:")
        print()
        means = model_clf.cv_results_['mean_test_score']
        stds = model_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds,model_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
        print()
        #Save Params to CSV
        write_params_to_csv(model_clf.best_params_,model_clf.best_score_,classifier='{0}_{1}'.format(model_type, score))
        
        #Save Model
        dump(model_clf, '../models/{0}_{1}.joblib'.format(classifier_title,score))

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        Y_true, Y_pred = Y_test, model_clf.predict(X_test)

        #Plot AURC Plot
        plot_aurc_plot(Y_true, Y_pred, title='{0}_ROC_Curve_{1}'.format(model_type, score))
        output_classification_report(Y_true, Y_pred, 'Logistic_Regression_{0}'.format(model_type, score))
        print(classification_report(Y_true, Y_pred))
        print(len(Y_true))
        print(len(Y_test))
        print(confusion_matrix(Y_true, Y_pred))

        print()

    print("{0} Training stage finished".format(model_type), flush = True)

    return model_clf


def main():
    '''
    Implement model training on training dataset

    Remember to Include Title Details Per Data Preperation
    -Example: Basline Modeling
    clf_lr = logistic_regression_pred(X_train, Y_train, X_test, Y_test, 'Baseline_Logistic_Regression')

    -Example: Oversampling
    clf_lr = logistic_regression_pred(X_train, Y_train, X_test, Y_test, 'Oversampled_Logistic_Regression')

    '''

    X_train, X_test, Y_train, Y_test = prepare_data(implement_undersampling=True, implement_oversampling=False)


    #Implement Logistic Regression
    clf_lr = modeling(X_train, Y_train, X_test, Y_test, model_type ='LR', classifier_title ='Baseline_Logistic_Regression')

    #Implement Support Vector Machine
    #clf_svm = modeling(X_train, Y_train, X_test, Y_test, model_type ='SVM', classifier_title ='Baseline_Support_Vector_Machine')

    #Implement Random Forest
    #clf_rf = modeling(X_train, Y_train, X_test, Y_test, model_type ='RF', classifier_title ='Baseline_Random_Forest')

if __name__ == "__main__":
    main()