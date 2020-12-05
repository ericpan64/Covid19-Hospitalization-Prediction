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
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

'''For saving models'''
from joblib import dump, load
import pickle

RANDOM_SEED = 420420

def unpack_model_args_dict(args_dict):
    data_dir = args_dict['data_dir']
    results_dir = args_dict['results_dir']
    cid_file_path = args_dict['cid_list_file']
    use_pca = args_dict['use_pca']
    return data_dir, results_dir, cid_file_path, use_pca

def prepare_data(args_dict):
    '''
    Function to prepare data for training
    import_specific_id_list and use_pca are in a default false state. 
    Utilize either to read in a feature list located in the /data folder listed in featuresList.txt
    '''

    #Set Paths
    TRAIN_PATH, OUTPUT_PATH, CID_FILE_PATH, USE_PCA = unpack_model_args_dict(args_dict)
    cid_list, df_base_path, cf_map_base_path = get_vars_based_on_cid_file_path(TRAIN_PATH, CID_FILE_PATH, OUTPUT_PATH)
    df_pickle, cf_map_pickle = df_base_path + '.pickle', cf_map_base_path + '.pickle'

    #Prep Data
    print("Preparing Data")
    if CID_FILE_PATH != None:
        print("\nUtilizing Custom Feature List")
    concept_feature_id_map_train_set = load(cf_map_pickle)

    #Save list of features for use in eval set
    dump(concept_feature_id_map_train_set, '/model/feature_dict.pickle')
        
    #Create feature data frames
    df_train_set = load(df_pickle)

    #Join gold standard
    gs = pd.read_csv(TRAIN_PATH + "/goldstandard.csv")
    df_gold_standard = gs.set_index('person_id')
    df_merged_train_set = df_train_set.join(df_gold_standard)

    #Initalize Data Sets
    X_set =  df_merged_train_set.loc[:, df_merged_train_set .columns != 'status']
    Y_set = df_merged_train_set.status

    #Implement PCA (Identifying Components Responsible for 90% of data variance)
    if USE_PCA:
        pca = PCA(n_components=0.85) 
        pca.fit(X_set)
        X_set = pca.transform(X_set)

        #Save PCA component for inference
        dump(pca, '/model/pca.pickle')

    X_set = np.array(X_set)
    Y_set = np.array(Y_set)

    return X_set, Y_set

def logit_model (X_set, Y_set):

    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=RANDOM_SEED)

    #Generate Pipelines

    #Logistic Regression
    pipe_lr = Pipeline([('clf', LogisticRegression(random_state=RANDOM_SEED))])
    grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                'clf__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'clf__solver': ['liblinear'],
                'clf__class_weight': (None, 'balanced')}]

    #Support Vector Machine
    pipe_svm = Pipeline([('svm', SVC(random_state=RANDOM_SEED))])
    grid_params_svm = [{'svm__kernel': ['rbf', 'linear'],
                    'svm__gamma': [1e-3, 1e-4],
                    'svm__C': [1, 10, 100, 1000]}]

    #Random Forest
    pipe_rf = Pipeline([('rf', RandomForestClassifier(random_state=RANDOM_SEED))])
    grid_params_rf = [{'rf__bootstrap': [True, False],
                    'rf__max_depth': [5, 8, 15, 25, 30],
                    'rf__max_features': ['auto', 'sqrt', 'log2'],
                    'rf__min_samples_leaf': [3, 4, 5],
                    'rf__min_samples_split': [8, 10, 12],
                    'rf__n_estimators': [5, 25, 50, 75, 100],
                    'rf__criterion': ['gini', 'entropy']}]

    #Assign Scores
    scores = {'AUC': 'roc_auc', 'Average_Precision': make_scorer(average_precision_score)}

    #Construct Grid Search Pipelines
    grid_lr = GridSearchCV(estimator=pipe_lr, 
                    param_grid=grid_params_lr, 
                    scoring= scores,
                    refit = 'AUC',
                    verbose = 0, 
                    n_jobs=-1,
                    cv=10)

    grid_svm = GridSearchCV(estimator=pipe_svm, 
                param_grid=grid_params_svm, 
                scoring= scores,
                refit = 'AUC',
                verbose = 0, 
                n_jobs=-1,
                cv=10)

    grid_rf = GridSearchCV(estimator=pipe_rf, 
            param_grid=grid_params_rf, 
            scoring= scores,
            refit = 'AUC',
            verbose = 0, 
            n_jobs=-1,
            cv=10)

    #List of pipelines for iteration
    # grids = [grid_lr,grid_svm, grid_rf]
    grids = [grid_lr]

    #Dictionary of pipelines and classifier types for reference
    # grid_dict = {0: 'Logistic Regression', 1:'Support Vector Machine', 2:'Random Forest'}
    grid_dict = {0: 'Logistic Regression'}

    #Fit the grid search objects
    print('\nPerforming Model Optimization')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])	
        # Fit grid search	
        gs.fit(X_train, Y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Predict on test data with params
        Y_pred = gs.predict(X_test)
        # Test data auc of model with params
        auc_score = roc_auc_score(Y_test, Y_pred)
        avg_precision_score = average_precision_score(Y_test, Y_pred)
        print('AUC score for best params: %.3f ' % auc_score)
        print('Average Precision score for best params: %.3f ' % avg_precision_score)
        # Track (highest test AUC or Average Precision) model
        if auc_score > best_acc:
            best_acc = auc_score
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best Score: %s' % grid_dict[best_clf])

    # Save best grid search pipeline to file
    dump(best_gs, '/model/baseline.joblib')
    print('\nSaved %s grid search pipeline to file: baseline.joblib' % (grid_dict[best_clf]))
    print("\nTraining stage finished", flush = True)
   
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Trains and generates the ML model to: /model/baseline.joblib')
    parser.add_argument('--data_dir', type=str, required=True, help='Filepath to directory containing OMOP-formatted .csv files (e.g. "~/Documents/data_dir")')
    parser.add_argument('--results_dir', type=str, required=True, help='Filepath to directory to save the output .pickle and .csv files (e.g. "~/Documents/results_dir")')
    parser.add_argument('--cid_list_file', type=str, required=False, help='Filepath to .txt file with concept_ids to use (e.g. "~/Documents/data_dir/cid_list.txt")')
    parser.add_argument('--use_pca', type=bool, required=False, help='Set to True to apply PCA before model training')
    args_dict = vars(parser.parse_args())
    
    X, Y = prepare_data(args_dict)
    logit_model(X, Y)