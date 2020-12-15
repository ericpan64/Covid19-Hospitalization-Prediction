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


docker = 1
TRAIN_PATH = '../synthetic_data/training/' # should be "/data/" for docker.
MODEL_PATH = '../model/' # should be "/model/" for docker.

if docker == 1:
    TRAIN_PATH = '/data/'
    MODEL_PATH = '/model/'

# ================ CODE FROM DREAM Challenge ======================
def add_COVID_measurement_date():
    measurement = pd.read_csv(TRAIN_PATH+"measurement.csv",usecols =['person_id','measurement_date','measurement_concept_id','value_as_concept_id'])
    measurement = measurement.loc[measurement['measurement_concept_id']==706163]
    measurement['value_as_concept_id'] = measurement['value_as_concept_id'].astype(int)
    measurement = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]
    measurement = measurement.sort_values(['measurement_date'],ascending=False).groupby('person_id').head(1)
    covid_measurement = measurement[['person_id','measurement_date']]
    return covid_measurement

def add_demographic_data(covid_measurement):
    '''add demographic data including age, gender and race'''
    person = pd.read_csv(TRAIN_PATH+'person.csv',usecols = ['person_id','gender_concept_id','year_of_birth','race_concept_id'])
    demo = pd.merge(covid_measurement,person,on=['person_id'], how='inner')
    demo['measurement_date'] = pd.to_datetime(demo['measurement_date'], format='%Y-%m-%d')
    demo['year_of_birth'] = pd.to_datetime(demo['year_of_birth'], format='%Y')
    demo['age'] = demo['measurement_date'] - demo['year_of_birth']
    demo['age'] = demo['age'].apply(lambda x: x.days/365.25)
    print("patients' ages are calculated", flush = True)
    person["count"] = 1
    gender = person.pivot(index = "person_id", columns="gender_concept_id", values="count")
    gender.reset_index(inplace = True)
    gender.fillna(0,inplace = True)
    race = person.pivot(index ="person_id", columns="race_concept_id", values="count")
    race.reset_index(inplace = True)
    race.fillna(0,inplace = True)
    race = race[['person_id', 8516, 8515, 8527, 8552]]
    gender = gender[['person_id',8532]]
    print("patients' gender and race information are added", flush = True)
    scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
    scaled_column = scaler.fit_transform(demo[['age']])
    demo = pd.concat([demo, pd.DataFrame(scaled_column,columns = ['scaled_age'])],axis=1)
    predictors = demo[['person_id','scaled_age']]
    predictors = predictors.merge(gender, on = ['person_id'], how = 'left')
    predictors = predictors.merge(race, on = ['person_id'], how = 'left')
    predictors.fillna(0,inplace = True)
    return predictors


# ================ OUR OWN CODE ======================
def get_id_list(n_features=300):
    print("number of features from id list: ", n_features)
    f = open('idlist.txt', 'r')
    ids = [int(line.strip()) for line in f]
    f.close()
    if n_features < len(ids):
        ids = ids[0:n_features]
    idSet = set(ids)
    return idSet

def get_id_from_train():
    f = open(MODEL_PATH + '/predictor_ids.txt')
    line = [line for line in f]
    ids = [int(i) for i in line[0].strip().split(',')]
    return set(ids)

def get_features_from_list(testing=False):
    idSet = {}
    if testing:
        idSet = get_id_from_train()
    else:
        idSet = get_id_list()
        
    # condition
    condition = pd.read_csv(TRAIN_PATH+"condition_occurrence.csv",usecols =['person_id','condition_concept_id'])
    condition = condition[condition['condition_concept_id'].isin(idSet)]
    print("N unique condition: ", len(np.unique(condition['condition_concept_id'].values)))
    condition = rename_col(condition)

    # drug exposure
    drug = pd.read_csv(TRAIN_PATH+"drug_exposure.csv",usecols =['person_id','drug_concept_id'])
    drug = drug[drug['drug_concept_id'].isin(idSet)]
    print("N unique drug: ", len(np.unique(drug['drug_concept_id'].values)))
    drug = rename_col(drug)

    # device exposure
    device = pd.read_csv(TRAIN_PATH+"device_exposure.csv",usecols =['person_id','device_concept_id'])
    device = device[device['device_concept_id'].isin(idSet)]
    print("N unique device: ", len(np.unique(device['device_concept_id'].values)))
    device = rename_col(device)

    # measurement
    measurement = pd.read_csv(TRAIN_PATH+"measurement.csv",usecols =['person_id','measurement_concept_id'])
    measurement = measurement[measurement['measurement_concept_id'].isin(idSet)]
    print("N unique measurement: ", len(np.unique(measurement['measurement_concept_id'].values)))
    measurement = rename_col(measurement)

    # observation
    observation = pd.read_csv(TRAIN_PATH+"observation.csv",usecols =['person_id','observation_concept_id'])
    observation = observation[observation['observation_concept_id'].isin(idSet)]
    print("N unique observation: ", len(np.unique(observation['observation_concept_id'].values)))
    observation = rename_col(observation)

    # procedure
    procedure = pd.read_csv(TRAIN_PATH+"procedure_occurrence.csv",usecols =['person_id','procedure_concept_id'])
    procedure = procedure[procedure['procedure_concept_id'].isin(idSet)]
    print("N unique procedure: ", len(np.unique(procedure['procedure_concept_id'].values)))
    procedure = rename_col(procedure)

    big_table = pd.concat([condition, drug, device, measurement, observation, procedure])
    big_table = pivot_table(big_table)
    return big_table


def rename_col(df):
    new_df = df
    original_cols = new_df.columns.values
    new_df.rename(columns={original_cols[1]: 'concept_id'}, inplace = True)
    return new_df

def pivot_table(df):
    # rename to generic name
    new_df = df

    new_df = new_df.value_counts(['person_id', 'concept_id']).reset_index(name='count')
    new_df = new_df.pivot(index = "person_id", columns="concept_id", values="count")
    new_df.reset_index(inplace = True)
    new_df.fillna(0,inplace = True)
    scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
    for col in new_df.columns.values[1:]: 
        new_df[col] = scaler.fit_transform(new_df[[col]])
    return new_df


def save_ids(path, ids):
    f = open(path, 'w')
    line = ','.join([str(concept) for concept in ids])
    f.write(line)
    f.close()

def add_additional_concepts(df, idSet):
    existing_concepts = set([int(i) for i in df.columns.values[1:]])
    missing_concepts = list(idSet - existing_concepts)
    for concept in missing_concepts:
        df[str(concept)] = 0
    return df
    
    