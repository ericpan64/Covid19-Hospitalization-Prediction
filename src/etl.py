import pandas as pd
import numpy as np
from os import getcwd
from sklearn.experimental import enable_iterative_imputer # https://stackoverflow.com/a/56738037
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix

''' Global Variables '''
RANDOM_SEED = 420420
DATA_PATH = '../data/DREAM_data'
TRAIN_PATH = DATA_PATH + '/training'
EVAL_PATH = DATA_PATH + '/evaluation'
FILENAME_LIST = ['condition_occurrence.csv', 'device_exposure.csv', 'goldstandard.csv', 
    'measurement.csv', 'observation_period.csv', 'observation.csv', 
    'person.csv', 'procedure_occurrence.csv', 'visit_occurrence.csv']
FILENAME_CLIN_CONCEPT_MAP = { # maps str->list(str)
    'condition_occurrence.csv': ['condition_concept_id',
                                'condition_type_concept_id',
                                'condition_source_concept_id',
                                'condition_status_concept_id'],
    'device_exposure.csv': ['device_concept_id',
                            'device_type_concept_id',
                            'device_source_concept_id'],
    'measurement.csv': ['measurement_concept_id',
                        'measurement_type_concept_id',
                        'operator_concept_id',
                        'value_as_concept_id',
                        'unit_concept_id',
                        'measurement_source_concept_id'],
    'observation.csv': ['observation_concept_id',
                        'observation_type_concept_id',
                        'value_as_concept_id',
                        'qualifier_concept_id',
                        'unit_concept_id',
                        'observation_source_concept_id'],
    'observation_period.csv': ['period_type_concept_id'],
    'procedure_occurrence.csv': ['procedure_concept_id',
                                'procedure_type_concept_id',
                                'modifier_concept_id',
                                'procedure_source_concept_id'],
    'visit_occurrence.csv': ['visit_concept_id',
                        'visit_type_concept_id',
                        'visit_source_concept_id',
                        'admitting_source_concept_id',
                        'discharge_to_concept_id']}
DATA_DICT_DF = pd.read_csv(DATA_PATH + '/data_dictionary.csv').loc[:, ['concept_id', 'concept_name', 'table']]
CONCEPT_ID_TO_NAME_MAP = DATA_DICT_DF.loc[:, ['concept_id', 'concept_name']].set_index('concept_id').to_dict()['concept_name']
CONCEPT_ID_TO_TABLE_MAP = DATA_DICT_DF.loc[:, ['concept_id', 'table']].set_index('concept_id').to_dict()['table']

''' Public '''
def get_unique_pid_list(path=TRAIN_PATH):
    """
    Gets unique list of patient IDs from person.csv
    :returns: list
    """
    df = pd.read_csv(path + '/person.csv')
    pid_list = df['person_id'].unique().tolist()
    return pid_list

def create_feature_df(pid_list, path=TRAIN_PATH, impute_strategy=0.0, use_multivariate_impute=False):
    """
    Generates the feature DataFrame of shape m x n, with m=len(pid_list) and n=# of features
        NOTE: the value of each cell is the _count_ of a feature for a given patient.
            Need to add additional OMOP-specific logic to parse-out values for concepts with values (e.g. heart rate)
        If a patient is missing a feature, the given inpute_strategy will be used (univariate unless use_multivariate_impute=True)
        By default this pulls all concepts sourced from csv columns specified in FILENAME_CLIN_CONCEPT_MAP
    :param pid_list: list of person IDs
    :param impute_strategy: str in {'most_frequent', 'mean', 'median'} OR any numeric (impute constant)
    #deprecated (takes forever, also doesn't make sense) :param use_multivariate_impute: bool
    #TODO implement this :param custom_concept_id_set: set (whitelist)
    #TODO implement this :param exclude_concept_id_set: set (blacklist)
    :return: DataFrame
    """
    # helper function
    def normalize_array(x):
        """ Given numpy array, return with the each column normalized """
        # from: https://stackoverflow.com/a/29661707
        x_normed = x / x.max(axis=0)
        return x_normed

    # handle input edge cases
    if len(pid_list) == 0:
        raise ValueError(f"Error: pid_list has length 0, got: {pid_list}")
    # if type(custom_concept_id_set) == list:
    #     custom_concept_id_set = set(custom_concept_id_set)
    # if type(exclude_concept_id_set) == list:
    #     exclude_concept_id_set = set(exclude_concept_id_set)

    # get concept indices, person list, and concept_id->feature_id dict
    concept_id_list = get_concept_list_ordered_by_sparsity(path=path)
    pid_list = get_unique_pid_list(path=path)
    concept_feature_id_map = generate_concept_feature_id_map(concept_id_list)

    # get all unique person_id, concept_id pairs as DataFrame
    df_all = get_concept_pid_pairs(path=path)

    # for each person_id, get corresponding concept_id, count pairs
    ## get counts
    tot = len(df_all)
    df_ones = pd.DataFrame(np.ones((tot, 1)))
    df_all_w_ones = pd.concat([df_all, df_ones], axis=1)
    df_all_summed = df_all_w_ones.groupby(['concept_id', 'person_id'])\
        .agg(['sum'])\
        .reset_index()\
        .set_index('person_id')
    ## remove column multiindex, rename column
    df_all_summed.columns = df_all_summed.columns.droplevel([1])
    df_all_summed = df_all_summed.rename(columns={df_all_summed.columns[1]: 'sum'})

    # generate matrix (rows=person_id, cols=feature_id, values=sum)
    get_feature_id = lambda cid: concept_feature_id_map[cid]
    rows = df_all_summed.index.to_list()
    cols = df_all_summed.loc[:, 'concept_id'].apply(get_feature_id).to_list()
    vals = df_all_summed.loc[:, 'sum'].to_list()
    m = len(pid_list)
    n = len(concept_id_list)
    df_sparse = coo_matrix((vals, (rows, cols)), shape=(m, n))
    arr_dense = df_sparse.toarray()
    
    # impute data
    # if use_multivariate_impute:
    #     arr_imputed = impute_missing_data_multivariate(arr_dense, missing_val=0.0, strategy=impute_strategy)

    arr_imputed = impute_missing_data_univariate(arr_dense, missing_val=0.0, strategy=impute_strategy)

    # normalize data
    arr_norm = normalize_array(arr_imputed)

    return pd.DataFrame(arr_norm)

''' "Private" '''
def load_csvs_to_dataframe_dict(fn_list=FILENAME_LIST, path=TRAIN_PATH):
    """
    Loads csvs into single dictionary data structure
    :returns: dict (str->DataFrame)
    """
    fn_to_df_dict = {}
    for fn in fn_list:
        try:
            df = pd.read_csv(path + '/' + fn)
            fn_to_df_dict[fn] = df
        except:
            raise ValueError(f"Error: could not read file: {path+'/'+fn}")
    return fn_to_df_dict

def impute_missing_data_univariate(X, missing_val=np.nan, strategy='most_frequent'):
    """
    Imputes missing values in X using univariate approaches
    :param X: DataFrame
    :param missing_val: int or np.nan
    :param strategy: str in {'most_frequent', 'mean', 'median'} OR any numeric (impute constant)
    :returns: X with missing_val imputed
    """
    # sklearn docs: https://scikit-learn.org/stable/modules/impute.html#impute
    # handle constant impute case
    val = None
    if type(strategy) != str:
        try:
            val = float(strategy)
            strategy = 'constant'
        except:
            raise ValueError(f"Error: parameter 'strategy' needs to be string or numeric, got: {strategy}")

    # build imputer, then apply the transform
    imp = SimpleImputer(missing_values=missing_val, strategy=strategy, fill_value=val)
    X_new = imp.fit_transform(X)
    return X_new

def get_concept_pid_pairs(path=TRAIN_PATH):
    """
    Gets all (non-unique) concept_id-person_id pairs as a DataFrame with the following columns:
        person_id
        concept_id
    To get unique, use pd.DataFrame.drop_duplicates after calling this function
    :returns: DataFrame
    """
    # use map to quickly access copy of df
    fn_to_df_map = load_csvs_to_dataframe_dict(path=path)

    # Get all person_id, concept_id occurrences (duplicates included) as a DataFrame
    ## estimate df size (assume worst case each person has 1 instance of every concept)
    person_count = len(fn_to_df_map['person.csv']['person_id'])
    clin_concept_count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            clin_concept_count += len(df[col].unique())
    ## init df_all
    idx = range(person_count * clin_concept_count)
    cols = ['person_id', 'concept_id']
    df_all = pd.DataFrame(index=idx, columns=cols)
    ## populate df_all (unique person_id per concept_id)
    count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            # pre-process: get all person_id, concept_id pairs (non-unique)
            df_sliced = df.loc[:, ['person_id', col]]
            df_sliced = df_sliced.dropna()
            df_sliced = df_sliced.rename(columns={col: 'concept_id'})
            # set appropriate index
            n_rows = len(df_sliced)
            idx = pd.Series(range(count, count+n_rows))
            df_sliced = df_sliced.set_index(idx) 
            # append to df_all
            df_all.iloc[idx, :] = df_sliced
            count += n_rows
    ## remove NaN
    df_all = df_all.dropna().astype('int')
    return df_all

def generate_concept_summary(path=TRAIN_PATH, save_csv=True):
    """
    Gets a summary of concept_id-person_id pairs as a DataFrame with the following columns:
        concept_id
        concept_name (if in data_dictionary.csv)
        avg_per_pid (if a patient had the concept, how many instances were there)
        from_table (if in data_dictionary.csv)
        unique_pid_count (how many unique patients had the concept)
    :creates_file: concept_summary.csv
    :returns: DataFrame
    """
    # get all concept_id-person_id pairs
    df_all = get_concept_pid_pairs(path)

    # Get count of unique person_id per concept_id
    df_all_summary = df_all.drop_duplicates(keep='first')\
        .groupby(['concept_id'])\
        .agg({'person_id': 'count'})\
        .rename(columns={'person_id': 'unique_pid_count'})\
        .sort_values('unique_pid_count', ascending=False)
    ## add concept_name, table labels from data_dict
    ### names
    concept_ids = df_all_summary.reset_index().loc[:, 'concept_id']
    cid_to_name = lambda cid: CONCEPT_ID_TO_NAME_MAP[cid] if cid in CONCEPT_ID_TO_NAME_MAP else pd.NA
    concept_names = concept_ids.apply(cid_to_name).rename('concept_name')
    concept_ids_with_names = pd.concat([concept_ids, concept_names], axis=1).set_index('concept_id')
    df_all_summary.insert(0, "concept_name", concept_ids_with_names)
    ### table
    cid_to_table = lambda cid: CONCEPT_ID_TO_TABLE_MAP[cid] if cid in CONCEPT_ID_TO_TABLE_MAP else pd.NA
    concept_table = concept_ids.apply(cid_to_table).rename('from_table')
    concept_ids_with_table = pd.concat([concept_ids, concept_table], axis=1).set_index('concept_id')
    df_all_summary.insert(1, "from_table", concept_ids_with_table)

    # Get count of average # of occurrences per person with the given concept_id
    m = len(df_all)
    df_ones = pd.DataFrame(np.ones((m, 1)))
    df_all_w_ones = pd.concat([df_all, df_ones], axis=1)
    df_all_avg = df_all_w_ones.groupby(['concept_id', 'person_id'])\
        .agg(['sum'])\
        .reset_index()\
        .rename(columns={0: 'avg_per_pid'})\
        .groupby(['concept_id'])\
        .agg(['mean'])
    ## clean-up formatting a bit
    df_all_avg.columns.droplevel([1,2]) # remove multiindex
    df_all_avg = df_all_avg.loc[:, 'avg_per_pid'] # keep only avg_per_pid
    ## add to df_all_summary
    df_all_summary.insert(1, "avg_per_pid", df_all_avg)

    # save to csv
    if save_csv:
        df_all_summary.to_csv(DATA_PATH + '/concept_summary.csv')

    return df_all_summary.reset_index()

def get_concept_list_ordered_by_sparsity(path=TRAIN_PATH, least_sparse_first=True):
    """
    Gets unique list of concept_ids from concept_summary.csv ordered by unique_pid_count, avg_per_pid
    :returns: list (sorted)
    """
    # get df (each row = concept)
    df = generate_concept_summary(path, save_csv=False)
    # sort dataframe
    asc = not least_sparse_first
    df_sorted = df.sort_values(['unique_pid_count', 'avg_per_pid'], axis=0, ascending=asc)
    return df_sorted.loc[:, 'concept_id'].tolist()

def generate_concept_feature_id_map(concept_id_list):
    """
    Generates dict mapping concept_id to feature_id (0-indexed)
    :return: dict (str->int)
    """
    if type(concept_id_list) != list:
        raise TypeError(f"Error: concept_id_list needs to be a list, got: {type(concept_id_list)}")
    # using: https://stackoverflow.com/a/36460020
    return {k: v for v, k in enumerate(concept_id_list)}

''' Deprecated '''
def impute_missing_data_multivariate(X, missing_val=np.nan, strategy='most_frequent', max_iter=5):
    """
    Imputes missing values in X using multivariate approach (MICE)
        NOTE: there are more possible parameters to tweak, though for simplicity focusing on a few to start
    :param X: DataFrame
    :param missing_val: int or np.nan
    :param strategy: str in {'most_frequent', 'mean', 'median', 'constant'}
    :param max_iter: int
    :returns: X with missing_val imputed
    """
    # sklearn docs: https://scikit-learn.org/stable/modules/impute.html#impute
    # MICE paper: https://www.jstatsoft.org/article/view/v045i03

    # handle passed constant case
    if type(strategy) in {int, float}:
        strategy='constant'

    # build imputer, then apply transform
    imp = IterativeImputer(random_state=RANDOM_SEED, missing_values=missing_val, \
        initial_strategy=strategy, max_iter=max_iter)
    X_new = imp.fit_transform(X)
    return X_new