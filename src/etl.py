import pandas as pd
import numpy as np
from os import getcwd
from sklearn.experimental import enable_iterative_imputer # https://stackoverflow.com/a/56738037
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix

''' Global Variables '''
RANDOM_SEED = 420420
DATA_PATH = '../data/DREAM_data'
TRAIN_PATH = DATA_PATH + '/training'
EVAL_PATH = DATA_PATH + '/evaluation'
FILENAME_LIST = ['condition_occurrence.csv', 'device_exposure.csv', 'goldstandard.csv', 
    'measurement.csv', 'observation_period.csv', 'observation.csv', 
    'person.csv', 'procedure_occurrence.csv', 'visit_occurrence.csv']
FILENAME_CLIN_CONCEPT_MAP = {
    'condition_occurrence.csv': ['condition_concept_id',
                                # 'condition_type_concept_id',
                                'condition_source_concept_id',
                                # 'condition_status_concept_id'
                                ],
    'device_exposure.csv': ['device_concept_id',
                            # 'device_type_concept_id',
                            'device_source_concept_id'],
    'measurement.csv': ['measurement_concept_id',
                        'measurement_type_concept_id',
                        # 'operator_concept_id',
                        # 'value_as_concept_id',
                        # 'unit_concept_id',
                        'measurement_source_concept_id'],
    'observation.csv': ['observation_concept_id',
                        # 'observation_type_concept_id',
                        # 'value_as_concept_id',
                        'qualifier_concept_id',
                        # 'unit_concept_id',
                        'observation_source_concept_id'],
    # 'observation_period.csv': ['period_type_concept_id'],
    'procedure_occurrence.csv': ['procedure_concept_id',
                                # 'procedure_type_concept_id',
                                'modifier_concept_id',
                                'procedure_source_concept_id'],
    'visit_occurrence.csv': ['visit_concept_id',
                        # 'visit_type_concept_id',
                        'visit_source_concept_id',
                        # 'admitting_source_concept_id',
                        # 'discharge_to_concept_id''
                        ],
    # 'person.csv': [
    #                 'gender_concept_id',
    #                 'race_concept_id',
    #                 'ethnicity_concept_id',
    #                 'location_id']
}
DATA_DICT_DF = pd.read_csv(DATA_PATH + '/data_dictionary.csv').loc[:, ['concept_id', 'concept_name', 'table']]
CONCEPT_ID_TO_NAME_MAP = DATA_DICT_DF.loc[:, ['concept_id', 'concept_name']].set_index('concept_id').to_dict()['concept_name']
CONCEPT_ID_TO_TABLE_MAP = DATA_DICT_DF.loc[:, ['concept_id', 'table']].set_index('concept_id').to_dict()['table']

''' Public Functions '''
def get_unique_pid_list(path=TRAIN_PATH):
    """
    Gets unique list of patient IDs from person.csv
    :returns: list
    """
    df = pd.read_csv(path + '/person.csv')
    pid_list = df['person_id'].unique().tolist()
    return pid_list

def get_concept_list_ordered_by_sparsity(path=TRAIN_PATH, most_dense_first=True):
    """
    Gets unique list of concept_ids from concept_summary.csv ordered by unique_pid_count, avg_per_pid
    :returns: list (sorted)
    """
    # get df (each row = concept)
    df = generate_concept_summary(path, save_csv=False)
    # sort dataframe
    asc = not most_dense_first # most_dense_first=True means asc=False, and vice-versa
    df_sorted = df.sort_values(['unique_pid_count', 'avg_per_pid'], axis=0, ascending=asc)
    return df_sorted.loc[:, 'concept_id'].tolist()

def get_concept_list_ordered_by_correlation(path, n=None, highest_correlation_first=True):
    """
    Get list of concept_ids and pd.Series of correlation magnitudes sorted by highest-correlation to goldstandard.csv
    :param n: Returns the first n concept_ids
    :returns:
        1) list (sorted concept_ids by highest magnitude correlation (i.e. abs(correlation)))
        2) pd.Series (unsorted series of raw correlation values, with concept_ids as indices)
    """
    # get DataFrame based on concept_id_list from given path
    concept_id_list = get_concept_list_ordered_by_sparsity(path)
    concept_feature_id_map = get_concept_feature_id_map(concept_id_list)
    df_features = create_feature_df(concept_feature_id_map, path=path)

    # join goldstandard.csv as a column
    try:
        df_gold_standard = pd.read_csv(path+'/goldstandard.csv')
    except:
        raise FileNotFoundError(f"Error: file {path + '/goldstandard.csv'} does not exist.")
    df_gold_standard = df_gold_standard.set_index('person_id')
    df_merged = df_features.join(df_gold_standard)
    
    # get correlation matrix. Using: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    df_corr = df_merged.corr() # takes ~1 minute to run on my machine

    # take last column ('status' from goldstandard.csv). Remove NaN and correlation w/ itself. Row indices (representing feature_id) are preserved!
    corr_series = df_corr.iloc[:, -1].dropna().drop('status')

    # take absolute value and sort. Get corresponding indices in-order
    asc = not highest_correlation_first # highest_correlation_first=True means asc=False
    sorted_corr_series = abs(corr_series).sort_values(ascending=asc)
    sorted_feature_ids = list(sorted_corr_series.index)

    # convert from feature_id back to concept_id, then return as list
    feature_concept_id_map = {v:k for k, v in concept_feature_id_map.items()} # reverse the dict
    sorted_concept_ids = [feature_concept_id_map[fid] for fid in sorted_feature_ids]
    
    # re-index sorted_corr_series so it corresponds to concept_id (instead of feature_id)
    sorted_corr_series.index = pd.Index(sorted_concept_ids)

    if n != None and n <= len(sorted_concept_ids):
        return sorted_concept_ids, sorted_corr_series
    else:
        return sorted_concept_ids[:n], sorted_corr_series[:n]

def get_highest_correlation_concept_feature_id_map(n=None, specific_path=None):
    """
    Finds the highest-correlation features using the Pearson Coefficient.
    Returns a dict mapping the n highest correlating concept_ids to feature_ids AND the corresponding correlation magnitudes
    If no n value is provided, then the entire original feature set is provided in sorted order (desc)
        NOTE: the value of each feature is the _count_ of a feature for a given patient.
    :param n: Upper-bound for the number of features to include in the map. This just slices the full df at the end
    :param specific_path: Specific data path to inspect for features
        Otherwise default is get correlation from TRAIN_PATH and EVAL_PATH, and then average the results
    :return: dict (int->int), pd.Series
    """
    # handle specific_path case
    if specific_path != None:
        concept_id_list, corr_series = get_concept_list_ordered_by_correlation(specific_path, n)
    # take average correlation between TRAIN_PATH and EVAL_PATH values of a feature
    else:
        train_cid_list, train_corr_series = get_concept_list_ordered_by_correlation(TRAIN_PATH, n)
        eval_cid_list, eval_corr_series = get_concept_list_ordered_by_correlation(EVAL_PATH, n)
        unique_cid_set = set(train_cid_list) | set(eval_cid_list)
        
        # aggregate as list of (avg_correlation, concept_id) tuples
        cid_tup_list = []
        for cid in unique_cid_set:
            # cases: only in eval set, only in train set, or in both sets
            if cid not in train_corr_series.index:
                cid_tup_list.append((eval_corr_series[cid], cid))
            elif cid not in eval_corr_series.index:
                cid_tup_list.append((train_corr_series[cid], cid))
            else:
                avg_corr = (eval_corr_series[cid] + train_corr_series[cid]) / 2
                cid_tup_list.append((avg_corr, cid))
        
        # sort list and parse-out concept_ids
        sorted_cid_tup_list = sorted(cid_tup_list, reverse=True)
        concept_id_list = [cid for _, cid in sorted_cid_tup_list]
        corr_series = pd.Series([corr for corr, _ in sorted_cid_tup_list])
        if n != None:
            concept_id_list = concept_id_list[:n]
            corr_series = corr_series[:n]
 
    # Generate feature_id map and return 
    concept_feature_id_map = get_concept_feature_id_map(concept_id_list)
    return concept_feature_id_map, corr_series

def create_feature_df(concept_feature_id_map,  path=TRAIN_PATH, count_impute_strategy=0.0):
    """
    Generates the feature DataFrame of shape m x n, with m=len(pid_list) and n=# of features
        NOTE: the value of each feature is the _count_ of a feature for a given patient.
            Need to add additional OMOP-specific logic to parse-out values for concepts with values (e.g. heart rate)
        If a patient is missing a feature, the given inpute_strategy will be used
        By default this pulls all concepts sourced from csv columns specified in FILENAME_CLIN_CONCEPT_MAP
    :param concept_feature_id_map: dict mapping concept_id->feature_id
    :param count_impute_strategy: str in {'most_frequent', 'mean', 'median'} OR any numeric (impute constant)
    :return: DataFrame
    """
    # handle input edge cases
    if type(concept_feature_id_map) != dict:
        raise TypeError(f"Error: concept_feature_id_map needs to be a dict, got: {type(concept_feature_id_map)}")

    # get concept indices, person list, and concept_id->feature_id dict
    concept_id_list = concept_feature_id_map.keys()
    pid_list = get_unique_pid_list(path=path)

    # get concept_id counts as a DataFrame (column names: ['person_id', 'concept_id', 'sum'])
    df_all_summed = get_pid_concept_counts(path, concept_id_list)

    # generate matrix (rows=person_id, cols=feature_id, values=sum)
    get_feature_id = lambda cid: concept_feature_id_map[cid]
    rows = df_all_summed.index.to_list()
    cols = df_all_summed.loc[:, 'concept_id'].apply(get_feature_id).to_list()
    vals = df_all_summed.loc[:, 'sum'].to_list()
    m = len(pid_list)
    n = len(concept_id_list)
    df_sparse = coo_matrix((vals, (rows, cols)), shape=(m, n))
    arr_dense = df_sparse.toarray()
    
    # impute data (by columns)
    arr_imputed = impute_missing_data_univariate(arr_dense, missing_val=0.0, strategy=count_impute_strategy)

    # normalize data (by columns)
    arr_norm = normalize(arr_imputed, axis=0, norm='max') # from: https://stackoverflow.com/a/44257532

    # return as DataFrame
    df_norm = pd.DataFrame(arr_norm)
    df_norm.index.name = 'person_id'
    return df_norm

def generate_concept_summary(path=TRAIN_PATH, save_csv=False, specific_concept_id_list=None):
    """
    Gets a summary of concept_id-person_id pairs as a DataFrame with the following columns:
        concept_id
        concept_name (if in data_dictionary.csv)
        avg_per_pid (if a patient had the concept, how many instances were there)
        from_table (if in data_dictionary.csv)
        unique_pid_count (how many unique patients had the concept)
    :param save_csv: If True, saves concept_summary.csv to DATA_PATH directory
    :param specific_concept_id_list: list of specific concept ids to generate the summary from
    :returns: DataFrame
    """
    # get all concept_id-person_id pairs
    df_all = get_concept_pid_pairs(path, specific_concept_id_list)

    # Get count of unique person_id per concept_id
    df_all_summary = df_all.drop_duplicates(keep='first')\
        .groupby(['concept_id'])\
        .agg({'person_id': 'count'})\
        .rename(columns={'person_id': 'unique_pid_count'})\
        .sort_values('unique_pid_count', ascending=False)
    # # add concept_name, table labels from data_dict
    # # # names
    concept_ids = df_all_summary.reset_index().loc[:, 'concept_id']
    cid_to_name = lambda cid: CONCEPT_ID_TO_NAME_MAP[cid] if cid in CONCEPT_ID_TO_NAME_MAP else pd.NA
    concept_names = concept_ids.apply(cid_to_name).rename('concept_name')
    concept_ids_with_names = pd.concat([concept_ids, concept_names], axis=1).set_index('concept_id')
    df_all_summary.insert(0, "concept_name", concept_ids_with_names)
    # # # table
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
        .agg(['mean'])\
        .reset_index()

    # # clean-up formatting a bit
    df_all_avg.columns = df_all_avg.columns.droplevel(level=[1,2]) # remove multiindex
    df_all_avg = df_all_avg.loc[:, 'avg_per_pid'] # keep only avg_per_pid
    # # add to df_all_summary
    df_all_summary.insert(1, "avg_per_pid", df_all_avg)

    # save to csv
    if save_csv:
        df_all_summary.to_csv(DATA_PATH + '/concept_summary.csv')

    return df_all_summary.reset_index()

''' "Private" Functions '''
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

def get_concept_feature_id_map(specific_concept_id_list=None):
    """
    Generates dict mapping concept_id to feature_id (0-indexed)
    :return: dict (str->int)
    """
    # if no concept_id is provided, then parse through both TRAIN_PATH and EVAL_PATH datasets
    if specific_concept_id_list == None:
        # consider making this every folder in DATA_PATH?
        train_set = set(get_concept_list_ordered_by_sparsity(path=TRAIN_PATH))
        eval_set = set(get_concept_list_ordered_by_sparsity(path=EVAL_PATH))
        concept_id_list = list(train_set | eval_set) # takes set union, then casts to list
    else:
        if type(specific_concept_id_list) != list:
            raise TypeError(f"Error: concept_id_list needs to be a list, got: {type(concept_id_list)}")
        concept_id_list = specific_concept_id_list
    
    return {k: v for v, k in enumerate(concept_id_list)} # from: https://stackoverflow.com/a/36460020

def get_concept_pid_pairs(path=TRAIN_PATH, specific_concept_id_list=None):
    """
    Gets all (non-unique) concept_id-person_id pairs as a DataFrame with the following columns:
        person_id
        concept_id
    To get unique, use pd.DataFrame.drop_duplicates after calling this function

    :param specific_concept_id_list: whitelist of concept_ids (int) to use when aggregating
    :returns: DataFrame
    """
    # use map to quickly access copy of df
    fn_to_df_map = load_csvs_to_dataframe_dict(path=path)

    # Get all person_id, concept_id occurrences (duplicates included) as a DataFrame
    # # estimate df size (assume worst case each person has 1 instance of every concept)
    person_count = len(fn_to_df_map['person.csv']['person_id'])
    clin_concept_count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            clin_concept_count += len(df[col].unique())

    # # init df_all
    idx = range(person_count * clin_concept_count)
    cols = ['person_id', 'concept_id']
    df_all = pd.DataFrame(index=idx, columns=cols)

    # # populate df_all (unique person_id per concept_id)
    count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            # pre-process: get all person_id, concept_id pairs (non-unique)
            df_sliced = df.loc[:, ['person_id', col]]
            df_sliced = df_sliced.dropna()
            df_sliced = df_sliced.rename(columns={col: 'concept_id'})
            # filter-out based on specific_concept_id_list if provided
            if specific_concept_id_list != None:
                df_sliced = df_sliced.loc[df_sliced.loc[:, 'concept_id'].isin(specific_concept_id_list)]
            # set appropriate index
            n_rows = len(df_sliced)
            idx = pd.Series(range(count, count+n_rows), dtype=int)
            df_sliced = df_sliced.set_index(idx) 

            # append to df_all
            df_all.iloc[idx, :] = df_sliced
            count += n_rows
    ## remove NaN
    df_all = df_all.dropna().astype('int')
    return df_all

def get_pid_concept_counts(path=TRAIN_PATH, specific_concept_id_list=None):
    """
    Aggregates all person_id, concept_id occurrences in the specified path and concept_id_list,
        and then sums-up the counts.
    :returns: DataFrame (column names: ['person_id', 'concept_id', 'sum'])
    """
    # get unique person_id, concept_id pairs for concept_ids in concept_id_list
    df_all = get_concept_pid_pairs(path, specific_concept_id_list)

    # for each person_id, get corresponding concept_id, count pairs
    # # get counts
    tot = len(df_all)
    df_ones = pd.DataFrame(np.ones((tot, 1)))
    df_all_w_ones = pd.concat([df_all, df_ones], axis=1)
    df_all_summed = df_all_w_ones.groupby(['concept_id', 'person_id'])\
        .agg(['sum'])\
        .reset_index()\
        .set_index('person_id')
    # # remove column multiindex, rename column
    df_all_summed.columns = df_all_summed.columns.droplevel(level=[1])
    df_all_summed = df_all_summed.rename(columns={df_all_summed.columns[1]: 'sum'})

    return df_all_summed

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

''' "Deprecated" Functions '''
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