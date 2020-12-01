import pandas as pd
import numpy as np
from os import getcwd
from sklearn.experimental import enable_iterative_imputer # https://stackoverflow.com/a/56738037
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from datetime import date, datetime
from config import TRAIN_PATH, EVAL_PATH, FILENAME_LIST, FILENAME_CLIN_CONCEPT_MAP, AGG_IMP_CONFIG, INCLUDE_PARSED_VALUES

''' Public Functions '''
def get_unique_pid_list(path=TRAIN_PATH):
    """
    Gets unique list of patient IDs from person.csv
    """
    df = pd.read_csv(path + '/person.csv')
    pid_list = df['person_id'].unique().tolist()
    return pid_list

def get_unique_cid_list(path=TRAIN_PATH, include_parsed_values=INCLUDE_PARSED_VALUES):
    """
    Gets unique list of concept IDs from path
    """
    concept_id_list = get_concept_pid_pairs_df(path).loc[:, 'concept_id'].drop_duplicates().tolist()
    if include_parsed_values:
        parsed_df = get_parsed_values_df(path)
        concept_id_list += list(parsed_df.columns)
    return concept_id_list

def get_concept_feature_id_map(specific_path=None, specific_concept_id_list=None, include_parsed_values=INCLUDE_PARSED_VALUES):
    """
    Generates dict mapping concept_id to feature_id (0-indexed)
    """
    if specific_concept_id_list == None:
        concept_id_list = get_unique_cid_list_from_TRAIN_and_EVAL(include_parsed_values) if specific_path == None else get_unique_cid_list(specific_path, include_parsed_values)
    else:
        if type(specific_concept_id_list) != list:
            raise TypeError(f"Error: concept_id_list needs to be a list, got: {type(concept_id_list)}")
        concept_id_list = (dict.fromkeys(specific_concept_id_list)).keys() # from: https://stackoverflow.com/a/39835527
    return {k: v for v, k in enumerate(concept_id_list)} # from: https://stackoverflow.com/a/36460020

def get_highest_corr_concept_feature_id_map_and_corr_series(specific_path=None, keep_first_n=None, use_parsed_values=INCLUDE_PARSED_VALUES, agg_imp_config=AGG_IMP_CONFIG):
    """
    Finds the highest-correlation features using the Pearson Coefficient and generates a concept_feature_id_map
    """
    if specific_path != None:
        concept_id_list, corr_series = get_concept_list_and_corr_series_ordered_by_correlation(path=specific_path, \
            use_parsed_values=use_parsed_values, agg_imp_config=agg_imp_config)
    else:
        train_cid_list, train_corr_series = get_concept_list_and_corr_series_ordered_by_correlation(path=TRAIN_PATH, \
            use_parsed_values=use_parsed_values, agg_imp_config=agg_imp_config)
        eval_cid_list, eval_corr_series = get_concept_list_and_corr_series_ordered_by_correlation(path=EVAL_PATH, \
            use_parsed_values=use_parsed_values, agg_imp_config=agg_imp_config)
        unique_cid_set = set(train_cid_list) | set(eval_cid_list)
        
        concept_id_list, corr_series = average_train_and_eval_cid_corr_series(unique_cid_set, train_corr_series, eval_corr_series)

    if keep_first_n != None:
        keep_first_n = int(keep_first_n)
        concept_id_list = concept_id_list[:keep_first_n]
        corr_series = corr_series[concept_id_list]

    concept_feature_id_map = get_concept_feature_id_map(specific_path, concept_id_list)
    return concept_feature_id_map, corr_series

def get_parsed_values_df(path=TRAIN_PATH, agg_imp_config=AGG_IMP_CONFIG):
    """
    Parses-out discrete from measurement.csv, observation.csv, and person.csv
    Resulting df is row-indexed by patient_id and column-indexed by ~concept_id~
    """
    _, val_aggregate_strategy, val_impute_strategy = unpack_config_var(agg_imp_config)
    pid_list = get_unique_pid_list(path=path)
    fn_to_df_dict = load_csvs_to_dataframe_dict(fn_list=['measurement.csv', 'observation.csv', 'person.csv'], path=path)

    rows, cols, vals = [], [], []
    append_value_as_number_to_rows_cols_vals('measurement_concept_id', fn_to_df_dict['measurement.csv'], val_aggregate_strategy, rows, cols, vals)
    append_value_as_number_to_rows_cols_vals('observation_concept_id', fn_to_df_dict['observation.csv'], val_aggregate_strategy, rows, cols, vals)
    first_half_df = generate_first_half_of_parsed_values_df(rows, cols, vals, pid_list, val_impute_strategy) # from: https://stackoverflow.com/a/44257532

    rows, cols, vals = [], [], []
    append_value_as_concept_id_to_rows_cols_vals('measurement_concept_id', fn_to_df_dict['measurement.csv'], rows, cols, vals)
    append_value_as_concept_id_to_rows_cols_vals('observation_concept_id', fn_to_df_dict['observation.csv'], rows, cols, vals)
    append_abnormal_counts_to_rows_cols_vals(fn_to_df_dict, rows, cols, vals)
    append_person_age_to_rows_cols_vals(fn_to_df_dict, rows, cols, vals)
    second_half_df = generate_second_half_of_parsed_values_df(rows, cols, vals, pid_list)

    merged_df = first_half_df.join(second_half_df)
    return merged_df

def create_feature_df(concept_feature_id_map,  path=TRAIN_PATH, use_parsed_values=INCLUDE_PARSED_VALUES, agg_imp_config=AGG_IMP_CONFIG):
    """
    Generates the feature DataFrame of shape m x n, with m=len(pid_list) and n=# of features
        If a patient is missing a feature, the impute settings from agg_imp_config will be used
        NOTE: if a feature column is only 0.0 values, it will be removed in the final df
    Resulting df is row-indexed by patient_id and column-indexed by ~feature_id~ from concept_feature_id_map
    """
    count_impute_strategy, _, _ = unpack_config_var(agg_imp_config)
    if type(concept_feature_id_map) != dict:
        raise TypeError(f"Error: concept_feature_id_map needs to be a dict, got: {type(concept_feature_id_map)}")

    concept_id_list = concept_feature_id_map.keys()
    pid_list = get_unique_pid_list(path)
    m = len(pid_list)
    n = len(concept_id_list)
    df_all_summed = aggregate_pid_concept_counts_into_df(path, concept_id_list)
    if use_parsed_values:
        df_all_summed = add_parsed_df_to_df_all_summed(path, agg_imp_config, concept_id_list, m, pid_list, df_all_summed)

    df_norm = get_normalized_df_from_df_all_summed(concept_feature_id_map, df_all_summed, m, n, count_impute_strategy)
    return df_norm

''' "Private" Functions '''
def get_concept_list_and_corr_series_ordered_by_correlation(path, specific_concept_id_list=None, use_parsed_values=INCLUDE_PARSED_VALUES, agg_imp_config=AGG_IMP_CONFIG):
    """
    Gets list of concept ids and pd.Series of correlation magnitudes sorted by highest-correlation to "goldstandard.csv"
    NOTE: this drops some concept_ids with N/A correlation value
    """
    concept_id_list = get_unique_cid_list(path, use_parsed_values) if specific_concept_id_list == None else specific_concept_id_list
    concept_feature_id_map = get_concept_feature_id_map(path, concept_id_list)
    feature_df = create_feature_df(concept_feature_id_map, path=path, agg_imp_config=agg_imp_config, use_parsed_values=use_parsed_values)
    try:
        gold_standard_df = pd.read_csv(path + '/goldstandard.csv').set_index('person_id')
    except:
        raise FileNotFoundError(f"Error: file {path + '/goldstandard.csv'} does not exist.")
    merged_df = feature_df.join(gold_standard_df)
    corr_df = merged_df.corr()
    corr_series = corr_df.loc[:, 'status'].dropna().drop('status') # Indices are feature_ids
    sorted_feature_ids = list(abs(corr_series).sort_values(ascending=False).index)
    sorted_concept_ids, corr_series_idx = convert_from_fids_to_cids(concept_feature_id_map, sorted_feature_ids, corr_series)
    corr_series.index = corr_series_idx
    return sorted_concept_ids, corr_series

def get_concept_pid_pairs_df(path=TRAIN_PATH, specific_concept_id_list=None):
    """
    Gets all (person_id, concept_id) pairs as a DataFrame
    """
    fn_to_df_map = load_csvs_to_dataframe_dict(path=path)
    est_max_size = get_n_persons_times_n_concepts(fn_to_df_map)
    est_max_size_idx = range(est_max_size)
    df_all = pd.DataFrame(index=est_max_size_idx, columns=['person_id', 'concept_id'])
    row_count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            df_sliced = df.loc[:, ['person_id', col]].dropna().rename(columns={col: 'concept_id'})
            if specific_concept_id_list != None:
                df_sliced = df_sliced.loc[df_sliced.loc[:, 'concept_id'].isin(specific_concept_id_list)]
            n_rows = len(df_sliced)
            idx = pd.Series(range(row_count, row_count+n_rows), dtype=int)
            df_sliced = df_sliced.set_index(idx) 
            df_all.iloc[idx, :] = df_sliced
            row_count += n_rows
    df_all = df_all.dropna().astype('int')
    return df_all

def aggregate_pid_concept_counts_into_df(path=TRAIN_PATH, specific_concept_id_list=None):
    """
    Aggregates all person_id, concept_id occurrences in the specified path and concept_id_list,
        and then sums-up the counts.
            Indexed by person_id
            Columns: ['concept_id', 'sum']
    """
    df_all = get_concept_pid_pairs_df(path, specific_concept_id_list)
    tot = len(df_all)
    df_ones = pd.DataFrame(np.ones((tot, 1)))
    df_all_w_ones = pd.concat([df_all, df_ones], axis=1)
    df_all_summed = df_all_w_ones.groupby(['concept_id', 'person_id'])\
        .agg(['sum'])\
        .reset_index()\
        .set_index('person_id')
    df_all_summed.columns = df_all_summed.columns.droplevel(level=[1]) # remove multiindex from aggregating
    df_all_summed = df_all_summed.rename(columns={df_all_summed.columns[1]: 'sum'})
    return df_all_summed

def unpack_config_var(conf=AGG_IMP_CONFIG):
    assert len(conf) == 3
    keys = tuple(conf.keys())
    assert keys[0] == 'count_impute_strat'
    assert keys[1] == 'parsed_agg_strat'
    assert keys[2] == 'parsed_impute_strat'
    return tuple(conf.values())

def load_csvs_to_dataframe_dict(fn_list=FILENAME_LIST, path=TRAIN_PATH):
    fn_to_df_dict = {}
    for fn in fn_list:
        try:
            df = pd.read_csv(path + '/' + fn)
            fn_to_df_dict[fn] = df
        except:
            raise ValueError(f"Error: could not read file: {path+'/'+str(fn)}")
    return fn_to_df_dict

def get_n_persons_times_n_concepts(fn_to_df_map):
    person_count = len(fn_to_df_map['person.csv']['person_id'])
    clin_concept_count = 0
    for fn in FILENAME_CLIN_CONCEPT_MAP:
        df = fn_to_df_map[fn]
        for col in FILENAME_CLIN_CONCEPT_MAP[fn]:
            clin_concept_count += len(df[col].unique())
    max_size = person_count * clin_concept_count
    return max_size

def impute_missing_data_univariate(X, missing_val=0.0, strategy='most_frequent'):
    val = None
    if type(strategy) != str:
        try:
            val = float(strategy)
            strategy = 'constant'
        except:
            raise ValueError(f"Error: parameter 'strategy' needs to be string or numeric, got: {strategy}")
    imp = SimpleImputer(missing_values=missing_val, strategy=strategy, fill_value=val)
    X_new = imp.fit_transform(X)
    return X_new

def convert_from_fids_to_cids(concept_feature_id_map, sorted_feature_ids, corr_series):
    fid_to_cid_map = {v:k for k, v in concept_feature_id_map.items()}
    sorted_concept_ids = [fid_to_cid_map[fid] for fid in sorted_feature_ids]
    cid_idx_list = [fid_to_cid_map[fid] for fid in list(corr_series.index)]
    return sorted_concept_ids, pd.Index(cid_idx_list)

def get_unique_cid_list_from_TRAIN_and_EVAL(include_parsed_values):
    train_set = set(get_unique_cid_list(TRAIN_PATH, include_parsed_values))
    eval_set = set(get_unique_cid_list(EVAL_PATH, include_parsed_values))
    concept_id_list = list(train_set | eval_set)
    return concept_id_list

def average_train_and_eval_cid_corr_series(unique_cid_set, train_corr_series, eval_corr_series):
    cid_tup_list = [] # list of (avg_correlation, concept_id) tuples
    for cid in unique_cid_set:
        if cid not in train_corr_series.index:
            cid_tup_list.append((eval_corr_series[cid], cid))
        elif cid not in eval_corr_series.index:
            cid_tup_list.append((train_corr_series[cid], cid))
        else:
            avg_corr = (eval_corr_series[cid] + train_corr_series[cid]) / 2
            cid_tup_list.append((avg_corr, cid))
    sorted_cid_tup_list = sorted(cid_tup_list, reverse=True)
    concept_id_list = [cid for _, cid in sorted_cid_tup_list]
    corr_series = pd.Series([corr for corr, _ in sorted_cid_tup_list])
    corr_series.index = concept_id_list
    return concept_id_list, corr_series

def get_normalized_df_from_df_all_summed(concept_feature_id_map, df_all_summed, m, n, count_impute_strategy):
    get_feature_id = lambda cid: concept_feature_id_map[cid]
    rows = df_all_summed.index.to_list()
    cols = df_all_summed.loc[:, 'concept_id'].apply(get_feature_id).to_list()
    vals = df_all_summed.loc[:, 'sum'].to_list()
    df_sparse = coo_matrix((vals, (rows, cols)), shape=(m, n))
    arr_dense = df_sparse.toarray()
    arr_imputed = impute_missing_data_univariate(arr_dense, missing_val=0.0, strategy=count_impute_strategy)
    arr_norm = normalize(arr_imputed, axis=0, norm='max')
    df_norm = pd.DataFrame(arr_norm)
    df_norm.index.name = 'person_id'
    remove_all_zero_cols_from_df = lambda df: df.loc[:, (df != 0).any(axis=0)] # https://stackoverflow.com/a/21165116
    df_norm = remove_all_zero_cols_from_df(df_norm)
    return df_norm

def add_parsed_df_to_df_all_summed(path, agg_imp_config, concept_id_list, m, pid_list, df_all_summed):
    parsed_df = get_parsed_values_df(path, agg_imp_config)
    concept_overlap = set(concept_id_list).intersection(set(parsed_df.columns))
    parsed_df = parsed_df[concept_overlap]

    max_index_size = m * len(parsed_df.columns)
    reformatted_parsed_df = pd.DataFrame(index=range(max_index_size), columns=['concept_id', 'sum'])
    final_index = []
    row_count = 0
    for pid in range(len(pid_list)):
        curr_pid_vals_as_df = parsed_df.iloc[pid, :].reset_index()
        curr_pid_vals_as_df.columns = ['concept_id', 'sum'] # 'sum' (instead of 'value') used to match original column name
        final_index += [pid] * len(curr_pid_vals_as_df)
        next_row_count = row_count + len(curr_pid_vals_as_df)
        curr_pid_vals_as_df.index = range(row_count, next_row_count)
        reformatted_parsed_df.iloc[row_count:next_row_count, :] = curr_pid_vals_as_df
        row_count = next_row_count
    reformatted_parsed_df.index = final_index
    updated_df_all_summed = pd.concat([df_all_summed, reformatted_parsed_df])
    return updated_df_all_summed

def generate_first_half_of_parsed_values_df(rows, cols, vals, pid_list, val_impute_strategy):
    concept_feature_map = get_concept_feature_id_map(specific_concept_id_list=cols)
    m = len(pid_list)
    n = len(concept_feature_map.keys())
    fids = [concept_feature_map[cid] for cid in cols]
    df_sparse = coo_matrix((vals, (rows, fids)), shape=(m, n))
    arr_dense = df_sparse.toarray()
    arr_imputed = impute_missing_data_univariate(arr_dense, missing_val=0.0, strategy=val_impute_strategy)
    arr_first_half = normalize(arr_imputed, axis=0, norm='max') # from: https://stackoverflow.com/a/44257532
    first_half_df = pd.DataFrame(arr_first_half, index=pid_list, columns=list(concept_feature_map.keys()), dtype=float)
    return first_half_df

def generate_second_half_of_parsed_values_df(rows, cols, vals, pid_list):
    second_concept_feature_map = get_concept_feature_id_map(specific_concept_id_list=cols)
    m = len(pid_list)
    n = len(second_concept_feature_map.keys())
    second_fids = [second_concept_feature_map[cid] for cid in cols]
    df_sparse = coo_matrix((vals, (rows, second_fids)), shape=(m, n))
    arr_dense = df_sparse.toarray()
    arr_second_half = normalize(arr_dense, axis=0, norm='max') # from: https://stackoverflow.com/a/44257532
    second_half_df = pd.DataFrame(arr_second_half, index=pid_list, columns=list(second_concept_feature_map.keys()), dtype=float)
    return second_half_df

def append_person_age_to_rows_cols_vals(fn_to_df_dict, rows, cols, vals):
    # NOTE: For model consistency, hard-coding ref date as 12-31-2020 (as opposed to dynamically getting date)
    str_format = '%Y-%m-%d'
    ref_date = datetime.strptime('2020-12-31', str_format)
    get_age_from_birthday_in_days = lambda d: float((ref_date - datetime.strptime(d, str_format)).days)
    df_ppl = fn_to_df_dict['person.csv'][['person_id', 'birth_datetime']]
    df_age = df_ppl['birth_datetime'].apply(get_age_from_birthday_in_days)
    pid_list = df_ppl.loc[:, 'person_id'].tolist()
    rows += pid_list
    cols += [1234567891011] * len(pid_list) # using random number as age concept_id (1234567891011)
    vals += df_age.tolist()

def append_abnormal_counts_to_rows_cols_vals(fn_to_df_dict, rows, cols, vals):
    df_abn = fn_to_df_dict['measurement.csv'][['person_id', 'measurement_concept_id', 'value_as_number', 'range_low', 'range_high']]
    df_abn.insert(2, "abnormal_count", df_abn.iloc[:, 2:].apply(convert_formatted_row_to_abonrmal_value_count, axis=1), True)
    df_abn = df_abn.iloc[:, :3]
    df_abn_grouped = df_abn.groupby(['person_id', 'measurement_concept_id']).agg(['sum'])
    df_abn_grouped.columns = list(map('_'.join, df_abn_grouped.columns.values)) # https://stackoverflow.com/a/26325610
    df_abn_grouped = df_abn_grouped.reset_index()
    rows += df_abn_grouped.loc[:, 'person_id'].tolist()
    cols += [pad_digits(cid, digit='1') for cid in df_abn_grouped.loc[:, 'measurement_concept_id'].tolist()]
    vals += df_abn_grouped.loc[:, 'abnormal_count_sum'].tolist()

def pad_digits(cid, digit=0, final_len=10):
    d = str(digit)
    s = str(cid)
    while len(s) < final_len:
        s += d
    return int(s)

def convert_formatted_row_to_abonrmal_value_count(row):
    val, low, high = row.tolist()
    abnormal = 0.0
    nan_set = {'nan', '<NA>'}
    if str(low) not in nan_set and val < low:
        abnormal = 1.0
    if str(high) not in nan_set and val > high:
        abnormal = 1.0
    return abnormal

def concat_ints_from_lists_element_wise(first_list, second_list):
    assert len(first_list) == len(second_list)
    first_list = [str(int(v)) for v in first_list]
    second_list = [str(int(v)) for v in second_list]
    res = [first_list[i] + second_list[i] for i in range(len(first_list))]
    return [int(v) for v in res]

def append_value_as_concept_id_to_rows_cols_vals(concept_id_name, table_df, rows, cols, vals):
    vals_df = table_df[['person_id', concept_id_name, 'value_as_concept_id']].dropna()
    vals_df['new_concept_id'] = concat_ints_from_lists_element_wise(vals_df[concept_id_name].tolist(), vals_df['value_as_concept_id'].tolist())
    vals_df = vals_df.reset_index(drop=True)
    tot = len(vals_df)
    df_ones = pd.DataFrame(np.ones((tot, 1))).rename(columns={0:'count'})
    vals_df = pd.concat([vals_df.loc[:, ['person_id', 'new_concept_id']], df_ones], axis=1)
    df_grouped = vals_df.groupby(['person_id', 'new_concept_id'])\
        .agg(['sum'])
    df_grouped.columns = list(map('_'.join, df_grouped.columns.values)) # https://stackoverflow.com/a/26325610
    df_grouped = df_grouped.reset_index()
    rows += df_grouped.loc[:, 'person_id'].tolist()
    cols += df_grouped.loc[:, 'new_concept_id'].tolist()
    vals += df_grouped.loc[:, 'count_sum'].tolist()

def append_value_as_number_to_rows_cols_vals(concept_id_name, table_df, val_aggregate_strategy, rows, cols, vals):
    floor = 1 # used to ensure 0.0 values in data are preserved. Floor gets normalized at the end so no biggie
    vals_df = table_df[['person_id', concept_id_name, 'value_as_number']].dropna()
    vals_df['value_as_number'] = vals_df['value_as_number'].apply(lambda v: v + floor) 
    df_grouped = vals_df.groupby(['person_id', concept_id_name])\
        .agg([val_aggregate_strategy])
    df_grouped.columns = list(map('_'.join, df_grouped.columns.values)) # https://stackoverflow.com/a/26325610
    df_grouped = df_grouped.reset_index()
    df_grouped.loc[:, concept_id_name] = df_grouped.loc[:, concept_id_name].apply(pad_digits)
    rows += df_grouped.loc[:, 'person_id'].tolist()
    cols += df_grouped.loc[:, concept_id_name].tolist()
    vals += df_grouped.loc[:, 'value_as_number_' + val_aggregate_strategy].tolist()

''' "Deprecated" Functions '''
def impute_missing_data_multivariate(X, missing_val=0.0, strategy='most_frequent', max_iter=5):
    """
    Imputes missing values in X using multivariate approach (MICE, paper here: https://www.jstatsoft.org/article/view/v045i03)
    """
    if type(strategy) in {int, float}:
        strategy='constant'
    RANDOM_SEED = 420 # "random"
    imp = IterativeImputer(random_state=RANDOM_SEED, missing_values=missing_val, initial_strategy=strategy, max_iter=max_iter)
    X_new = imp.fit_transform(X)
    return X_new

def generate_concept_summary(path=TRAIN_PATH, save_csv=False, specific_concept_id_list=None):
    """
    Gets a summary of how frequently concept ids appear in the dataset as a DataFrame and/or csv
    NOTE: Unfortunately, this does not incorporate parsed values
    """
    df_all = get_concept_pid_pairs_df(path, specific_concept_id_list)
    df_all_summary = df_all.drop_duplicates(keep='first')\
        .groupby(['concept_id'])\
        .agg({'person_id': 'count'})\
        .rename(columns={'person_id': 'unique_pid_count'})\
        .sort_values('unique_pid_count', ascending=False)

    add_concept_name_and_table_columns_to_df_all_summary(df_all_summary)
    df_avg_per_pid = get_avg_counts_per_pid_from_concept_pid_pairs_df(df_all)
    df_all_summary.insert(1, "avg_per_pid", df_avg_per_pid)
    if save_csv:
        df_all_summary.to_csv(DATA_PATH + f'/concept_summary_{str(date.today())}.csv')

    return df_all_summary.reset_index()

def get_concept_list_ordered_by_sparsity(path=TRAIN_PATH, desc=True, sparsity_cutoff=None):
    """
    Gets unique list of concept ids from concept_summary.csv ordered by unique pid count (population density), average occurrences per pid when present
    NOTE: Unfortunately, this does not incorporate parsed values
    """
    df = generate_concept_summary(path, save_csv=False)
    asc = not desc
    df_sorted = df.sort_values(['unique_pid_count', 'avg_per_pid'], axis=0, ascending=asc)
    if sparsity_cutoff != None:
        cutoff = float(sparsity_cutoff) * len(get_unique_pid_list(path))
        df_sorted = df_sorted[df_sorted.unique_pid_count >= cutoff]
    return df_sorted.loc[:, 'concept_id'].tolist()

def add_concept_name_and_table_columns_to_df_all_summary(df_all_summary):
    data_dict_df = pd.read_csv(DATA_PATH + '/data_dictionary.csv').loc[:, ['concept_id', 'concept_name', 'table']]
    cid_to_name_map = data_dict_df.loc[:, ['concept_id', 'concept_name']].set_index('concept_id').to_dict()['concept_name']
    cid_to_table_map = data_dict_df.loc[:, ['concept_id', 'table']].set_index('concept_id').to_dict()['table']

    concept_ids = df_all_summary.reset_index().loc[:, 'concept_id']
    cid_to_name = lambda cid: cid_to_name_map[cid] if cid in cid_to_name_map else pd.NA
    concept_names = concept_ids.apply(cid_to_name).rename('concept_name')
    concept_ids_with_names = pd.concat([concept_ids, concept_names], axis=1).set_index('concept_id')
    df_all_summary.insert(0, "concept_name", concept_ids_with_names)

    cid_to_table = lambda cid: cid_to_table_map[cid] if cid in cid_to_table_map else pd.NA
    concept_table = concept_ids.apply(cid_to_table).rename('from_table')
    concept_ids_with_table = pd.concat([concept_ids, concept_table], axis=1).set_index('concept_id')
    df_all_summary.insert(1, "from_table", concept_ids_with_table)

def get_avg_counts_per_pid_from_concept_pid_pairs_df(df_all):
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
    df_all_avg.columns = df_all_avg.columns.droplevel(level=[1,2]) # remove multiindex
    df_avg_per_pid = df_all_avg.loc[:, 'avg_per_pid']
    return df_avg_per_pid

''' Command Line Tool '''
def get_vars_based_on_cid_file_path(cid_file_path):
    file_ext_is_dot_txt = lambda fn: fn[-4:] == '.txt'
    timestamp = str(date.today())
    if cid_file_path != None:
        assert(file_ext_is_dot_txt(basename(cid_file_path)))
        cid_filename_without_ext = basename(cid_file_path)[:-4]
        cid_file = open(cid_file_path)
        cid_list = [int(line) for line in cid_file]
        df_base_path = results_dir + f'/df_features_USING_{cid_filename_without_ext}_{timestamp}'
        cf_map_base_path = results_dir + f'/cf_map_USING_{cid_filename_without_ext}_{timestamp}'
    else:
        cid_list = None
        df_base_path = results_dir + f'/df_features_{timestamp}'
        cf_map_base_path = results_dir + f'/cf_map_{timestamp}'
    return cid_list, df_base_path, cf_map_base_path

if __name__ == '__main__':
    import argparse
    import pickle
    from os.path import basename

    parser = argparse.ArgumentParser(description='Converts OMOP-formatted csv files into feature DataFrame for training. Saves as pickle files and csvs to specified directory.')
    parser.add_argument('--data_dir', type=str, required=True, help='Filepath to directory containing OMOP-formatted .csv files (e.g. "~/Documents/data_dir")')
    parser.add_argument('--results_dir', type=str, required=True, help='Filepath to directory to save the output .pickle and .csv files (e.g. "~/Documents/results_dir")')
    parser.add_argument('--cid_list_file', type=str, required=False, help='Filepath to .txt file with concept_ids to use (e.g. "~/Documents/data_dir/cid_list.txt")')
    args_dict = vars(parser.parse_args())
    
    # Convert inputs to appropriate variables
    data_dir = args_dict['data_dir']
    results_dir = args_dict['results_dir']
    cid_file_path = args_dict['cid_list_file']
    cid_list, df_base_path, cf_map_base_path = get_vars_based_on_cid_file_path(cid_file_path)

    df_pickle_filepath = df_base_path + '.pickle'
    df_csv_filepath = df_base_path + '.csv'
    cf_map_pickle_filepath = cf_map_base_path + '.pickle'
    cf_map_csv_filepath = cf_map_base_path + '.csv'

    # Run ETL
    cf_map = get_concept_feature_id_map(specific_path=data_dir, specific_concept_id_list=cid_list)
    feature_df = create_feature_df(cf_map, path=data_dir)

    # Save as .pickle files
    pickle.dump(feature_df, open(df_pickle_filepath, 'wb'))
    pickle.dump(cf_map, open(cf_map_pickle_filepath, 'wb'))

    # Save as .csv files
    feature_df.to_csv(path_or_buf=df_csv_filepath)
    cf_map_as_df = pd.DataFrame(cf_map.items())
    cf_map_as_df.columns = ['concept_id', 'feature_id']
    cf_map_as_df.to_csv(path_or_buf=cf_map_csv_filepath, index=False)