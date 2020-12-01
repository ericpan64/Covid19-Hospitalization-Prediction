import unittest
import etl
import numpy as np
import pandas as pd

PATHS = [etl.TRAIN_PATH, etl.EVAL_PATH]

class TestFeatureETL(unittest.TestCase):
    def test_pid_and_cid_list_generation(self):
        list_is_unique = lambda l: len(l) == len(set(l))
        first_is_shorter_than_second = lambda x, y: len(x) < len(y)
        for path in PATHS:
            unique_pid_list = etl.get_unique_pid_list(path)
            self.assertTrue(list_is_unique(unique_pid_list))

            unique_cid_list_no_parsed_vals = etl.get_unique_cid_list(path, include_parsed_values=False)
            unique_cid_list_with_parsed_vals = etl.get_unique_cid_list(path)
            self.assertTrue(list_is_unique(unique_cid_list_no_parsed_vals))
            self.assertTrue(list_is_unique(unique_cid_list_with_parsed_vals))
            self.assertTrue(first_is_shorter_than_second(unique_cid_list_no_parsed_vals, unique_cid_list_with_parsed_vals))

    def test_feature_id_processing(self):
        def assert_that_feature_ids_are_preserved(concept_id_list, should_be_all_zeros=False):
            cf_map = etl.get_concept_feature_id_map(specific_concept_id_list=concept_id_list)
            feature_ids_in_cf_map_is_superset_of_df_columns = lambda cf_map, df: set([cf_map[cid] for cid in cf_map]).issuperset(set(df.columns.values))
            df_is_all_zeros = lambda df: (df == 0.0).all().all()
            for path in PATHS:
                df = etl.create_feature_df(cf_map, path)
                self.assertTrue(feature_ids_in_cf_map_is_superset_of_df_columns(cf_map, df))

                if should_be_all_zeros:
                    self.assertTrue(df_is_all_zeros(df))
                else:
                    self.assertFalse(df_is_all_zeros(df))
         
        dense_concept_id_list = [4033240, 44818518, 44818702]
        assert_that_feature_ids_are_preserved(dense_concept_id_list)

        sparse_concept_id_list = [3037684, 4057277, 2003223]
        assert_that_feature_ids_are_preserved(sparse_concept_id_list)

        bogus_concept_id_list = [1111111111111111, 2222222222222222]
        assert_that_feature_ids_are_preserved(bogus_concept_id_list, should_be_all_zeros=True)

    def test_concept_feature_map_generation(self):
        ids_in_first_are_superset_of_second = lambda x, y: set(x.keys()).issuperset(set(y.keys()))
        objects_have_same_length = lambda x, y: len(x) == len(y)
        dict_keys_match_series_index = lambda d, s: set(d.keys()) == set(s.index)
        object_length_matches_n = lambda x, n: len(x) == n
        def run_tests(path):
            cf_map = etl.get_concept_feature_id_map(path)
            sorted_cf_map, corr_series = etl.get_highest_corr_concept_feature_id_map_and_corr_series(path)
            self.assertTrue(ids_in_first_are_superset_of_second(cf_map, sorted_cf_map))
            self.assertTrue(objects_have_same_length(sorted_cf_map, corr_series))
            self.assertTrue(dict_keys_match_series_index(sorted_cf_map, corr_series))
            
            n = 10
            top_n_sorted_cf_map, top_n_corr_series = etl.get_highest_corr_concept_feature_id_map_and_corr_series(path, keep_first_n=n)
            self.assertTrue(object_length_matches_n(top_n_sorted_cf_map, n))
            self.assertTrue(object_length_matches_n(top_n_corr_series, n))
            self.assertTrue(dict_keys_match_series_index(top_n_sorted_cf_map, top_n_corr_series))
        
        run_tests(None)
        for path in PATHS:
            run_tests(path)

    def test_parsed_value_df_generation(self):
        conf = {
            'count_impute_strat': 0.0, # options: {'mean', 'median', 'most_frequent', 0.0}
            'parsed_agg_strat': 'mean',  # options: {'mean', 'median', 'sum'}
            'parsed_impute_strat': 'most_frequent',  # options: {'mean', 'median', 'most_frequent', 0.0}
        }
        sorted_objects_are_equal = lambda x, y: sorted(x) == sorted(y)
        sorted_objects_are_NOT_equal = lambda x, y: sorted(x) != sorted(y)
        ids_in_first_are_superset_of_second = lambda x, y: set(x).issuperset(set(y))
        dfs_have_same_indices = lambda dfa, dfb: (dfa.index == dfb.index).all()
        dfs_are_equal_on_given_columns = lambda dfa, dfb, col: dfa[col].equals(dfb[col])
        
        for path in PATHS:
            cf_map, corr_series_from_mapfn = etl.get_highest_corr_concept_feature_id_map_and_corr_series(path, agg_imp_config=conf)
            cid_list, corr_series_from_listfn = etl.get_concept_list_and_corr_series_ordered_by_correlation(path, agg_imp_config=conf)
            self.assertTrue(sorted_objects_are_equal(corr_series_from_mapfn, corr_series_from_listfn))
            self.assertTrue(sorted_objects_are_equal(cid_list, list(cf_map.keys())))

            parsed_df = etl.get_parsed_values_df(path=path, agg_imp_config=conf)
            full_df = etl.create_feature_df(cf_map, path=path, agg_imp_config=conf, use_parsed_values=True)
            orig_df = etl.create_feature_df(cf_map, path=path, agg_imp_config=conf, use_parsed_values=False)
            self.assertTrue(dfs_have_same_indices(parsed_df, full_df))
            self.assertTrue(dfs_have_same_indices(orig_df, full_df))

            fc_map = {v:k for k, v in cf_map.items()}
            parsed_concepts = set(parsed_df.columns)
            tot_concepts = set([fc_map[fid] for fid in full_df.columns])
            orig_concepts = set([fc_map[fid] for fid in orig_df.columns])
            parsed_concepts_in_tot = tot_concepts.difference(orig_concepts)

            self.assertTrue(ids_in_first_are_superset_of_second(tot_concepts, orig_concepts))
            self.assertTrue(ids_in_first_are_superset_of_second(parsed_concepts, parsed_concepts_in_tot))
            self.assertTrue(sorted_objects_are_NOT_equal(parsed_concepts, orig_concepts))

            orig_features = orig_df.columns
            parsed_features_in_tot = [cf_map[cid] for cid in parsed_concepts_in_tot]
            parsed_part_of_full_df = full_df[parsed_features_in_tot]
            parsed_part_of_full_df.columns = parsed_df[parsed_concepts_in_tot].columns
            self.assertTrue(dfs_are_equal_on_given_columns(full_df, orig_df, orig_features))
            self.assertTrue(dfs_are_equal_on_given_columns(parsed_part_of_full_df, parsed_df, parsed_concepts_in_tot))
            
if __name__ == '__main__':
    print("\nRunning ETL tests... (takes ~5 minutes)")
    unittest.main()