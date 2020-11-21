import unittest
import etl
import numpy as np
import pandas as pd

class TestFeatureETL(unittest.TestCase):
    def test_feature_filtering(self):
        # helper fn
        def run_test(concept_id_list):
            concept_feature_id_map = etl.get_concept_feature_id_map(concept_id_list)
            df = etl.create_feature_df(concept_feature_id_map)

            # check if headers match initial list
            feature_id_list = [concept_feature_id_map[cid] for cid in concept_feature_id_map]
            self.assertTrue(feature_id_list == list(df.columns.values))
         
        # Test densely populated concepts
        dense_concept_id_list = [4033240, 44818518, 44818702]
        run_test(dense_concept_id_list)

        # Test sparsely populated concepts
        sparse_concept_id_list = [3037684, 4057277, 2003223]
        run_test(sparse_concept_id_list)

        # Test fake concept_ids -- columns return as all 0's
        bogus_concept_id_list = [1111111111111111, 2222222222222222]
        run_test(bogus_concept_id_list)
    
    def test_concept_feature_map_generation(self):
        # Test aggregate both train+eval into feature map
        concept_feature_id_map = etl.get_concept_feature_id_map()

        # Test filtering of features (just tests that it is different)
        sorted_concept_feature_id_map, sorted_corr_series = etl.get_highest_corr_concept_feature_id_map_and_corr_series()
        self.assertTrue(concept_feature_id_map != sorted_concept_feature_id_map)
        # print(sorted_corr_series)

        # Test picking top n features
        top_10_concept_feature_id_map, top_10_corr_series = etl.get_highest_corr_concept_feature_id_map_and_corr_series(n=10)
        self.assertTrue(len(top_10_concept_feature_id_map) == 10)
        self.assertTrue(len(top_10_corr_series) == len(sorted_corr_series))
        # print(top_10_corr_series)

        # Re-run concept_summary generation
        _ = etl.generate_concept_summary(save_csv=True)

        # Save concept_feature_id_map as a csv
        concepts = pd.Series(sorted_concept_feature_id_map.keys())
        df_corr = pd.concat([concepts, sorted_corr_series], axis=1)
        df_corr.to_csv(etl.DATA_PATH + '/concept_correlation.csv')

    def test_concept_list_generation(self):
        # Test sparsity cutoff
        cid_list_by_sparsity = etl.get_concept_list_ordered_by_sparsity(etl.TRAIN_PATH)
        cid_list_by_sparsity_cutoff = etl.get_concept_list_ordered_by_sparsity(sparsity_cutoff=0.5)

        self.assertTrue(len(cid_list_by_sparsity) > len(cid_list_by_sparsity_cutoff))

        # Test correlation list
        cid_list_by_correlation_TRAIN, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.TRAIN_PATH)
        cid_list_by_correlation_EVAL, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.EVAL_PATH)

        self.assertTrue(set(cid_list_by_correlation_TRAIN) == set(cid_list_by_sparsity))
        self.assertTrue(len(cid_list_by_correlation_TRAIN) != len(cid_list_by_correlation_EVAL)) # weak test using valid assumption

        # Test passing specific concept_id list to get correlations (just checking compilation)
        dense_concept_id_list = [4033240, 44818518, 44818702]
        sparse_concept_id_list = [3037684, 4057277, 2003223]
        _, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.TRAIN_PATH, specific_concept_id_list=dense_concept_id_list)
        _, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.TRAIN_PATH, specific_concept_id_list=sparse_concept_id_list)
        _, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.EVAL_PATH, specific_concept_id_list=dense_concept_id_list)
        _, _ = etl.get_concept_list_and_corr_series_ordered_by_correlation(etl.EVAL_PATH, specific_concept_id_list=sparse_concept_id_list)
    
    def test_parsed_value_df_generation(self):
        parsed_aggregate_strategies = ['mean', 'median', 'sum']
        parsed_impute_strategies = ['mean', 'median', 'most_frequent', 0.0]
        paths = [etl.TRAIN_PATH, etl.EVAL_PATH]
        # Test different usage scenarios
        for path in paths:
            for agg in parsed_aggregate_strategies:
                for imp in parsed_impute_strategies:
                    df_parsed = etl.get_parsed_values_df(path=path, val_aggregate_strategy=agg, val_impute_strategy=imp)
                    cid_list, corr_series = etl.get_concept_list_and_corr_series_ordered_by_correlation(path, parsed_aggregate_strategy=agg, parsed_impute_strategy=imp)
                    cf_map, corr_series = etl.get_highest_corr_concept_feature_id_map_and_corr_series(path, parsed_aggregate_strategy=agg, parsed_impute_strategy=imp)
                    df_fin = etl.create_feature_df(cf_map, path=path, parsed_aggregate_strategy=agg, parsed_impute_strategy=imp)

if __name__ == '__main__':
    unittest.main()