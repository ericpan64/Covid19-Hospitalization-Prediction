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

        # test aggregate both train+eval into feature map
        concept_feature_id_map = etl.get_concept_feature_id_map()

        # test filtering of features (just tests that it is different)
        sorted_concept_feature_id_map, sorted_corr_series = etl.get_highest_correlation_concept_feature_id_map()
        self.assertTrue(concept_feature_id_map != sorted_concept_feature_id_map)
        # print(sorted_corr_series)

        # test picking top n features
        top_10_concept_feature_id_map, top_10_corr_series = etl.get_highest_correlation_concept_feature_id_map(n=10)
        self.assertTrue(len(top_10_concept_feature_id_map) == 10)
        self.assertTrue(len(top_10_corr_series) == 10)
        # print(top_10_corr_series)

        # Re-run concept_summary generation
        etl.generate_concept_summary(save_csv=True)
        # Save concept_feature_id_map as a csv
        concepts = pd.Series(sorted_concept_feature_id_map.keys())
        df_corr = pd.concat([concepts, sorted_corr_series], axis=1)
        df_corr.to_csv(etl.DATA_PATH + '/concept_correlation.csv')

if __name__ == '__main__':
    unittest.main()