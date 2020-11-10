import unittest
import etl

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

if __name__ == '__main__':
    unittest.main()