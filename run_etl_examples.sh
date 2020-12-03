python3 src/etl.py --data_dir './data/DREAM_data/training' --results_dir './data/DREAM_ETL_results'
python3 src/etl.py --data_dir './data/DREAM_data/training' --results_dir './data/DREAM_ETL_results' --order_cf_map_by_corr 'True'
python3 src/etl.py --data_dir './data/DREAM_data/training' --results_dir './data/DREAM_ETL_results' --cid_list_file './data/DREAM_ETL_results/test_cids.txt'
python3 src/etl.py --data_dir './data/DREAM_data/training' --results_dir './data/DREAM_ETL_results' --cid_list_file './data/DREAM_ETL_results/test_cids.txt' --order_cf_map_by_corr 'True'