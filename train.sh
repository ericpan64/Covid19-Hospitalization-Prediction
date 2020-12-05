
#!/usr/bin/env bash
DATA_DIR='/data'
RESULTS_DIR='/model'
CID_LIST_FILE='/model/CustomIdList.txt'
USE_CORR='False'
USE_PCA='False'

python3 /app/etl.py --data_dir $DATA_DIR --results_dir $RESULTS_DIR --cid_list_file $CID_LIST_FILE --order_cf_map_by_corr $USE_CORR
python3 /app/train.py --data_dir $DATA_DIR --results_dir $RESULTS_DIR --cid_list_file $CID_LIST_FILE --use_pca $USE_PCA
