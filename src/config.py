''' ETL Config Variables '''
TRAIN_PATH ='./data/DREAM_data/training' # path running from dir containing src folder
EVAL_PATH = './data/DREAM_data/evaluation'
FILENAME_LIST = ['condition_occurrence.csv', 'device_exposure.csv', 'goldstandard.csv', 
    'measurement.csv', 'observation_period.csv', 'observation.csv', 
    'person.csv', 'procedure_occurrence.csv', 'visit_occurrence.csv']
FILENAME_CLIN_CONCEPT_MAP = {
    'condition_occurrence.csv': ['condition_concept_id'],
    'device_exposure.csv': ['device_concept_id'],
    'measurement.csv': ['measurement_concept_id',
                        'measurement_type_concept_id'],
    'observation.csv': ['observation_concept_id',
                        'qualifier_concept_id'],
    'procedure_occurrence.csv': ['procedure_concept_id',
                                'modifier_concept_id'],
    'visit_occurrence.csv': ['visit_concept_id'],
    # 'person.csv': [
    #                 'gender_concept_id',
    #                 'race_concept_id',
    #                 'ethnicity_concept_id',
    #                 'location_id']
}
AGG_IMP_CONFIG = {
    'count_impute_strat': 0.0, # options: {'mean', 'median', 'most_frequent', 0.0}
    'parsed_agg_strat': 'mean',  # options: {'mean', 'median', 'sum'}
    'parsed_impute_strat': 'mean',  # options: {'mean', 'median', 'most_frequent', 0.0}
}
'''
"Parsed values" introduce the custom concept_ids from the following .csvs and schema:
    measurement.csv
        value_as_number: original concept_id padded with 0's to 10 digits
        value_as_concept_id: original concept_id appended with value concept_id (will be >10 digits)
        abnormal (range_low/range_high): original concept_id padded with 1's to 10 digits
    observation.csv:
        value_as_number: same as above
        value_as_concept_id: same as above
    person.csv:
        person age (birthdate): custom concept_id is 1234567891011. Yes this was arbitrarily picked
'''
INCLUDE_PARSED_VALUES=True