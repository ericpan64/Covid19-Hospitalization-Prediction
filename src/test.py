import pickle

def prepare_data():
    #Set Paths
    EVAL_PATH = "/data"

    #Prep Data
    print("Preparing Data")
    
    #Read In Feature List
    with open('../model/feature_dict.pickle', 'rb') as feature_dict:
        feature_id_map = pickle.load(feature_dict)
    
    return feature_id_map

if __name__ == "__main__":
    X = prepare_data()
    print(X)
