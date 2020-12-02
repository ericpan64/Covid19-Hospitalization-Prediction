with open("features.txt", "r") as feature_list:
    features = [int(line.rstrip('\n')) for line in feature_list]
feature_cols = list(range(0, len(features)))
concept_feature_id_map = {k: v for k, v in zip(features, feature_cols)}