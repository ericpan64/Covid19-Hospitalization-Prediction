import sklearn.metrics as slm

def classification_metrics(Y_true, Y_pred):
    """
    Returns classification metrics for evaluation. Deliberately ordered based on importance
    Y_true is BINARY
    Y_pred is BINARY
    """
    # Get scores
    accuracy = slm.balanced_accuracy_score(Y_true, Y_pred)
    auroc_value = slm.roc_auc_score(Y_true, Y_pred)
    precision = slm.precision_score(Y_true, Y_pred) # TP vs. TP+FP (predicted P)
    recall = slm.recall_score(Y_true, Y_pred) # recall == sensitivity, TP vs. TP+FN (actually P)
    f1 = slm.f1_score(Y_true, Y_pred) # want max, balances precision vs. recall
    return auroc_value,accuracy,precision,recall,f1