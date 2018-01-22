from sklearn.metrics import roc_auc_score

def score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)