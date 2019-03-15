import numpy as np

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def anneal(min_val, max_val, t, anneal_len):
    """"Anneal linearly from min_val to max_val over anneal_len."""
    if t >= anneal_len:
        return max_val
    else:
        return (max_val - min_val) * t/anneal_len
