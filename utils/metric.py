from utils.common_import import *


######################### Metrics for CARLA inference ######################### 


def f1score(precision: float, recall: float) -> float:
    """
    Calculate F1 score for given precision and recall.

    Parameters:
        precision: Precision.
        reacll:    Recall.

    Returns:
        F1 score.
    """
    return (2 * precision * recall) / (precision + recall)


def f1_stat(prediction: Vector, gt: Vector) -> Tuple[float, int, int, int]:
    """
    Calculates F1 score, number of true positives, number of false positives,
    and number of false negatives.

    Parameters:
        prediction: The vector consisting of 0 and 1. 0 is for data point that
                    model detected as normal, and 1 is for data point that 
                    model detected as abnormal.
        gt:         True label.
    
    Returns:
        F1 score, number of true positives, number of false positives, and 
        number of false negatives.
    """
    assert len(prediction) == len(gt), 'Lengths of prediction and gt mismatch'
    num_tp = 0
    num_fp = 0
    num_fn = 0

    for i in range(len(prediction)):
        if prediction[i] == 1 and gt[i] == 1:
            num_tp += 1
        elif prediction[i] == 1 and gt[i] == 0:
            num_fp += 1
        elif prediction[i] == 0 and gt[i] == 1:
            num_fn += 1
    
    precision = (num_tp) / (num_tp + num_fp)
    recall = (num_tp) / (num_tp + num_fn)
    f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score, num_tp, num_fp, num_fn 


def mirco_f1(
    tp_list: List[int],
    fp_list: List[int],
    fn_list: List[int],
) -> float:
    """
    For dataset that consists of more than one train/test datasets, it first
    sums up all total true positives, false positives, and false negatives. 
    Based on these three valuse, it calculates F1 score. This is called micro 
    F1 score, which is widely used metric across anomaly detection study.

    Parameters:
        tp_list: The list of number of true positives from each test data.
        fp_list: The list of number of false positives from each test data.
        fn_list: The list of number of false negatives from each test data.

    Returns:
        Micro F1 score.
    """
    assert (len(tp_list) == len(fp_list)) and(len(fp_list) == len(fn_list)), \
    'Lengths of tp_list, fp_list, fn_list mismatch'
    sum_tp = sum(tp_list)
    sum_fp = sum(fp_list)
    sum_fn = sum(fn_list)
    precision = (sum_tp) / (sum_tp + sum_fp)
    recall = (sum_tp) / (sum_tp + sum_fn)

    return (2 * precision + recall) + (precision + recall)


def macro_f1(f1_list: List[float]) -> float:
    """
    For dataset with more than one train/test datasets, it gives the average 
    of F1 scores from each datasets. This is called macro F1 score.

    Parameters:
        f1_list: The list of F1 scores from each test data.

    Returns:
        Macro F1 score.     
    """
<<<<<<< HEAD
    assert len(f1_list) > 1, 'This funcation is for dataset with multiple data' 
=======
    assert len(f1_list) > 1, \
    'This funcation is for dataset with multiple data' 
>>>>>>> main
    return round(np.mean(f1_list), 4)


def auc_pr_statistics(auc_pr_list: List[float]) -> Tuple[float, float]:
    """
    For dataset with more than one train/test datasets, it gives the mean and
    standard deviation of AUC-PR scores from test datasets.

    Parameters:
        auc_pr_list: List of AUC-PR scores collected from test datasets.

    Returns:
        Mean and standard deviation of AUC-PR scores.
    """
    auc_pr_mean = round(np.mean(auc_pr_list), 4)
    auc_pr_std = round(np.std(auc_pr_list), 4)
    return auc_pr_mean, auc_pr_std


####################### Metrics for anomaly generation ####################### 


