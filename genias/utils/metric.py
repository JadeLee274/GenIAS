import re
from sklearn.metrics import precision_recall_curve, auc
from genias.utils.common_import import *
from genias.utils.utils import *
from genias.data_factory.loader import *
from genias.models.carla import *


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
) -> Tuple[float, float, float]:
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
        Precision, recall, (micro) F1 score.
    """
    assert (len(tp_list) == len(fp_list)) and(len(fp_list) == len(fn_list)), \
    'Lengths of tp_list, fp_list, fn_list mismatch'
    sum_tp = sum(tp_list)
    sum_fp = sum(fp_list)
    sum_fn = sum(fn_list)
    precision = (sum_tp) / (sum_tp + sum_fp)
    recall = (sum_tp) / (sum_tp + sum_fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def macro_f1(f1_list: List[float]) -> float:
    """
    For dataset with more than one train/test datasets, it gives the average 
    of F1 scores from each datasets. This is called macro F1 score.

    Parameters:
        f1_list: The list of F1 scores from each test data.

    Returns:
        Macro F1 score.     
    """
    assert len(f1_list) > 1, 'This function is for dataset with multiple data' 
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


def inference(
    dataset: str,
    gpu_num: int,
    carla_scheme: str,
    time: str,
    seed: int = 42,
) -> None:
    """
    If you only want to infer scores with pre-trained models, by using this 
    function, you can get informations on F1 scores, AUC-PR.

    Parameters:
        dataset: Name of dataset.
        gpu_num: The inference will be done on this GPU.
        scheme:  What scheme was applied for pretext.
        time:    The start time of the model train & inference process.
        seed:    The seed will be fixed. Default 42.
    """
    assert carla_scheme in [
        'carla_original', 'genias', 'mix', 'genias_multiple'
    ], "'carla_original', 'genias', 'mix', 'genias_multiple"

    fix_seed_all(seed=seed)

    device = torch.device(f'cuda:{gpu_num}')

    best_f1_list = []
    best_tp_list = []
    best_fp_list = []
    best_fn_list = []
    auc_pr_list = []

    ckpt_path = os.path.join(
        'genias', 'checkpoints', 'classification', dataset
    )

    if dataset in ['MSL', 'SMAP', 'SMD', 'KPI', 'Yahoo-A1']:
        data_list = sorted(os.listdir(f'data/{dataset}/test'))
        data_list = [f.replace('.npy', '') for f in data_list]

        for subdata in data_list:
            test_set = ClassificationDataset(
                dataset=dataset,
                subdata=subdata,
                mode='test',
                scheme=carla_scheme,
            )
            data_dim = test_set.data_dim
            model = ClassificationModel(in_channels=data_dim)
            ckpt_path = os.path.join(ckpt_path, subdata, carla_scheme)
            ckpt = torch.load(os.path.join(ckpt_path, f'{time}.pt'))
            model.load_state_dict(ckpt['model'])
            model = model.to(device)
            model.eval()

            logits = []

            for test_data in test_set:
                test_data = torch.tensor(test_data).float().to(device)
                logit = model.forward(test_data.unsqueeze(0).transpose(-2, -1))
                logit = logit.squeeze(0).detach().cpu().numpy()
                logits.append(logit)
            
            classes = [0 for _ in range(10)]

            for logit in logits:
                max_idx = np.argmax(logit)
                classes[max_idx] += 1
            
            major_class = classes.index(max(classes))

            anomaly_scores = []

            for logit in logits:
                major_probability = logit[major_class]
                anomaly_scores.append(1 - major_probability)
            
            anomaly_scores = np.array(anomaly_scores)

            precision, recall, thresholds = precision_recall_curve(
                y_true=test_set.labels,
                y_score=anomaly_scores,
            )

            auc_pr = auc(recall, precision)

            best_threshold = 0
            best_f1 = 0

            for i in range(len(thresholds)):
                f1_score = f1score(precision[i], recall[i])
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_threshold = thresholds[i]
            
            best_anomaly_prediction = np.where(
                anomaly_scores >= best_threshold, 1, 0
            )

            best_f1_score, best_tp, best_fp, best_fn = f1_stat(
                prediction=best_anomaly_prediction,
                gt=test_set.labels,
            )

            best_f1_list.append(best_f1_score)
            best_tp_list.append(best_tp)
            best_fp_list.append(best_fp)
            best_fn_list.append(best_fn)
            auc_pr_list.append(auc_pr)
    
    else:
        test_set = ClassificationDataset(
            dataset=dataset,
            mode='test',
            scheme=carla_scheme,
        )
        data_dim = test_set.data_dim
        model = ClassificationModel(in_channels=data_dim)
        ckpt_path = os.path.join(ckpt_path, subdata, carla_scheme)
        ckpt = torch.load(os.path.join(ckpt_path, f'{time}.pt'))
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        model.eval()

        logits = []

        for test_data in test_set:
            test_data = torch.tensor(test_data).float().to(device)
            logit = model.forward(test_data.unsqueeze(0).transpose(-2, -1))
            logit = logit.squeeze(0).detach().cpu().numpy()
            logits.append(logit)
        
        classes = [0 for _ in range(10)]

        for logit in logits:
            max_idx = np.argmax(logit)
            classes[max_idx] += 1
        
        major_class = classes.index(max(classes))

        anomaly_scores = []

        for logit in logits:
            major_probability = logit[major_class]
            anomaly_scores.append(1 - major_probability)
        
        anomaly_scores = np.array(anomaly_scores)

        precision, recall, thresholds = precision_recall_curve(
            y_true=test_set.labels,
            y_score=anomaly_scores,
        )

        auc_pr = auc(recall, precision)

        best_threshold = 0
        best_f1 = 0

        for i in range(len(thresholds)):
            f1_score = f1score(precision[i], recall[i])
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = thresholds[i]
        
        best_anomaly_prediction = np.where(
            anomaly_scores >= best_threshold, 1, 0
        )

        best_f1_score, best_tp, best_fp, best_fn = f1_stat(
            prediction=best_anomaly_prediction,
            gt=test_set.labels,
        )

        best_f1_list.append(best_f1_score)
        best_tp_list.append(best_tp)
        best_fp_list.append(best_fp)
        best_fn_list.append(best_fn)
        auc_pr_list.append(auc_pr)

    best_f1_list = np.array(best_f1_list)
    best_tp_list = np.array(best_tp_list)
    best_fp_list = np.array(best_fp_list)
    best_fn_list = np.array(best_fn_list)
    auc_pr_list = np.array(auc_pr_list)

    f1_score_best = np.max(best_f1_list)
    precision, recall, f1_micro = mirco_f1(
        tp_list=best_tp_list,
        fp_list=best_fp_list,
        fn_list=best_fn_list
    )    
    auc_pr_mean = np.mean(auc_pr_list)
    auc_pr_std = np.std(auc_pr_list)
    f1_macro = macro_f1(f1_list=best_f1_list)

    print(f'- Best F1: {round(f1_score_best, 4)}')
    print(f'- Micro F1: {round(f1_micro, 4)}')
    print(f'- Precision: {round(precision, 4)}')
    print(f'- Recall: {round(recall, 4)}')
    print(f'- AUC-PR mean: {round(auc_pr_mean, 4)}')
    print(f'- AUC-PR std: {round(auc_pr_std, 4)}')
    print(f'- Macro F1: {round(f1_macro, 4)}')

    return


def check_log(log_file_path: str) -> None:
    """
    Calculates scores of merged log file.

    Parameters:
        log_file_path: The path of merged log file.

    If train and inference stopped for some reason, then you can restart the 
    process from train/test subdata or epoch where it stopped, and merge the 
    log files of before- and after- restart. 

    This function helps calculating the final score of the entire process by 
    reading the merged log file. It prints the best f1, micro f1, precision, 
    recall, auc-pr mean & std, macro f1.

    Make sure that the final line of merged log file ('Scores' part) is erased.
    """

    best_f1_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    auc_list = []


    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Best F1' in line:
                best_f1_list.append(
                    float(re.search(r": \s*([0-9]+(?:\.[0-9]+)?)", line).group(1))
                )
            elif 'True Positives' in line:
                tp_list.append(
                    int(re.search(r": \s*([0-9]+(?:\.[0-9]+)?)", line).group(1))
                )
            elif 'False Positives' in line:
                fp_list.append(
                    int(re.search(r": \s*([0-9]+(?:\.[0-9]+)?)", line).group(1))
                )
            elif 'False Negatives' in line:
                fn_list.append(
                    int(re.search(r": \s*([0-9]+(?:\.[0-9]+)?)", line).group(1))
                )
            elif 'AUC-PR' in line:
                auc_list.append(
                    float(re.search(r": \s*([0-9]+(?:\.[0-9]+)?)", line).group(1))
                )
            elif 'Scores' in line:
                raise ValueError("Make sure that you omit Score part")

    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    micro_f1 = (2 * precision * recall) / (precision + recall)
    macro_f1 = np.mean(best_f1_list)

    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)

    print('Scores')
    print(f'- Best F1: {round(max(best_f1_list), 4)}')
    print(f'- Micro F1: {round(micro_f1, 4)}')
    print(f'- Precision: {round(precision, 4)}')
    print(f'- Recall: {round(recall, 4)}')
    print(f'- AUC-PR mean: {round(auc_mean, 4)}')
    print(f'- AUC-std: {round(auc_std, 4)}')
    print(f'- Macro F1: {round(macro_f1, 4)}')

    return


def point_adjustment(pred: Vector, gt: Vector) -> Vector:
    assert len(pred) == len(gt), \
    f"length of pred {len(pred)} and length of gt {len(gt)} mismatch"

    for i in range(len(pred)):
        if pred[i] == 1 and gt[i] == 1:
            for j in range(i, len(gt)):
                if pred[j] == 0 and gt[j] == 1:
                    pred[j] = 1
                elif pred[j] == 1:
                    break
            for k in range(i, 0, -1):
                if pred[k] == 0 and gt[k] == 1:
                    pred[k] = 1
                elif pred[k] == 1:
                    break
    
    return pred


####################### Metrics for anomaly generation ####################### 


