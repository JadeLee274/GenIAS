import csv
import argparse
import logging
from math import cos, pi
from torch.utils.data import DataLoader
import torch.optim as optim 
from faiss import IndexFlatL2
from sklearn.metrics import precision_recall_curve, auc
from utils.common_import import *
from data_factory.loader import *
from carla.model import *
from utils.loss import pretextloss, classificationloss, entropy
from utils.metric import *
from utils.fix_seed import fix_seed_all


def str2bool(v: str) -> bool:
    """
    Changes string to bool.

    Parameters:
        v: String. Must be either 'True' or 'False'.
    """
    assert v in ['True', 'False'], "string must be either 'True' or 'False'"

    return v.lower() in 'true'


def cosine_schedule(
    optimizer: optim.Adam,
    current_epoch: int,
    total_epochs: int = 30,
    initial_learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.01,
) -> None:
    """
    Customized cosine scheduler. Updates optimizer's learning rate.

    Parameters:
        optimizer:             Adam.
        current_epoch:         Current training epoch.
        total_epochs:          Total training epochs. Default 30.
        initial_learning_rate: Initial learning rate. Defalut 1e-3.
        lr_dacay_rate:         Decay rate of initial learning rate. 
                               Default 0.01.
    """
    eta_min = initial_learning_rate * (lr_decay_rate ** 3)
    scheduled_learning_rate = eta_min \
    + (initial_learning_rate - eta_min) \
    * (1 + cos(pi * current_epoch / total_epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduled_learning_rate

    return


def pretext(
    dataset: str,
    subdata: str,
    use_genias: bool = False,
    epochs: int = 30,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    gpu_num: int = 0,
    model_save_interval: int = 5,
    num_neighbors: int = 2,
) -> None:
    """
    Uses Resnet model and mlp head to map anchor, positive pair, and negative
    pair to the representation space (with dimension 128, in this case).

    While training, the pretext loss is optimized so that the distance between
    the anchor and the positive pair get smaller, while that of
    the anchor and the negative pair get larger, in the representation space.

    The model is saved once in a model_save_interval epochs, in order to be
    used for the self-supervised stage of CARLA.
    """
    assert dataset in ['MSL_SEPARATED', 'SMAP_SEPARATED'], \
    "dataset must be one of 'MSL_SEPARATED', 'SMAP_SEPARATED'"

    logging.info('Pretext training loop start...\n')

    train_dataset = PretextDataset(
        dataset=dataset,
        subdata=subdata,
        use_genias=use_genias,
    )

    model = PretextModel(
        in_channels=train_dataset.data_dim,
        mid_channels=4,
    )

    device = torch.device(f'cuda:{gpu_num}')

    model = model.to(device)
    criterion = pretextloss()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} start.')

        cosine_schedule(optimizer=optimizer, current_epoch=epoch)
        epoch_loss = 0.0
        prev_loss = None

        for batch in train_loader:
            optimizer.zero_grad()
            anchor, positive_pair, negative_pair = batch
            B, W, F = anchor.shape
            anchor = anchor.to(device)
            positive_pair = positive_pair.to(device)
            negative_pair = negative_pair.to(device)

            triplets = torch.cat(
                tensors=[anchor, positive_pair, negative_pair],
                dim=0
            ).float()

            triplets = triplets.view(3 * B, F, W)
            representations = model(triplets)
            loss = criterion(
                representations=representations,
                current_loss=prev_loss,
            )
            loss.backward()
            optimizer.step()
            prev_loss = loss.item()
            epoch_loss += prev_loss
        
        epoch_loss /= len(train_loader)
        logging.info(f'Epoch {epoch + 1} train loss: {epoch_loss:.4e}')

        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'resnet': model.resnet.state_dict(),
                    'contrastive_head': model.contrastive_head.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(
                    log_dir, 'model_pretext', f'{subdata.replace(
                        '.npy', ''
                        )}_epoch_{epoch + 1}.pt'
                    )
            )

    logging.info(f'{subdata} loop done. Start selecting neighborhoods...\n')
    model.eval()

    timeseries_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    anchor_reps = []
    negative_reps = []

    print('Loading representations of each anchor and its negative pair.')

    for batch in timeseries_loader:
        anchor, _, negative_pair = batch
        anchor = anchor.to(device).float().transpose(-2, -1)
        negative_pair = negative_pair.to(device).float().transpose(-2, -1)
        anchor_rep = model.forward(anchor).detach().cpu()
        negative_rep = model.forward(negative_pair).detach().cpu()
        anchor_reps.append(anchor_rep)
        negative_reps.append(negative_rep)
    
    anchor_reps = torch.cat(anchor_reps, dim=0).numpy()
    negative_reps = torch.cat(negative_reps, dim=0).numpy()
    
    reps = np.concatenate([anchor_reps, negative_reps], axis=0)

    index_searcher = IndexFlatL2(reps.shape[1])
    assert index_searcher.d == reps.shape[1],\
    f'{index_searcher.d} != {reps.shape[1]}'
    index_searcher.add(reps)

    anchor_and_negative_pairs = np.concatenate(
        [train_dataset.anchors, train_dataset.negative_pairs],
        axis=0,
    )

    classification_data_dir = os.path.join(
        'classification_dataset', dataset
    )
    os.makedirs(os.path.join(
        classification_data_dir, 'anchor_nn'
        ), exist_ok=True)
    os.makedirs(os.path.join(
        classification_data_dir, 'anchor_fn'
        ), exist_ok=True
    )
    os.makedirs(os.path.join(
        classification_data_dir, 'negative_nn'
        ), exist_ok=True
    )
    os.makedirs(os.path.join(
        classification_data_dir, 'negative_fn'
        ), exist_ok=True)

    print(f'\nSelecting top-{num_neighbors} neighbors of the anchor...')

    nearest_neighbors = []
    furthest_neighbors = []

    for anchor_rep in anchor_reps:
        anchor_query = anchor_rep.reshape(1, -1)
        _, indices = index_searcher.search(anchor_query, reps.shape[0])
        indices = indices.reshape(-1)
        nearest_indices = indices[1:num_neighbors + 1]
        furthest_indices = indices[-num_neighbors:]
        nearest_neighbors.append(
            anchor_and_negative_pairs[nearest_indices]
        )
        furthest_neighbors.append(
            anchor_and_negative_pairs[furthest_indices]
        )
    
    print('\nSaving nearest neighborhoods of the anchor...')
    nearest_neighbors = np.array(nearest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'anchor_nn', subdata
        ),
        arr=nearest_neighbors,
    )

    print('Saving furthest neighborhoods of the anchor...')
    furthest_neighbors = np.array(furthest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'anchor_fn', subdata
        ),
        arr=furthest_neighbors,
    )

    print(f'\nSelecting top-{num_neighbors} neighbors of the negative pair...')

    nearest_neighbors = []
    furthest_neighbors = []

    for negative_rep in negative_reps:
        negative_query = negative_rep.reshape(1, -1)
        _, indices = index_searcher.search(negative_query, reps.shape[0])
        indices = indices.reshape(-1)
        nearest_indices = indices[1:num_neighbors + 1]
        furthest_indices = indices[-num_neighbors:]
        nearest_neighbors.append(
            anchor_and_negative_pairs[nearest_indices]
        )
        furthest_neighbors.append(
            anchor_and_negative_pairs[furthest_indices]
        )
    
    print('\nSaving nearest neighborhoods of the anchor...')
    nearest_neighbors = np.array(nearest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'negative_nn', subdata
        ),
        arr=nearest_neighbors,
    )

    print('Saving furthest neighborhoods of the anchor...')
    furthest_neighbors = np.array(furthest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'negative_fn', subdata
        ),
        arr=furthest_neighbors,
    )

    print('\nPretext stage done. Moving on to classification stage.\n')

    return


def classification(
    dataset: str,
    subdata: str,
    use_genias: bool = False,
    gpu_num: int = 0,
    epochs: int = 100,
    batch_size: int = 50,
    learning_rate: float = 1e-2,
    model_save_interval: int = 5,
) -> Tuple[float, int, int, int, float]:
    print(f'Classification on {dataset} {subdata} start...')
    device = torch.device(f'cuda:{gpu_num}')

    train_dataset = ClassificationDataset(
        dataset=dataset,
        subdata=subdata,
        mode='train',
        use_genias=use_genias
    )
    data_dim = train_dataset.data_dim
    model = ClassificationModel(in_channels=data_dim)

    if use_genias:
        ckpt_dir = f'temp_checkpoints/carla/use_genias/{dataset}/pretext/{subdata}'
        ckpt = torch.load(os.path.join(ckpt_dir, 'epoch_30.pt'))
    else:
        ckpt_dir = f'temp_checkpoints/carla/without_genias/{dataset}/pretext/{subdata}'
        ckpt = torch.load(os.path.join(ckpt_dir, 'epoch_30.pt'))

    model.resnet.load_state_dict(ckpt['resnet'])
    model = model.to(device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    criterion = classificationloss()

    if use_genias:
        save_dir = f'temp_checkpoints/carla/use_genias/{dataset}/classification/{subdata}'
        log_dir = f'temp_log/carla/use_genias/{dataset}'
    else:
        save_dir = f'temp_checkpoints/carla/without_genias/{dataset}/classification/{subdata}'
        log_dir = f'temp_log/carla/without_genias/{dataset}'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'{subdata}_classification.csv')
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','consistenty','inconsistency','entropy','total'])

    print(f'Classification training loop start...\n')
    model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_inconsistency_loss = 0.0
        epoch_entropy_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=device)

            window, nearest_neighbor, furthest_neighbor = batch

            window = window.to(device).float()
            nearest_neighbor = nearest_neighbor.to(device).float()
            furthest_neighbor = furthest_neighbor.to(device).float()

            window = window.transpose(-2, -1)
            window_logit = model.forward(window)

            entropy_loss = entropy(torch.mean(window_logit, 0))
            batch_loss -= entropy_loss
            epoch_entropy_loss += entropy_loss.item()

            batch_consistency_sum = torch.zeros(1, device=device)
            batch_consistency = 0.0
            batch_inconsistency = 0.0

            for i in range(nearest_neighbor.shape[1]):
                nearest = nearest_neighbor[:, i].transpose(-2, -1)
                furthest = furthest_neighbor[:, i].transpose(-2, -1)

                nearest_logit = model.forward(nearest)
                furthest_logit = model.forward(furthest)

                consistency_sum, consistency, inconsistency \
                = criterion(
                    window_logit=window_logit,
                    nearest_logit=nearest_logit,
                    furthest_logit=furthest_logit,
                )
                
                batch_consistency_sum += consistency_sum
                batch_consistency += consistency
                batch_inconsistency += inconsistency
            
            batch_loss += batch_consistency_sum

            epoch_consistency_loss += batch_consistency
            epoch_inconsistency_loss += batch_inconsistency

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        epoch_consistency_loss /= len(train_loader)
        epoch_inconsistency_loss /= len(train_loader)
        epoch_entropy_loss /= len(train_loader)
        epoch_loss /= len(train_loader)
        
        logging.info(f'Epoch {epoch + 1} finished.')
        logging.info(f'- Consistency loss: {epoch_consistency_loss:.4e}')
        logging.info(f'- Inconsistency loss: {epoch_inconsistency_loss:.4e}')
        logging.info(f'- Entropy loss: {epoch_entropy_loss:.4e}')
        logging.info(f'- Total loss: {epoch_loss:.4e}\n')


        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(
                    log_dir, 'model_classification', f'{subdata}_epoch_{epoch + 1}.pt'
                    ),
                )
        
        epoch_consistency_loss /= len(train_loader)
        epoch_inconsistency_loss /= len(train_loader)
        epoch_entropy_loss /= len(train_loader)
        epoch_loss /= len(train_loader)
        
        print(f'- Consistency loss: {epoch_consistency_loss:.4e}')
        print(f'- Inconsistency loss: {epoch_inconsistency_loss:.4e}')
        print(f'- Entropy loss: {epoch_entropy_loss:.4e}')
        print(f'- Total loss: {epoch_loss:.4e}\n')

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                    epoch+1,
                    round(epoch_consistency_loss, 4),
                    round(epoch_inconsistency_loss, 4),
                    round(epoch_entropy_loss, 4),
                    round(epoch_loss, 4)
                ]
            )

        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(save_dir, f'epoch_{epoch + 1}.pt'),
            )
    
    print('Classification training done.')

    model.eval()
    print(f'\nStarting inference on {dataset} {subdata}...')

    test_dataset = ClassificationDataset(
        dataset=dataset,
        subdata=subdata,
        mode='test',
        use_genias=use_genias,
    )
    test_data = torch.tensor(
        data=test_dataset.data,
        dtype=torch.float32,
    )
    test_data = test_data.unsqueeze(1).transpose(-2, -1).to(device)
    labels = test_dataset.label.reshape(-1)
    
    logits = model.forward(test_data)
    logits = logits.detach().cpu()

    classes = [0 for _ in range(10)]

    for i in range(len(logits)):
        logit = logits[i]
        max_index = np.argmax(logit)
        classes[max_index] += 1

    major_class = classes.index(max(classes))

    anomaly_labels = []
    anomaly_scores = []

    for i in range(len(logits)):
        logit = logits[i]
        major_probability = logit[major_class]
        
        if np.argmax(logit) == major_class:
            anomaly_labels.append(1)
        else:
            anomaly_labels.append(0)
        
        anomaly_scores.append(1 - major_probability)

    anomaly_labels = np.array(anomaly_labels)
    anomaly_scores = np.array(anomaly_scores)

    precision, recall, thresholds = precision_recall_curve(
        y_true=labels,
        y_score=anomaly_scores,
    )

    auc_pr = auc(recall, precision)

    best_threshold = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0

    for i in range(len(thresholds)):
        f1_score = f1score(precision[i], recall[i])
        if f1_score > best_f1:
            best_f1 = f1_score
            best_precision = precision[i]
            best_recall = recall[i]
            best_threshold = thresholds[i]

    best_threshold = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0

    print('\nResults')

    print(f'- Best F1 score: {round(best_f1, 4)}')
    print(f'- Best Precision: {round(best_precision, 4)}')
    print(f'- Best Recall: {round(best_recall, 4)}')
    print(f'- Best Threshold: {round(best_threshold, 4)}')
    print(f'- AUC-PR: {round(auc_pr, 4)}')

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(
            ['Best F1', 'Best Precision', 'Best Recall', 'Best Threshold', 'AUC-PR'],
        )
        writer.writerow(
            [
                round(best_f1, 4),
                round(best_precision, 4),
                round(best_recall, 4),
                round(best_threshold, 4),
                round(auc_pr, 4)
            ]
        )
    best_anomaly_prediction = np.where(anomaly_scores >= best_threshold, 1, 0)
    
    best_f1, best_tp, best_fp, best_fn = f1_stat(
        prediction=best_anomaly_prediction,
        gt=labels
    )

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['Best TP', 'Best FP', 'Best FN'])
        writer.writerow([best_tp, best_fp, best_fn])

    return best_f1, best_tp, best_fp, best_fn, auc_pr


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--dataset',
        type=str,
        help="Dataset. Either 'MSL_SEPARATED' or 'SMAP_SEPARATED'",
    )
    args.add_argument(
        '--use-genias',
        type=str2bool,
        help='Whether to use genias or not.'
        '--seed',
        type=int, 
        default=42,
        help="Fixed Seed. Default 42."
    )
    args.add_argument(
        '--use-wandb',
        type=str2bool, 
        default=False,
        help="Control whether use wandb log or not. Default False."
    )
    args.add_argument(
        '--save-ckpt',
        type=bool, 
        default=False,
        help="Save the checkpoint. Default False."
    )
    args.add_argument(
        '--window-size',
        type=int,
        default=200,
        help="Window size. Default 200."
    )
    args.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help="Batch size. Default 50."
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="gpu number. Default 0.",
    )
    args.add_argument(
        '-seed',
        type=int,
        default=42,
        help='Fixed seed. Default 42.',
    )
    config = args.parse_args()

    fix_seed_all(config.seed)

    assert config.dataset in ['MSL_SEPARATED', 'SMAP_SEPARATED'], \
    "data-type must be either 'MSL_SEPARATED' or 'SMAP_SEPARATED'"
    data_list = os.listdir('/data/seungmin/MSL_SEPARATED/train')
    data_list = sorted(data_list)
    data_list = [data.replace('_train.npy', '') for data in data_list]

    if config.use_genias:
        log_dir = f'temp_log/carla/use_genias/{config.dataset}/results.csv'
    else:
        log_dir = f'temp_log/carla/without_genias/{config.dataset}/results.csv'
    
    with open(log_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'Inference result of {config.dataset}'])
        writer.writerow([])
        writer.writerow(['subdata', 'Best F1', 'AUC-PR'])

    best_f1_list = []
    best_tp_list = []
    best_fp_list = []
    best_fn_list = []
    auc_pr_list = []

    for subdata in data_list:
        pretext(
            dataset=config.dataset,
            subdata=subdata,
            use_genias=config.use_genias,
            gpu_num=config.gpu_num,
        )
        best_f1, best_tp, best_fp, best_fn, auc_pr = classification(
            dataset=config.dataset,
            subdata=subdata,
            use_genias=config.use_genias,
            gpu_num=config.gpu_num
        )
        best_f1_list.append(best_f1)
        best_tp_list.append(best_tp)
        best_fp_list.append(best_fp)
        best_fn_list.append(best_fn)
        auc_pr_list.append(auc_pr)

        with open(log_dir, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([subdata, round(best_f1, 4), round(auc_pr, 4)])
    
    best_f1_list = np.array(best_f1_list)
    auc_pr_list = np.array(auc_pr_list)

    f1_score_best = np.max(best_f1_list)
    f1_micro = mirco_f1(
        tp_list=best_tp_list,
        fp_list=best_fp_list,
        fn_list=best_fn_list
    )
    f1_macro = macro_f1(f1_list=best_f1_list)
    
    auc_pr_mean = np.mean(auc_pr_list)
    auc_pr_std = np.std(auc_pr_list)

    print(f'Best F1: {round(f1_score_best, 4)}')
    print(f'Micro F1: {round(f1_micro, 4)}')
    print(f'Macro F1: {round(f1_macro, 4)}')
    print(f'AUC-PR mean: {round(auc_pr_mean, 4)}')
    print(f'AUC-PR std: {round(auc_pr_std, 4)}')

    with open(log_dir, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(
            [
                'Best F1',
                'Micro F1',
                'Macro F1',
                'AUC-PR mean',
                'AUC-PR std'
            ]
        )
        writer.writerow(
            [
                round(f1_score_best, 4),
                round(f1_micro, 4),
                round(f1_macro, 4),
                round(auc_pr_mean, 4),
                round(auc_pr_std, 4),
            ]
        )
