import argparse, logging
from datetime import datetime
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
from utils.set_logging import set_logging_filehandler


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
    timestamp: str,
    subdata: Optional[str] = None,
    scheme: str = 'carla',
    mix_step: Optional[int] = None,
    dataloader_shuffle: bool = True,
    epochs: int = 30,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    gpu_num: int = 0,
    num_neighbors: int = 5,
    cut_negative_pairs: bool = True,
) -> None:
    assert scheme in ['carla', 'genias', 'mix', 'genias_multiple'], \
    "'carla', 'genias', 'mix', 'genias_multiple'"

    print(f'Pretext training on {dataset} {subdata} start...\n')

    train_dataset = PretextDataset(
        dataset=dataset,
        subdata=subdata,
        scheme=scheme,
        mix_step=mix_step,
        cut_negative_pairs=cut_negative_pairs,
    )

    data_dim = train_dataset.data_dim
    model = PretextModel(in_channels=data_dim, mid_channels=4)

    device = torch.device(f'cuda:{gpu_num}')
    model = model.to(device)
    criterion = pretextloss()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
    )
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    ckpt_dir = f'checkpoints/pretext/{dataset}'
    classification_dir = f'classification_dataset/{dataset}'
    
    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        ckpt_dir = f'{ckpt_dir}/{subdata}/{scheme}'
        classification_dir = f'{classification_dir}/{subdata}/{scheme}'
    else:
        ckpt_dir = f'{ckpt_dir}/{scheme}'
        classification_dir = f'{classification_dir}/{scheme}'

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
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

            if scheme == 'genias_multiple':
                loss = torch.zeros(1, requires_grad=True).float().to(device)
                for i in range(positive_pair.shape[1]):
                    positive_pair_i = positive_pair[:, i]
                    negative_pair_i = negative_pair[:, i]
                    triplets_i = torch.cat(
                        tensors=[anchor, positive_pair_i, negative_pair_i],
                        dim=0,
                    ).float()
                    triplets_i = triplets_i.view(3 * B, F, W)
                    representations_i = model.forward(triplets_i)
                    loss_i = criterion(
                        representations=representations_i,
                        current_loss=prev_loss,
                    )
                    prev_loss = loss_i.item()
                loss += loss_i
            
            else:
                triplets = torch.cat(
                    tensors=[anchor, positive_pair, negative_pair],
                    dim=0
                ).float()

                triplets = triplets.view(3 * B, F, W)
                representations = model.forward(triplets)
                loss = criterion(
                    representations=representations,
                    current_loss=prev_loss,
                )
            
            loss.backward()
            optimizer.step()

            if not scheme == 'genias_multiple':
                prev_loss = loss.item()
            
            epoch_loss += prev_loss
        
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch + 1} train loss: {epoch_loss:.4e}')

    torch.save(
        obj={
            'resnet': model.resnet.state_dict(),
            'contrastive_head': model.contrastive_head.state_dict(),
            'optim': optimizer.state_dict(),
        },
        f=f'{ckpt_dir}/{timestamp}.pt'
    )

    print(f'Pretext training on {dataset} {subdata} finished.')

    print(f'Start saving top-{num_neighbors} neighbors...')
    model.eval()

    if cut_negative_pairs:
        train_dataset._cut_negative_pairs()
    
    timeseries_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    anchor_reps = []
    negative_reps = []

    # Loading representations of each anchor and its negative pair.
    for batch in timeseries_loader:
        anchor, _, negative_pair = batch
        anchor = anchor.to(device).float().transpose(-2, -1)
        anchor_rep = model.forward(anchor).detach().cpu()
        anchor_reps.append(anchor_rep)

        if scheme == 'genias_multiple':
            if cut_negative_pairs:
                negative_pair = negative_pair.float().transpose(-2, -1)
                negative_pair = negative_pair.to(device)
                negative_rep = model.forward(negative_pair).detach().cpu()
                negative_reps.append(negative_rep)
            else:
                for i in range(negative_pair.shape[1]):
                    negative_pair_i = negative_pair[:, i].float().transpose(-2, -1)
                    negative_pair_i = negative_pair_i.to(device)
                    negative_rep_i = model.forward(negative_pair_i).detach().cpu()
                    negative_reps.append(negative_rep_i)
        else:
            negative_pair = negative_pair.to(device).float().transpose(-2, -1)
            negative_rep = model.forward(negative_pair).detach().cpu()
            negative_reps.append(negative_rep)
    
    anchor_reps = torch.cat(anchor_reps, dim=0).numpy()
    negative_reps = torch.cat(negative_reps, dim=0).numpy()
    
    reps = np.concatenate([anchor_reps, negative_reps], axis=0)

    index_searcher = IndexFlatL2(reps.shape[1])
    assert index_searcher.d == reps.shape[1],\
    f'{index_searcher.d} != {reps.shape[1]}'
    index_searcher.add(reps)

    nearest_indices_list = []
    furthest_indices_list = []

    for anchor_rep in anchor_reps:
        anchor_query = anchor_rep.reshape(1, -1)
        _, indices = index_searcher.search(anchor_query, reps.shape[0])
        indices = indices.reshape(-1)
        nearest_indices = indices[1: num_neighbors+1]
        furthest_indices = indices[-num_neighbors:]
        nearest_indices_list.append(nearest_indices)
        furthest_indices_list.append(furthest_indices)
    
    # Saving nearest/furthest indeces of the anchor.
    nearest_indices_list = np.array(nearest_indices_list)
    furthest_indices_list = np.array(furthest_indices_list)
    np.save(
        file=f'{classification_dir}/anchor_nn_indices.npy',
        arr=nearest_indices_list,
    )
    np.save(
        file=f'{classification_dir}/anchor_fn_indices.npy',
        arr=furthest_indices_list,
    )

    # Selecting nearest/furthest indices of the negative pair.
    nearest_indices_list = []
    furthest_indices_list = []

    for negative_rep in negative_reps:
        negative_query = negative_rep.reshape(1, -1)
        _, indices = index_searcher.search(negative_query, reps.shape[0])
        indices = indices.reshape(-1)
        nearest_indices = indices[1: num_neighbors+1]
        furthest_indices = indices[-num_neighbors:]
        nearest_indices_list.append(nearest_indices)
        furthest_indices_list.append(furthest_indices)

    # Saving nearest/furthest indices of the negative pair.
    nearest_indices_list = np.array(nearest_indices_list)
    furthest_indices_list = np.array(furthest_indices_list)
    np.save(
        file=f'{classification_dir}/negative_nn_indices.npy',
        arr=nearest_indices_list,
    )
    np.save(
        file=f'{classification_dir}/negative_fn_indices.npy',
        arr=furthest_indices_list,
    )

    print('\nPretext stage done. Moving on to classification stage.\n')

    return


def classification(
    dataset: str,
    timestamp: str,
    subdata: Optional[str] = None,
    scheme: str = 'carla',
    dataloader_shuffle: bool = True,
    gpu_num: int = 0,
    epochs: int = 100,
    batch_size: int = 50,
    learning_rate: float = 1e-2,
) -> Tuple[float, int, int, int, float]:
    device = torch.device(f'cuda:{gpu_num}')

    train_dataset = ClassificationDataset(
        dataset=dataset,
        subdata=subdata,
        mode='train',
        scheme=scheme,
    )
    data_dim = train_dataset.data_dim
    model = ClassificationModel(in_channels=data_dim)

    resnet_dir = f'checkpoints/pretext/{dataset}'
    classification_dir = f'classification_datset/{dataset}'
    ckpt_dir = f'checkpoints/classification/{dataset}'
    
    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        resnet_dir = f'{resnet_dir}/{subdata}/{scheme}'
        classification_dir = f'{classification_dir}/{subdata}/{scheme}'
        ckpt_dir = f'{ckpt_dir}/{subdata}/{scheme}'
    else:
        resnet_dir = f'{resnet_dir}/{scheme}'
        classification_dir = f'{classification_dir}/{scheme}'
        ckpt_dir = f'{ckpt_dir}/{scheme}'

    os.makedirs(ckpt_dir, exist_ok=True)

    resnet_ckpt = torch.load(f'{resnet_dir}/{timestamp}.pt')
    model.resnet.load_state_dict(resnet_ckpt['resnet'])
    model = model.to(device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
    )
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    criterion = classificationloss()

    logging.info(f'Classification training on {dataset} {subdata} start...\n')
    model.train()

    for epoch in range(epochs):
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

            nearest_logit = model.forward(nearest_neighbor.transpose(-2, -1))
            furthest_logit = model.forward(furthest_neighbor.transpose(-2, -1))

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
        
        logging.info(f'Epoch {epoch + 1} loss:')
        logging.info(
            f'- Consistency loss: {round(epoch_consistency_loss, 4)}'
        )
        logging.info(
            f'- Inconsistency loss: {round(epoch_inconsistency_loss, 4)}'
        )
        logging.info(
            f'- Entropy loss: {round(epoch_entropy_loss, 4)}'
        )
        logging.info(
            f'- Total loss: {round(epoch_loss, 4)}\n'
        )

    torch.save(
        obj={
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        },
        f=f'{ckpt_dir}/{timestamp}.pt',
    )
    
    logging.info(f'Starting inference on {dataset} {subdata}...\n')
    model.eval()

    test_dataset = ClassificationDataset(
        dataset=dataset,
        subdata=subdata,
        mode='test',
        scheme=scheme,
    )

    logits = []

    for idx in range(len(test_dataset)):
        test_data = torch.tensor(test_dataset[idx], dtype=torch.float32)
        test_data = test_data.to(device)
        logit = model.forward(test_data.unsqueeze(0).transpose(-2, -1))
        logit = logit.squeeze(0).detach().cpu().numpy()
        logits.append(logit)

    classes = [0 for _ in range(10)]

    for logit in logits:
        max_index = np.argmax(logit)
        classes[max_index] += 1

    major_class = classes.index(max(classes))

    anomaly_scores = []

    for i in range(len(logits)):
        logit = logits[i]
        major_probability = logit[major_class]
        anomaly_scores.append(1 - major_probability)

    anomaly_scores = np.array(anomaly_scores)

    precision, recall, thresholds = precision_recall_curve(
        y_true=test_dataset.labels,
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

    logging.info(f'- Best F1 score: {round(best_f1, 4)}')
    logging.info(f'- Best Precision: {round(best_precision, 4)}')
    logging.info(f'- Best Recall: {round(best_recall, 4)}')
    logging.info(f'- Best Threshold: {round(best_threshold, 4)}')
    logging.info(f'- AUC-PR: {round(auc_pr, 4)}')

    best_anomaly_prediction = np.where(anomaly_scores >= best_threshold, 1, 0)
    
    best_f1_score, best_tp, best_fp, best_fn = f1_stat(
        prediction=best_anomaly_prediction,
        gt=test_dataset.labels
    )

    return best_f1_score, best_tp, best_fp, best_fn, auc_pr


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--dataset',
        type=str,
        help="Dataset.",
    )
    args.add_argument(
        '--scheme',
        type=str,
        default='carla',
        help="How the pairs are made in pretext stage. Default 'carla'"
    )
    args.add_argument(
        '--mix-step',
        type=Optional[int],
        default=None,
        help='CARLA and GenIAS scheme alters every this timestep. For mix scheme.'
    )
    args.add_argument(
        '--cut-negative-pairs',
        type=str2bool,
        default=False,
        help='Whether to bring only one negative pair for each anchor in classification. Default True.'
    )
    args.add_argument(
        '--dataloader-shuffle',
        type=str2bool,
        default=True,
        help='Shuffle batches of dataloader. Default True'
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

    fix_seed_all(seed=config.seed)

    log_dir = f'log/carla/{config.dataset}'
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")

    log_file_path = f'{log_dir}/{config.scheme}/{timestamp}.log'
    
    set_logging_filehandler(log_file_path=log_file_path)
    logging.info(f'Experiment log of {config.dataset}\n')
    logging.info(f'Settings:')
    logging.info(f'- Date: {timestamp.replace('_', ' ')}')
    logging.info(f'- Dataset: {config.dataset}')
    logging.info(f'- Scheme: {config.scheme}')

    if config.scheme == 'mix':
        logging.info(f'- Mix step: {config.mix_step}')
    
    if config.scheme == 'multiple_genias':
        logging.info(f'- Cut negative pairs: {config.cut_negative_pairs}')

    logging.info(f'- Dataloader Shuffle: {config.dataloader_shuffle}')
    logging.info(f'- GPU number: {config.gpu_num}\n')


    best_f1_list = []
    best_tp_list = []
    best_fp_list = []
    best_fn_list = []
    auc_pr_list = []

    if config.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        data_dir = f'data/{config.dataset}/train'
        data_list = sorted(os.listdir(data_dir))
        data_list = [data.replace('.npy', '') for data in data_list]

        for subdata in data_list:
            pretext(
                dataset=config.dataset,
                timestamp=timestamp,
                subdata=subdata,
                scheme=config.scheme,
                mix_step=config.mix_step,
                dataloader_shuffle=config.dataloader_shuffle,
                gpu_num=config.gpu_num,
                cut_negative_pairs=config.cut_negative_pairs,
            )
            best_f1_score, best_tp, best_fp, best_fn, auc_pr = classification(
                dataset=config.dataset,
                timestamp=timestamp,
                subdata=subdata,
                scheme=config.scheme,
                dataloader_shuffle=config.dataloader_shuffle,
                gpu_num=config.gpu_num
            )
            best_f1_list.append(best_f1_score)
            best_tp_list.append(best_tp)
            best_fp_list.append(best_fp)
            best_fn_list.append(best_fn)
            auc_pr_list.append(auc_pr)

            logging.info(f'- True Positives: {best_tp}')
            logging.info(f'- False Positives: {best_fp}')
            logging.info(f'- False Negatives: {best_fn}\n')
    
    else:
        pretext(
            dataset=config.dataset,
            timestamp=timestamp,
            scheme=config.scheme,
            mix_step=config.mix_step,
            dataloader_shuffle=config.dataloader_shuffle,
            gpu_num=config.gpu_num,
            cut_negative_pairs=config.cut_negative_pairs,
        )
        best_f1_score, best_tp, best_fp, best_fn, auc_pr = classification(
            dataset=config.dataset,
            timestamp=timestamp,
            scheme=config.scheme,
            dataloader_shuffle=config.dataloader_shuffle,
            gpu_num=config.gpu_num
        )
        best_f1_list.append(best_f1_score)
        best_tp_list.append(best_tp)
        best_fp_list.append(best_fp)
        best_fn_list.append(best_fn)
        auc_pr_list.append(auc_pr)

        logging.info(f'- True Positives: {best_tp}')
        logging.info(f'- False Positives: {best_fp}')
        logging.info(f'- False Negatives: {best_fn}\n')

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

    logging.info('Scores')
    logging.info(f'- Best F1: {round(f1_score_best, 4)}')
    logging.info(f'- Micro F1: {round(f1_micro, 4)}')
    logging.info(f'- Precision: {round(precision, 4)}')
    logging.info(f'- Recall: {round(recall, 4)}')
    logging.info(f'- AUC-PR mean: {round(auc_pr_mean, 4)}')
    logging.info(f'- AUC-PR std: {round(auc_pr_std, 4)}')
    logging.info(f'- Macro F1: {round(f1_macro, 4)}')
        