import argparse, logging
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from faiss import IndexFlatL2
from sklearn.metrics import precision_recall_curve, auc
from genias.utils.common_import import *
from genias.data_factory.loader import *
from genias.models.carla import *
from genias.utils.loss import *
from genias.utils.metric import *
from genias.utils.main import *


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
    assert scheme in [
        'carla_original',
        'genias',
        'mix',
        'genias_multiple',
    ], "'carla', 'genias', 'mix', 'genias_multiple', 'carla_temp'"

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

    ckpt_dir = os.path.join('genias', 'checkpoints', 'pretext', dataset)
    classification_dir = os.path.join(
        'genias', 'data', 'classification_dataset', dataset)
    
    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        ckpt_dir = os.path.join(ckpt_dir, subdata)
        classification_dir = os.path.join(classification_dir, subdata)

    ckpt_dir = os.path.join(ckpt_dir, scheme)
    classification_dir = os.path.join(classification_dir, scheme)
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
                    loss_i = criterion(representations_i, prev_loss)
                    prev_loss = loss_i.item()
                
                loss += loss_i
            
            else:
                triplets = torch.cat(
                    tensors=[anchor, positive_pair, negative_pair],
                    dim=0
                ).float()

                triplets = triplets.view(3 * B, F, W)
                representations = model.forward(triplets)
                loss = criterion(representations, prev_loss)
            
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
        f=os.path.join(ckpt_dir, f'{timestamp}.pt')
    )

    print(f'Pretext training on {dataset} {subdata} finished.')

    print(f'Start saving top-{num_neighbors} neighbors...')
    model.eval()

    if scheme == 'genias_multiple' and cut_negative_pairs:
        train_dataset.cut_negative_pairs()
    
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
                    negative_pair_i = negative_pair[:, i]
                    negative_pair_i = negative_pair_i.float().transpose(-2, -1)
                    negative_pair_i = negative_pair_i.to(device)
                    negative_rep_i = model.forward(negative_pair_i)
                    negative_rep_i = negative_rep_i.detach().cpu()
                    negative_reps.append(negative_rep_i)
        else:
            negative_pair = negative_pair.to(device).float().transpose(-2, -1)
            negative_rep = model.forward(negative_pair).detach().cpu()
            negative_reps.append(negative_rep)
    
    anchor_reps = torch.cat(anchor_reps, dim=0).numpy()
    negative_reps = torch.cat(negative_reps, dim=0).numpy()
    
    reps = np.concatenate([anchor_reps, negative_reps], axis=0)

    index_searcher = IndexFlatL2(reps.shape[1])
    assert index_searcher.d == reps.shape[1], \
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
        file=os.path.join(classification_dir, 'anchor_nn_indices.npy'),
        arr=nearest_indices_list,
    )
    np.save(
        file=os.path.join(classification_dir, 'anchor_fn_indices.npy'),
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
        file=os.path.join(classification_dir, 'negative_nn_indices.npy'),
        arr=nearest_indices_list,
    )
    np.save(
        file=os.path.join(classification_dir, 'negative_fn_indices.npy'),
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
    cut_negative_pairs: bool = True,
) -> Tuple[float, int, int, int, float]:
    assert scheme in [
        'carla_original', 'genias', 'mix', 'genias_multiple'
    ], "'carla_original', 'genias', 'mix', 'genias_multiple'"

    device = torch.device(f'cuda:{gpu_num}')

    train_dataset = ClassificationDataset(
        dataset=dataset,
        subdata=subdata,
        mode='train',
        scheme=scheme,
    )
    data_dim = train_dataset.data_dim
    model = ClassificationModel(in_channels=data_dim)

    resnet_dir = os.path.join('genias', 'checkpoints', 'pretext', dataset)
    classification_dir = os.path.join('classification_dataset', dataset)
    ckpt_dir = os.path.join('genias', 'checkpoints', 'classification', dataset)
    
    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        resnet_dir = os.path.join(resnet_dir, subdata)
        classification_dir = os.path.join(classification_dir, subdata)
        ckpt_dir = os.path.join(ckpt_dir, subdata)
    
    resnet_dir = os.path.join(resnet_dir, scheme)
    classification_dir = os.path.join(classification_dir, scheme)
    ckpt_dir = os.path.join(ckpt_dir, scheme)
    os.makedirs(ckpt_dir, exist_ok=True)

    resnet_ckpt = torch.load(os.path.join(resnet_dir, f'{timestamp}.pt'))
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
            nearest_neighbor = nearest_neighbor.to(device)
            nearest_neighbor = nearest_neighbor.float().transpose(-2, -1)
            furthest_neighbor = furthest_neighbor.to(device)
            furthest_neighbor = furthest_neighbor.float().transpose(-2, -1)

            window = window.transpose(-2, -1)
            window_logit = model.forward(window)

            entropy_loss = entropy(torch.mean(window_logit, 0))
            batch_loss -= entropy_loss
            epoch_entropy_loss += entropy_loss.item()

            batch_consistency_sum = torch.zeros(1, device=device)
            batch_consistency = 0.0
            batch_inconsistency = 0.0

            if cut_negative_pairs:
                nearest_logit = model.forward(nearest_neighbor)
                furthest_logit = model.forward(furthest_neighbor)

                consistency_sum, consistency, inconsistency \
                = criterion(window_logit, nearest_logit, furthest_logit)
            
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
        f=os.path.join(ckpt_dir, f'{timestamp}.pt'),
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


def vae_train(
    dataset: str,
    subdata: str,
    batch_size: int = 100,
    depth: int = 10,
    window_size: int = 200,
    latent_dim: int = 100,
    gpu_num: int = 0,
    epochs: int = 1000,
    init_lr: float = 1e-4,
    checkpoint_step: int = 100,
) -> None:
    train_data = GenIASDataset(
        dataset=dataset,
        subdata=subdata,
        window_size=window_size,
    )
    data_dim = train_data.data_dim

    device = torch.device(f'cuda:{gpu_num}')

    model = VAE(
        window_size=window_size,
        data_dim=data_dim,
        latent_dim=latent_dim,
        depth=depth,
    ).to(device)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=init_lr)
    scheduler = sched.StepLR(optimizer=optimizer, step_size=10, gamma=0.99)

    ckpt_dir = os.path.join('genias', 'checkpoints', 'vae', dataset)

    if dataset in ['MSL', 'SMAP', 'SMD', 'KPI', 'Yahoo-A1']:
        ckpt_dir = os.path.join(ckpt_dir, subdata)

    os.makedirs(ckpt_dir, exist_ok=True)

    logging.info(f'VAE training log for {dataset} {subdata} dataset\n')

    for epoch in range(epochs):
        recon_loss = 0.0
        pert_loss = 0.0
        zero_pert_loss = 0.0
        kld_loss = 0.0
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device).float()
            optimizer.zero_grad()
            mu, logvar, x_hat, x_tilde = model(data)
            recon, pert, zero_pert, kld, total_loss = vae_loss(
                x=data,
                x_hat=x_hat,
                x_tilde=x_tilde,
                mu=mu,
                logvar=logvar
            )
            total_loss.backward()
            recon_loss += recon
            pert_loss += pert
            zero_pert_loss += zero_pert
            kld_loss += kld
            train_loss += total_loss.item()
            optimizer.step()

        scheduler.step()

        recon_loss /= len(train_loader)
        pert_loss /= len(train_loader)
        zero_pert_loss /= len(train_loader)
        kld_loss /= len(train_loader)
        train_loss /= len(train_loader)

        logging.info(f'Epoch {epoch+1} loss:')
        logging.info(f'- Reconstruction loss: {recon_loss:.4f}')
        logging.info(f'- Perturbation loss: {pert_loss:.4f}')
        logging.info(f'- Zero perturbation loss: {zero_pert_loss:.4f}')
        logging.info(f'- KL-Divergence loss: {kld_loss:.4f}')
        logging.info(f'- Total loss: {train_loss:.4f}\n')

        if epoch == 0 or (epoch + 1) % checkpoint_step == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )
            
    logging.info('Training Finished')

    return


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--exp-name',
        type=str,
        required=True,
        help='The name of the experiment.',
    )
    args.add_argument(
        '--task',
        type=str,
        default='carla_all',
        help='Which task will be performed.'
    )
    args.add_argument(
        '--dataset',
        type=str,
        help="Dataset.",
    )
    args.add_argument(
        '--start-subdata',
        type=str,
        help='From which subdata that the process will be started.'
    )
    args.add_argument(
        '--carla-scheme',
        type=str,
        default='carla_original',
        help="How the pairs will be made for CARLA stage. Default 'carla_original'."
    )
    args.add_argument(
        '--timestamp',
        type=str,
        help='When task is classification, uses pretext model trained in this time.'
    )
    args.add_argument(
        '--mix-step',
        type=int,
        help='CARLA and GenIAS scheme alters every this timestep. For mix scheme.'
    )
    args.add_argument(
        '--cut-negative-pairs',
        type=str2bool,
        default=True,
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
    args.add_argument(
        '--vae-depth',
        type=int,
        default=10,
        help='Depth of encoder and decoder of VAE. Default 10.'
    )
    config = args.parse_args()
    assert config.task in [
        'pretext',
        'classification',
        'carla_all',
        'vae_train'
    ], "task: 'pretext', 'classification', 'carla_all', 'vae_train'"

    fix_seed_all(seed=config.seed)

    if config.task == 'classification':
        timestamp = config.timestamp
    else:
        now = datetime.now()
        timestamp = now.strftime("%m%d_%H%M")
        

    if config.task != 'pretext':
        log_dir = os.path.join('log', config.task, config.dataset)

        if config.task != 'vae_train':
            log_dir = os.path.join(log_dir, config.carla_scheme)

        os.makedirs(log_dir, exist_ok=True)
        
        log_file_path = os.path.join(log_dir, f'{timestamp}.log')

        set_logging_filehandler(log_file_path=log_file_path)

        logging.info(f'Experiment: {config.exp_name}\n')
        logging.info(f'Settings:')
        logging.info(f'- Task: {config.task}')
        logging.info(f'- Date: {timestamp.replace('_', ' ')}')
        logging.info(f'- Dataset: {config.dataset}')

        if config.start_subdata is not None:
            logging.info(f'- Starts from {config.start_subdata}')

        if config.task == 'classification':
            logging.info(f'- Timestamp of pretext used: {timestamp}')

        if config.task != 'vae_train':
            logging.info(f'- Carla scheme: {config.carla_scheme}')

        if config.carla_scheme == 'mix':
            logging.info(f'- Mix step: {config.mix_step}')
        
        elif config.carla_scheme == 'multiple_genias':
            logging.info(f'- Cut negative pairs: {config.cut_negative_pairs}')

        logging.info(f'- Dataloader Shuffle: {config.dataloader_shuffle}')
        logging.info(f'- GPU number: {config.gpu_num}\n')

    best_f1_list = []
    best_tp_list = []
    best_fp_list = []
    best_fn_list = []
    auc_pr_list = []
    
    if config.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        data_dir = os.path.join(
            'genias', 'data', 'dataset', config.dataset, 'train'
        )
        data_list = sorted(os.listdir(data_dir))
        data_list = [data.replace('.npy', '') for data in data_list]

        if config.start_subdata is not None:
            start_idx = data_list.index(config.start_subdata)
            data_list = data_list[start_idx:]

        for subdata in data_list:
            if config.task == 'pretext':
                pretext(
                    dataset=config.dataset,
                    timestamp=timestamp,
                    subdata=subdata,
                    scheme=config.carla_scheme,
                    mix_step=config.mix_step,
                    dataloader_shuffle=config.dataloader_shuffle,
                    gpu_num=config.gpu_num,
                    cut_negative_pairs=config.cut_negative_pairs,
                )

            elif config.task == 'classification':
                best_f1_score, best_tp, best_fp, best_fn, auc_pr = \
                    classification(
                        dataset=config.dataset,
                        timestamp=timestamp,
                        subdata=subdata,
                        scheme=config.carla_scheme,
                        dataloader_shuffle=config.dataloader_shuffle,
                        gpu_num=config.gpu_num
                    )
                logging.info(f'- True Positives: {best_tp}')
                logging.info(f'- False Positives: {best_fp}')
                logging.info(f'- False Negatives: {best_fn}\n')

            elif config.task == 'carla_all':
                pretext(
                    dataset=config.dataset,
                    timestamp=timestamp,
                    subdata=subdata,
                    scheme=config.carla_scheme,
                    mix_step=config.mix_step,
                    dataloader_shuffle=config.dataloader_shuffle,
                    gpu_num=config.gpu_num,
                    cut_negative_pairs=config.cut_negative_pairs,
                )
                best_f1_score, best_tp, best_fp, best_fn, auc_pr = \
                    classification(
                        dataset=config.dataset,
                        timestamp=timestamp,
                        subdata=subdata,
                        scheme=config.carla_scheme,
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
        
            elif config.task == 'vae_train':
                vae_train(
                    dataset=config.dataset,
                    subdata=subdata,
                    depth=config.vae_depth,
                    gpu_num=config.gpu_num,
                )

        if config.task in ['classification', 'carla_all']:
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
