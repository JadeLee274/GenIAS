from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from faiss import IndexFlatL2
import torch.optim.lr_scheduler as sched
from sklearn.metrics import precision_recall_curve, auc
from genias.data_factory.loader import *
from genias.utils.metric import *
from genias.utils.loss import *


def pretext(
    dataset: str,
    timestamp: str,
    subdata: Optional[str] = None,
    downsample: bool = False,
    downsample_step: int = 5,
    scheme: str = 'carla',
    inject_different_anomalies: bool = False,
    use_pretrained_vae: bool = True,
    mix_step: Optional[int] = None,
    epochs: int = 30,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    gpu_num: int = 0,
    num_neighbors: int = 5,
    cut_negative_pairs: bool = True,
) -> None:
    assert scheme in [
        'carla',
        'carla_modified',
        'genias',
        'mix',
        'genias_multiple',
    ], "'carla', 'carla_modified', 'genias', 'mix', 'genias_multiple'"

    if (scheme != 'carla' and not use_pretrained_vae):
        vae_train(
            dataset=dataset,
            subdata=subdata,
            gpu_num=gpu_num,
        )
    
    if dataset in ['MSL', 'SMAP', 'SMD']:
        logging.info(f'Pretext training on {dataset} {subdata} start...\n')
    elif dataset in ['SWaT', 'WADI']:
        logging.info(f'Pretext training on {dataset} start...\n')

    train_dataset = PretextDataset(
        dataset=dataset,
        timestamp=timestamp,
        subdata=subdata,
        scheme=scheme,
        inject_different_anomalies=inject_different_anomalies,
        mix_step=mix_step,
        cut_negative_pairs=cut_negative_pairs,
        downsample=downsample,
        downsample_step=downsample_step,
    )
    data_dim = train_dataset.data_dim

    model = PretextModel(in_channels=data_dim, mid_channels=4)
    device = torch.device(f'cuda:{gpu_num}')
    model = model.to(device)
    criterion = pretextloss()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
        logging.info(f'Epoch {epoch + 1} train loss: {epoch_loss:.4e}')

    torch.save(
        obj={
            'resnet': model.resnet.state_dict(),
            'contrastive_head': model.contrastive_head.state_dict(),
            'optim': optimizer.state_dict(),
        },
        f=os.path.join(ckpt_dir, f'{timestamp}.pt')
    )

    logging.info('')
    if dataset in ['MSL', 'SMAP', 'SMD']:
        logging.info(f'Pretext training on {dataset} {subdata} finished.\n')
    elif dataset in ['SWaT', 'WADI']:
        logging.info(f'Pretext training on {dataset} finished.\n')

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

    for anchor_rep in tqdm(anchor_reps):
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
        file=os.path.join(
            classification_dir, f'anchor_nn_indices_{timestamp}.npy'
        ),
        arr=nearest_indices_list,
    )
    np.save(
        file=os.path.join(
            classification_dir, f'anchor_fn_indices_{timestamp}.npy'
        ),
        arr=furthest_indices_list,
    )

    # Selecting nearest/furthest indices of the negative pair.
    nearest_indices_list = []
    furthest_indices_list = []

    for negative_rep in tqdm(negative_reps):
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
        file=os.path.join(
            classification_dir, f'negative_nn_indices_{timestamp}.npy'
        ),
        arr=nearest_indices_list,
    )
    np.save(
        file=os.path.join(
            classification_dir, f'negative_fn_indices_{timestamp}.npy'
        ),
        arr=furthest_indices_list,
    )

    print('\nPretext stage done. Moving on to classification stage.\n')

    return


def classification(
    dataset: str,
    timestamp: str,
    subdata: Optional[str] = None,
    downsample: bool = False,
    downsample_step: int = 5,
    scheme: str = 'carla',
    gpu_num: int = 0,
    epochs: int = 100,
    batch_size: int = 50,
    learning_rate: float = 1e-2,
    cut_negative_pairs: bool = True,
) -> Tuple[float, int, int, int, float]:
    assert scheme in [
        'carla', 'carla_modified', 'genias', 'mix', 'genias_multiple'
    ], "'carla', 'carla_modified', 'genias', 'mix', 'genias_multiple'"

    device = torch.device(f'cuda:{gpu_num}')

    train_dataset = ClassificationDataset(
        dataset=dataset,
        timestamp=timestamp,
        downsample=downsample,
        downsample_step=downsample_step,
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
        shuffle=True,
    )
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    criterion = classificationloss()

    if dataset in ['MSL', 'SMAP', 'SMD']:
        logging.info(
            f'Classification training on {dataset} {subdata} start...\n'
        )
    elif dataset in ['SWaT', 'WADI']:
        logging.info(
            f'Classification training on {dataset} start...\n'
        )
    
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
        timestamp=timestamp,
        subdata=subdata,
        downsample=downsample,
        downsample_step=downsample_step,
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
        y_true=test_dataset.test_labels,
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
        gt=test_dataset.test_labels
    )

    return best_f1_score, best_tp, best_fp, best_fn, auc_pr


def vae_train(
    dataset: str,
    timestamp: str,
    subdata: Optional[str] = None,
    batch_size: int = 100,
    depth: int = 10,
    window_size: int = 200,
    latent_dim: int = 100,
    gpu_num: int = 0,
    epochs: int = 1000,
    init_lr: float = 1e-4,
    prior_var: float = 0.5,
    recon_weight: float = 1.0,
    pert_weight: float = 0.1,
    zero_pert_weight: float = 0.01,
    kld_weight: float = 0.1,
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
                logvar=logvar,
                prior_var=prior_var,
                recon_weight=recon_weight,
                pert_weight=pert_weight,
                zero_pert_weight=zero_pert_weight,
                kld_weight=kld_weight,
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

    torch.save(
        obj={
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        },
        f=os.path.join(ckpt_dir, f'{timestamp}.pt')
    )
            
    logging.info('Training Finished')

    return
