import csv
import argparse
from tqdm import tqdm
import logging
from math import cos, pi
from torch.utils.data import DataLoader
import torch.optim as optim 
from faiss import IndexFlatL2
from sklearn.metrics import precision_recall_curve, auc
from utils.common_import import *
from data_factory.temp_loader import *
from carla.model import *
from utils.loss import pretextloss, classificationloss, entropy
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
    data_type: str,
    data_name: str,
    mode: str = 'train',
    epochs: int = 30,
    batch_size: int = 50,
    gpu_num: int = 0,
    model_save_interval: int = 5,
    num_neighbors: int = 2,
) -> None:
    assert data_type in ['MSL_SEPARATED', 'SMAP_SEPARATED'], \
    "dataset must be one of 'MSL_SEPARATED', 'SMAP_SEPARATED'"

    train_dataset = PretextDataset(
        data_type=data_type,
        data_name=data_name,
        mode=mode,
    )
    
    model = PretextModel(in_channels=train_dataset.data_dim)
    device = torch.device(f'cuda:{gpu_num}')
    model = model.to(device)
    cri = pretextloss()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters())

    save_dir = os.path.join(f'temp/model_save/{data_type}/{data_name}')
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    print(f'Pretext on {data_type}; {data_name} start...\n')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        cosine_schedule(
            optimizer=optimizer,
            current_epoch=epoch,
            total_epochs=epochs
        )
        epoch_loss = 0.0
        prev_loss = None

        for batch in train_loader:
            optimizer.zero_grad()
            
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            B, W, F = anchor.shape
            triplets = torch.cat(
                tensors=[anchor, positive, negative],
                dim=0
            ).float().view(3*B, F, W)
            representations = model.forward(triplets)
            loss = cri(
                representations=representations,
                current_loss=prev_loss,
            )
            loss.backward()
            optimizer.step()
            prev_loss = loss.item()
            epoch_loss += prev_loss
        
        epoch_loss /= len(train_loader)
        print(f'Train loss: {epoch_loss:.4e}')

        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'resnet': model.resnet.state_dict(),
                    'contrastive_head': model.contrastive_head.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(save_dir, f'epoch_{epoch + 1}.pt'),
            )
        
    print(f'Pretext training done. Start selecting neighbors of {data_name}\n')

    model.eval()

    timeseries_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    anchor_reps = []
    negative_reps = []
        
    print('Loading representations of each anchor and its negative pair.')

    for batch in tqdm(timeseries_loader):
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
    num_anchor_reps = anchor_reps.shape[0]

    index_searcher = IndexFlatL2(reps.shape[1])
    assert index_searcher.d == reps.shape[1],\
    f'{index_searcher.d} != {reps.shape[1]}'
    index_searcher.add(reps)

    anchor_and_negative_pairs = np.concatenate(
        [train_dataset.anchors, train_dataset.negative_pairs],
        axis=0,
    )

    classification_data_dir = os.path.join(
        CLASSIFICATION_PATH, data_type, data_name
    )
    os.makedirs(classification_data_dir, exist_ok=True)

    print(f'\nSelecting top-{num_neighbors} neighbors of the anchor...')

    nearest_neighbors = []
    furthest_neighbors = []

    anchor_idx = 0
    for anchor_rep in tqdm(anchor_reps):
        anchor_query = anchor_rep.reshape(1, -1)
        _, indices = index_searcher.search(anchor_query, reps.shape[0])
        indices = indices.reshape(-1)
        indices = indices[indices != anchor_idx]
        nearest_indices = indices[:num_neighbors]
        furthest_indices = indices[-num_neighbors:]
        nearest_neighbors.append(
            anchor_and_negative_pairs[nearest_indices]
        )
        furthest_neighbors.append(
            anchor_and_negative_pairs[furthest_indices]
        )
        anchor_idx += 1
    
    print('\nSaving nearest neighborhoods of the anchor...')
    nearest_neighbors = np.array(nearest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'anchor_nns.npy'
        ),
        arr=nearest_neighbors,
    )

    print('Saving furthest neighborhoods of the anchor...')
    furthest_neighbors = np.array(furthest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'anchor_fns.npy'
        ),
        arr=furthest_neighbors,
    )

    print(f'\nSelecting top-{num_neighbors} neighbors of the negative pair...')

    nearest_neighbors = []
    furthest_neighbors = []

    negative_idx = 0
    for negative_rep in tqdm(negative_reps):
        negative_query = negative_rep.reshape(1, -1)
        _, indices = index_searcher.search(negative_query, reps.shape[0])
        indices = indices.reshape(-1)
        indices = indices[indices != negative_idx + num_anchor_reps]
        nearest_indices = indices[:num_neighbors]
        furthest_indices = indices[-num_neighbors:]
        nearest_neighbors.append(
            anchor_and_negative_pairs[nearest_indices]
        )
        furthest_neighbors.append(
            anchor_and_negative_pairs[furthest_indices]
        )
        negative_idx += 1
    
    print('\nSaving nearest neighborhoods of the anchor...')
    nearest_neighbors = np.array(nearest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'negative_nns.npy'
        ),
        arr=nearest_neighbors,
    )

    print('Saving furthest neighborhoods of the anchor...')
    furthest_neighbors = np.array(furthest_neighbors)
    np.save(
        file=os.path.join(
            classification_data_dir, 'negative_fns.npy'
        ),
        arr=furthest_neighbors,
    )

    print('\nPretext stage done. Move on to the classification stage.')

    return


def classification(
    data_type: str,
    data_name: str,
    gpu_num: int = 0,
    epochs: int = 100,
    batch_size: int = 50,
    learning_rate: float = 1e-2,
    model_save_interval: int = 5,
) -> None:
    device = torch.device(f':cuda{gpu_num}')

    train_dataset = ClassificationDataset(
        data_type=data_type,
        data_name=data_name,
        mode='train'
    )
    data_dim = train_dataset.data_dim
    model = ClassificationModel(in_channels=data_dim)
    ckpt_dir = os.path.join(f'temp/model_save/{data_type}/{data_name}')
    ckpt = torch.load(
        os.path.join(ckpt_dir, 'epoch_30.pt')
    )
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

    model.train()
    print('Classification training loop start...\n')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} start.')
        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_inconsistency_loss = 0.0
        epoch_entropy_loss = 0.0

        for batch in tqdm(train_loader):
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
        
        print(f'Epoch {epoch + 1} finished.')
        print(f'- Consistency loss: {epoch_consistency_loss:.4e}')
        print(f'- Inconsistency loss: {epoch_inconsistency_loss:.4e}')
        print(f'- Entropy loss: {epoch_entropy_loss:.4e}')
        print(f'- Total loss: {epoch_loss:.4e}\n')


        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(
                    log_dir, 'model_classification', f'epoch_{epoch + 1}.pt'
                    ),
                )
    
    print('Classification training done.')

    # inference

    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--data',
        type=str,
        help="Dataset. Either 'MSL_SEPARATED' or 'SMAP_SEPARATED'"
    )
    args.add_argument(
        '--data-name',
        type=str,
        help='Name of data.'
    )
    args.add_argument(
        '-seed',
        type=int,
        default=42,
        help='Fixed seed. Default 42.'
    )
    args.add_argument(
        '--save-ckpt',
        type=str2bool,
        default=False,
        help='Save checkpoint. Default False'
    )
    config = args.parse_args()
    assert config.data in ['MSL_SEPARATED', 'SMAP_SEPARATED']
    fix_seed_all(config.seed)
    pretext(data_type=config.data, data_name=config.data_name)

