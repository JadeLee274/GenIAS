import logging
import argparse
from math import cos, pi
from tqdm import tqdm
from faiss import IndexFlatL2
from torch.nn import Softmax
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
from utils.common_import import *
from data_factory.loader import PretextDataset, ClassificationDataset
from carla.model import PretextModel, ClassificationModel
from utils.loss import pretextloss, classificationloss, entropy
from utils.metric import f1


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
    window_size: int = 200,
    epochs: int = 30,
    batch_size: int = 50,
    gpu_num: int = 0,
    learning_rate: float = 1e-3,
    model_save_interval: int = 5,
    use_genias: bool = False,
    num_neighbors: int = 2,
) -> None:
    """
    Training code for CARLA pretext stage.

    Parameters:
        dataset:             Name of the training dataset.
        window_size:         Window size. Default 200.
        batch_size:          Batch size. Default 50.
        gpu_num:             The model is trained in this GPU. Default 0.
        epochs:              Training epoch: Default 30.
        learning_rate:       Initial learning rate. Default 1e-3.
        model_save_interval: The ResNet is saved once in this epoch. 
                             Default 5.
        use_genias:          Whether or not to use GenIAS for creating 
                             negative pair. Default True.
        num_neighbors:       Choose this number of nearese/furthers 
                             neighborhood after the training loop.
                             Default 5.

    Uses Resnet model and mlp head to map anchor, positive pair, and negative
    pair to the representation space (with dimension 128, in this case).

    While training, the pretext loss is optimized so that the distance between
    the anchor and the positive pair get smaller, while that of
    the anchor and the negative pair get larger, in the representation space.

    The model is saved once in a model_save_interval epochs, in order to be
    used for the self-supervised stage of CARLA.
    """
    train_dataset = PretextDataset(
        dataset=dataset,
        window_size=window_size,
        mode='train',
        use_genias=use_genias,
    )
    
    model = PretextModel(
        in_channels=train_dataset.data_dim,
        mid_channels=4,
    )
    
    ckpt_dir = os.path.join(f'checkpoints/carla_pretext/{dataset}')
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(f'cuda:{gpu_num}')

    model = model.to(device)
    criterion = pretextloss(batch_size=batch_size)

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
    print('Pretext training loop start...\n')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} start.')

        cosine_schedule(optimizer=optimizer, current_epoch=epoch)
        epoch_loss = 0.0
        prev_loss = None

        for batch in tqdm(train_loader):
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
        print(f'Epoch {epoch + 1} train loss: {epoch_loss:.4e}')

        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'resnet': model.resnet.state_dict(),
                    'contrastive_head': model.contrastive_head.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )

    print('Pretext training done. Start selecting neighborhoods...\n')
      
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
    num_negative_reps = negative_reps.shape[0]

    index_searcher = IndexFlatL2(reps.shape[1])
    index_searcher.add(reps)

    anchor_and_negative_pairs = np.concatenate(
        [train_dataset.anchors, train_dataset.negative_pairs],
        axis=0,
    )

    classification_data_dir = f'classification_dataset/{dataset}'
    
    if not os.path.exists(classification_data_dir):
        os.makedirs(classification_data_dir, exist_ok=True)

    print(f'\nSelecting top-{num_neighbors} neighbors of the anchor...')

    nearest_neighbors = []
    furthest_neighbors = []

    for i, anchor_rep in tqdm(enumerate(anchor_reps)):
        anchor_query = anchor_rep.reshape(1, -1)
        _, indices = index_searcher.search(anchor_query, reps.shape[0])
        indices = indices.reshape(-1)
        indices = indices[indices != i]
        nearest_indices = indices[:num_neighbors]
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

    for j, negative_rep in tqdm(enumerate(negative_reps)):
        negative_query = negative_rep.reshape(1, -1)
        _, indices = index_searcher.search(negative_query, reps.shape[0])
        indices = indices.reshape(-1)
        indices = indices[indices != j + num_negative_reps]
        nearest_indices = indices[:num_neighbors]
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
    dataset: str,
    window_size: int = 200,
    batch_size: int = 100,
    gpu_num: int = 0,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    model_save_interval: int = 5,
) -> None:
    """
    Training code for CARLA slef-supervised classification stage.

    Parameters:
        dataset:             Name of the dataset.
        window_size:         Window size. Default 200.
        batch_size:          Batch size. Default 100.
        gpu_num:             The model is trained in this GPU. Default 0.
        epochs:              Training epochs. Default 100.
        learning_rate:       The initial learning rate. Default 1e-4.
        model_save_interval: The model is saved once in this epoch. Default 5.

    Uses the pre-trained ResNet model and classification head to map the
    window, nearest neighbors, and furthest neighbors to the C-dimensional
    space, where C is the number of classes that the classification model
    wants to classify data.

    If trained well, the classification model sends the majority of normal data
    to the specific class, namely C_m. In the inference stage, the input from
    the test set is fed to the classification model, and considered normal
    if the probability such that the data is sent to C_m - th class is larger
    than the probabilities such that the data is sent to another class; 
    abnormal otherwise.
    """
    device = torch.device(f'cuda:{gpu_num}')
    
    train_dataset = ClassificationDataset(
        dataset=dataset,
        window_size=window_size,
        mode='train'
    )
    data_dim = train_dataset.data_dim

    resnet_dir = os.path.join('checkpoints/carla_pretext', dataset)
    resnet_ckpt = torch.load(os.path.join(resnet_dir, 'epoch_30.pt'))
    model = ClassificationModel(in_channels=data_dim)
    model.resnet.load_state_dict(resnet_ckpt['resnet'])
    model = model.to(device)

    ckpt_dir = os.path.join(f'checkpoints/carla_classification/{dataset}')
    
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
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt'),
            )
    
    print('Classification training done.')

    model.eval()
    print('\nStarting inference...')

    test_dataset = ClassificationDataset(dataset='MSL', mode='test')
    test_data = torch.tensor(
        data=test_dataset.unprocessed_data,
        dtype=torch.float32
    )
    test_data = test_data.unsqueeze(1).transpose(-2, -1).to(device)
    labels = test_dataset.unprocessed_labels.reshape(-1)

    model.eval()

    logits = model(test_data)
    logits = logits.detach().cpu().numpy()

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
    probas_pred=anomaly_scores,
    )

    auc_pr = auc(recall, precision)

    best_threshold = 0
    best_precision = 0
    best_recall = 0
    best_fl = 0

    for i in range(len(thresholds)):
        f1score = f1(precision[i], recall[i])
        if f1score > best_fl:
            best_fl = f1score
            best_precision = precision[i]
            best_recall = recall[i]
            best_threshold = thresholds[i]

    print('\nResults')

    print(f'\nBest F1 score: {round(best_fl, 4)}')
    print(f'Best Precision: {round(best_precision, 4)}')
    print(f'Best Recall: {round(best_recall, 4)}')
    print(f'Best Threshold: {best_threshold:.4e}')

    print(f'\nAUC-PR: {round(auc_pr, 4)}')

    return 


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--task',
        type=str,
        help="Which model to train. Either 'pretext' or 'classification'."
    )
    args.add_argument(
        '--dataset',
        type=str,
        help="Name of the dataset."
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
        help="Batch size. Default 100."
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="Which GPU will be used. Default 0."
    )
    args.add_argument(
        '--use-genias',
        type=str2bool,
        default=False,
        help="Whether to use GenIAS for generating negative pairs." \
        "Default False."
    )
    args.add_argument(
        '--pretext-epochs',
        type=int,
        default=30,
        help="Training epochs for pretext stage. Default 30."
    )
    args.add_argument(
        '--classification-epochs',
        type=int,
        default=100,
        help="Training epochs for classification stage. Default 100."
    )
    args.add_argument(
        '--pretext-learning-rate',
        type=float,
        default=1e-3,
        help="Initial learning rate. Default 1e-3."
    )
    args.add_argument(
        '--classification-learning-rate',
        type=float,
        default=1e-2,
        help="Initial learning rate. Default 1e-2."
    )
    args.add_argument(
        '--model-save-interval',
        type=int,
        default=5,
        help='The model will be saved once in this epochs.'
    )
    config = args.parse_args()

    assert config.task in ['pretext', 'classification'], \
    "task must be either 'pretext' or 'classification'."

    if config.task == 'pretext':
        pretext(
            dataset=config.dataset,
            window_size=config.window_size,
            batch_size=config.batch_size,
            gpu_num=config.gpu_num,
            epochs=config.pretext_epochs,
            learning_rate=config.pretext_learning_rate,
            model_save_interval=config.model_save_interval,
            use_genias=config.use_genias,
        )

    if config.task == 'classification':
        classification(
            dataset=config.dataset,
            window_size=config.window_size,
            batch_size=config.batch_size,
            gpu_num=config.gpu_num,
            epochs=config.classification_epochs,
            learning_rate=config.classification_learning_rate,
            model_save_interval=config.model_save_interval,
        )
    