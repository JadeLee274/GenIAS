import argparse
from math import cos, pi
from tqdm import tqdm
from faiss import IndexFlatL2
from torch.nn import Softmax
from sklearn.metrics import f1_score, roc_auc_score, \
                            precision_recall_curve, auc
from utils.common_import import *
from data_factory.loader import PretextDataset, ClassificationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from carla.model import PretextModel, ClassificationModel
from utils.loss import pretextloss, classificationloss, entropy


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
    criterion = pretextloss(batch_size=batch_size).to(device)

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

            _inputs = torch.cat(
                tensors=[anchor, positive_pair, negative_pair],
                dim=0
            ).float()

            _inputs = _inputs.view(3 * B, F, W)
            _features = model(_inputs)
            loss = criterion.forward(
                features=_features,
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

    print('Pretext training done.\n')
      
    resnet = model.resnet
    resnet.eval()

    timeseries_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    anchor_features = []
    negative_features = []

    print('Loading features of each anchor and its negative pair.')

    for batch in tqdm(timeseries_loader):
        anchor, _, negative_pair = batch
        anchor = anchor.to(device).float().transpose(1, 2)
        negative_pair = negative_pair.to(device).float().transpose(1, 2)
        anchor_feature = resnet(anchor).detach().cpu()
        negative_feature = resnet(negative_pair).detach().cpu()
        anchor_features.append(anchor_feature)
        negative_features.append(negative_feature)
    
    anchor_features = torch.cat(anchor_features, dim=0).numpy()
    negative_features = torch.cat(negative_features, dim=0).numpy()

    feature_dim = model.feature_dim
    index_searcher = IndexFlatL2(feature_dim)
    features = np.concatenate([anchor_features, negative_features], axis=0)
    index_searcher.add(features)

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

    for anchor_feature in tqdm(anchor_features):
        query = anchor_feature.reshape(1, -1)
        _, distance_based_indices = index_searcher.search(query, len(features))
        distance_based_indices = distance_based_indices.reshape(-1)
        nearest_indices = distance_based_indices[:num_neighbors]
        furthest_indices = distance_based_indices[-num_neighbors:]
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

    for anchor_feature in tqdm(negative_features):
        query = anchor_feature.reshape(1, -1)
        _, distance_based_indices = index_searcher.search(query, len(features))
        distance_based_indices = distance_based_indices.reshape(-1)
        nearest_indices = distance_based_indices[:num_neighbors]
        furthest_indices = distance_based_indices[-num_neighbors:]
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

    print('Classification training loop start...\n')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} start.')
        epoch_loss = 0.0
        epoch_anchor_consistency_loss = 0.0
        epoch_anchor_inconsistency_loss = 0.0
        epoch_negative_consistency_loss = 0.0
        epoch_negative_inconsistency_loss = 0.0
        epoch_entropy_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=device)

            anchor, anchor_nn, anchor_fn, negative, negative_nn, negative_fn \
            = batch

            anchor = anchor.to(device).float()
            anchor_nn = anchor_nn.to(device).float()
            anchor_fn = anchor_fn.to(device).float()
            negative = negative.to(device).float()
            negative_nn = negative_nn.to(device).float()
            negative_fn = negative_fn.to(device).float()

            anchor = anchor.transpose(-2, -1)
            negative = negative.transpose(-2, -1)
            anchor_softmax = model.forward(anchor)
            negative_softmax = model.forward(negative)

            anchor_entropy_loss = entropy(anchor_softmax)
            negative_entropy_loss = entropy(negative_softmax)

            batch_loss += anchor_entropy_loss
            batch_loss += negative_entropy_loss

            epoch_entropy_loss += anchor_entropy_loss.item()
            epoch_entropy_loss += negative_entropy_loss.item()

            anchor_total_consistency = torch.zeros(1, device=device)
            negative_total_consistency = torch.zeros(1, device=device)

            anchor_consistency = 0.0
            anchor_inconsistency = 0.0
            negative_consistency = 0.0
            negative_inconsistency = 0.0

            for i in range(anchor_nn.shape[1]):
                anchor_nearest = anchor_nn[:, i].transpose(-2, -1)
                anchor_furthest = anchor_fn[:, i].transpose(-2, -1)
                negative_nearest = negative_nn[:, i].transpose(-2, -1)
                negative_furthest = negative_fn[:, i].transpoze(-2, -1)

                anchor_nearest_softmax = model.forward(anchor_nearest)
                anchor_furthest_softmax = model.forward(anchor_furthest)
                negative_nearest_softmax = model.forward(negative_nearest)
                negative_furthest_softmax = model.forward(negative_furthest)

                anchor_total_consist, anchor_consist, anchor_inconsist \
                = criterion(
                    anchor_softmax=anchor_softmax,
                    nearest_softmax=anchor_nearest_softmax,
                    furthest_softmax=anchor_furthest_softmax,
                )

                negative_total_consist, negative_consist, negative_inconsist \
                = criterion(
                    anchor_softmax=negative_softmax,
                    nearest_softmax=negative_nearest_softmax,
                    furthest_softmax=negative_furthest_softmax,
                )
                
                anchor_total_consistency += anchor_total_consist
                anchor_consistency += anchor_consist
                anchor_inconsistency += anchor_inconsist

                negative_total_consistency += negative_total_consist
                negative_consistency += negative_consist
                negative_inconsistency += negative_inconsist
            
            batch_loss += anchor_total_consistency
            batch_loss += negative_total_consistency

            epoch_anchor_consistency_loss += anchor_consistency
            epoch_anchor_inconsistency_loss += anchor_inconsistency
            epoch_negative_consistency_loss += negative_consistency
            epoch_negative_inconsistency_loss += negative_inconsistency

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        epoch_loss /= len(train_loader)
        epoch_anchor_consistency_loss /= len(train_loader)
        epoch_anchor_inconsistency_loss /= len(train_loader)
        epoch_negative_consistency_loss /= len(train_loader)
        epoch_negative_inconsistency_loss /= len(train_loader)
        epoch_entropy_loss /= len(train_loader)
        
        print(f'Epoch {epoch + 1} finished.')
        print(f' Anchor Consistency: {epoch_anchor_consistency_loss:.4e}')
        print(f' Anchor Inconsistency: {epoch_anchor_inconsistency_loss:.4e}')
        print(f' Negative Consistency: {epoch_negative_consistency_loss:.4e}')
        print(f' Negative Inonsistency: {epoch_negative_inconsistency_loss:.4e}')
        print(f' Entropy loss: {epoch_entropy_loss:.4e}')
        print(f' Total loss: {epoch_loss:.4e}\n')


        if epoch == 0 or (epoch + 1) % model_save_interval == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt'),
            )
    
    print('Classification training done.')

    print('\nStarting inference...')

    softmax = Softmax(dim=1)

    test_dataset = ClassificationDataset(dataset='MSL', mode='test')
    test_data = torch.tensor(
        data=test_dataset.unprocessed_data,
        dtype=torch.float32
    )
    test_data = test_data.unsqueeze(1).transpose(-2, -1).to(device)
    labels = test_dataset.unprocessed_labels.reshape(-1)

    model.eval()

    logits = model(test_data)
    logits = softmax(logits)
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

    f1 = f1_score(
        y_true=labels,
        y_pred=anomaly_labels,
    )

    auc_roc = roc_auc_score(
        y_true=labels,
        y_score=anomaly_scores,
    )

    precision, recall, _ = precision_recall_curve(
        y_true=labels,
        probas_pred=anomaly_scores,
    )

    auc_pr = auc(recall, precision)

    print('\nResults')
    print(f'- F1 Score: {round(f1, 4)}')
    print(f'- AUC-ROC: {round(auc_roc, 4)}')
    print(f'- AUC-PR: {round(auc_pr, 4)}')

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
        '--classifiation-epochs',
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
    