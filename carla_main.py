import argparse
from math import cos, pi
from tqdm import tqdm
from faiss import IndexFlatL2
from utils.common_import import *
from data_factory.loader import PretextDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from carla.model import ContrastiveModel
from utils.loss import pretextloss, classificationloss


def str2bool(v: str) -> bool:
    return v.lower() in 'true'


def cosine_scheduler(
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
    resnet_save_interval: int = 5,
    use_genias: bool = False,
    num_neighbors: int = 5,
) -> None:
    """
    Training code for CARLA pretext stage.

    Parameters:
        dataset:              Name of the training dataset.
        window_size:          Window size. Default 200.
        batch_size:           Batch size. Default 50.
        gpu_num:              The model is trained in this GPU. Default 0.
        epochs:               Training epoch: Default 30.
        learning_rate:        Initial learning rate. Default 1e-3.
        resnet_save_interval: The ResNet is saved once in this epoch. 
                              Default 5.
        use_genias:           Whether or not to use GenIAS for creating 
                              negative pair. Default True.
        num_neighbors:        Choose this number of nearese/furthers 
                              neighborhood after the training loop.
                              Default 5.

    Uses Resnet model and mlp head to map anchor, positive pair, and negative
    pair to the representation space (with dimension 128, in this case).

    While training, the pretext loss is optimized so that the distance between
    the anchor and the positive pair get smaller, while that of
    the anchor and the negative pair get larger, in the representation space.

    The ResNet part is saved once in a resnet_save_interval epochs, in order to
    be used for the self-supervised stage of CARLA.
    """
    train_dataset = PretextDataset(
        dataset=dataset,
        window_size=window_size,
        mode='train',
        use_genias=use_genias,
    )
    
    model = ContrastiveModel(
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
    print('Training loop start...')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} start')

        cosine_scheduler(optimizer=optimizer, current_epoch=epoch)
        epoch_loss = 0.0
        prev_loss = None

        for data in tqdm(train_loader):
            optimizer.zero_grad()
            anchor, positive_pair, negative_pair = data
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

        if epoch == 0 or (epoch + 1) % resnet_save_interval == 0:
            torch.save(
                obj={
                    'resnet': model.resnet.state_dict(),
                    'contrastive_head': model.contrastive_head.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )

    print('Done.\n')

    classification_data_dir = f'classification_dataset/{dataset}'
    
    if not os.path.exists(classification_data_dir):
        os.makedirs(classification_data_dir, exist_ok=True) 

    print(f'\nSaving anchor dataset for classification task...')
    np.save(
        file=os.path.join(classification_data_dir, 'anchor.npy'),
        arr=train_dataset.windows,
    )

    print(f'Saving negative pair dataset for classification task...')
    np.save(
        file=os.path.join(classification_data_dir, 'negative_pair.npy'),
        arr=train_dataset.negative_pairs,
    )
       
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

    features = np.concatenate([anchor_features, negative_features], axis=0) 

    print(f'\nSelecting top-{num_neighbors} nearest/furthest neighbors...')

    nearest_neighbors = []
    furthest_neighbors = []

    feature_dim = model.backbone_dim
    index_searcher = IndexFlatL2(feature_dim)
    index_searcher.add(features)

    anchor_and_negative_pairs = np.concatenate(
        [train_dataset.windows, train_dataset.negative_pairs],
        axis=0,
    )

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
    
    print('\nSaving nearest neighborhoods...')
    nearest_neighbors = np.array(nearest_neighbors)
    np.save(
        file=os.path.join(classification_data_dir, 'nearest_neighbors.npy'),
        arr=nearest_neighbors,
    )

    print('Saving furthest neighborhoods...')
    furthest_neighbors = np.array(furthest_neighbors)
    np.save(
        file=os.path.join(classification_data_dir, 'furthest_neighbors.npy'),
        arr=furthest_neighbors,
    )

    print('\nSelecting process done. Move on to the classification stage.')
    
    return


def classification(
    dataset: str,
    window_size: int = 200,
    batch_size: int = 100,
    num_neighbors: int = 10,
    gpu_num: int = 0,
    epochs: int = 100,
    learning_rate: float = 1e-4,
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
        '--classifiation-epochs',
        type=int,
        default=100,
        help="Training epochs for classification stage. Default 100."
    )
    args.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help="Initial learning rate. Default 1e-4."
    )
    args.add_argument(
        '--resnet-save-interval',
        type=int,
        default=5,
        help='The resnet in pretext stage will be saved once in this epochs.'
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
            learning_rate=config.learning_rate,
            resnet_save_interval=config.resnet_save_interval,
            use_genias=config.use_genias,
        )

    if config.task == 'classification':
        classification(
            dataset=config.dataset,
            window_size=config.window_size,
            batch_size=config.batch_size,
            gpu_num=config.gpu_num,
            epochs=config.classification_epochs,
            learning_rate=config.learning_rate,
            resnet_save_interval=config.resnet_save_interval,
        )
    