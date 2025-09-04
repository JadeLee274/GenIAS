import argparse
from tqdm import tqdm
from data_factory.loader import CARLADataset
from torch.utils.data import DataLoader
import torch.optim as optim
from carla.model import ContrastiveModel
from utils.common_import import *
from utils.loss import pretextloss, classificationloss


def pretext(
    dataset: str,
    window_size: int = 200,
    batch_size: int = 50,
    gpu_num: int = 0,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    resnet_save_interval: int = 5,
) -> None:
    """
    Training code for CARLA pretext stage.

    Parameters:
        dataset:              Name of the training dataset.
        window_size:          Window size. Default 200.
        batch_size:           Batch size. Default 50.
        gpu_num:              The model is trained in this GPU. Default 0.
        epochs:               Training epoch: Default 30.
        learning_rate:        Learning rate. Default 1e-4.
        resnet_save_interval: The ResNet is saved once in this epoch. 
                              Default 5.

    Uses Resnet model and mlp head to map anchor, positive pair, and negative
    pair to the representation space (with dimension 128, in this case).

    While training, the pretext loss is optimized so that the distance between
    the anchor and the positive pair get smaller, while that of
    the anchor and the negative pair get larger, in the representation space.

    The ResNet part is saved once in a resnet_save_interval epochs, in order to
    be used for the self-supervised stage of CARLA.
    """
    train_dataset = CARLADataset(
        dataset=dataset,
        window_size=window_size,
        mode='train',
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )
    
    device = torch.device(f'cuda:{gpu_num}')
    
    model = ContrastiveModel(
        in_channels=train_dataset.data_dim,
        mid_channels=4,
    ).to(device)
    
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )

    criterion = pretextloss(
        batch_size=batch_size,
    ).to(device)

    ckpt_dir = os.path.join(f'checkpoints/carla_pretext/{dataset}')
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    print('Training loop start...')

    for epoch in range(epochs):
        epoch_loss = 0.0
        prev_loss = None
        print(f'Epoch {epoch + 1} start')

        for data in tqdm(train_loader):
            optimizer.zero_grad()
            anchor, positive_pair, negative_pair = data
            B, W, F = anchor.shape
            anchor = anchor.to(device)
            positive_pair = positive_pair.to(device)
            negative_pair = negative_pair.to(device)
            _inputs = torch.cat([anchor, positive_pair, negative_pair], dim=0)
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
    
    print('Done.')

    return


def classification(
    dataset: str,
    window_size: int = 200,
    batch_size: int = 100,
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
    