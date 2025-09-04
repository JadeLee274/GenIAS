import argparse
from math import cos, pi
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.common_import import *
from data_factory.dataset import *
from carla.model import ContrastiveModel
from utils.loss import pretextloss


def cosine_scheduler(
    optimizer: optim.Adam,
    current_epoch: int,
    total_epochs: int = 30,
    learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.01,
) -> None:
    """
    Customized cosine scheduler.
    """
    eta_min = learning_rate * (lr_decay_rate ** 3)
    scheduled_learning_rate = eta_min \
    + (learning_rate-eta_min) * (1 + cos(pi*current_epoch/total_epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduled_learning_rate

    return 


def pretext(
    data_dim: int = 55,
    epochs: int = 30,
    gpu_num: int = 0,
    batch_size: int = 50,
    initial_leraning_rate: float = 1e-3,
    checkpoint_save_interval: int = 5,
    retrain: bool = False,
    retrain_start_epoch: Optional[int] = None,
    noise_sigma: float = 0.01,
    sub_anomaly_portion_len: float = 0.99,
    skip_training: bool = False,
    topk: int = 10,
) -> None:
    """
    Training code for CARLA pretext stage.

    Parameters:
        data_dim:                 Dimension of data.
        epochs:                   Training epochs. Default 30.
        gpu_num:                  Training will be on this GPU. Default 0.
        batch_size:               Batch size. Default 50.
        initial_leraning_rate:    Initial learning rate. Default 1e-3.
        checkpoint_save_interval: Model and optimizers are saved once in this
                                  epochs. Default 5.
        retrain:                  Whether to retrain or not. Default False.
        retrain_start_epoch:      If retrain, the training will be started from
                                  this epoch. Default None.
        noise_sigma:              The level of noise that will be used in the
                                  basic augmentation of training data.
                                  Default 0.01.
        sub_anomaly_portion_len:  Anoamly portion length that will be used in
                                  the noise injection stage. Default 0.99.
        skip_training:            If you only want to get nearest/furthest
                                  neighborhoods by using pre-trained model,
                                  set it to True. Default False.
        topk:                     The number of nearest/furthest neighbors for
                                  each windows. Deafault 10.


    """
    if gpu_num == 0:
        pretext_device = DEVICE
    else:
        pretext_device = torch.device(f'cuda:{gpu_num}')
    
    model = ContrastiveModel(in_channels=data_dim)
    transform = NoiseTransformation(sigma=noise_sigma)
    sub_anomaly = SubAnomaly(portion_len=sub_anomaly_portion_len)

    train_dataset = dataset(
        transform=transform,
        sub_anomaly=sub_anomaly,
        data='msl',
        mode='augment',
    )

    val_dataset = dataset(
        transform=transform,
        sub_anomaly=sub_anomaly,
        mode='augment',
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    base_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = pretextloss(
        batch_size=batch_size,
        temperature=0.4,
    ).to(pretext_device)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=initial_leraning_rate,
    )

    ckpt_dir = os.path.join('checkpoints/carla_pretext')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    
    if not skip_training:
        start_epoch = 0

        if retrain:
            start_epoch = retrain_start_epoch
            ckpt = torch.load(
                f=os.path.join(ckpt_dir, f'epoch_{start_epoch}.pt')
            )
            model.resnet.load_state_dict(ckpt['resnet'])
            model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
            optimizer.load_state_dict(ckpt['optim'])
            print(f'Pretext training restart from epoch {start_epoch + 1}...')
        else:
            print('Pretext training start...')

        model = model.to(pretext_device)
        prev_loss = None

        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1} Start.")
            cosine_scheduler(optimizer=optimizer, current_epoch=epoch)
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                window, positive_pair, negative_pair, _ = batch
                window = window.to(pretext_device)
                positive_pair = positive_pair.to(pretext_device)
                negative_pair = negative_pair.to(pretext_device)
                
                B, W, F = window.shape
                x = torch.cat(
                    tensors=[window, positive_pair, negative_pair],
                    dim=0,
                ).view(3*B, F, W)
                features = model.forward(x)

                if prev_loss:
                    loss = criterion.forward(
                        features=features,
                        current_loss=prev_loss,
                    )
                else:
                    loss = criterion.forward(features=features)
                
                loss.backward()
                optimizer.step()
                prev_loss = loss.item()
                epoch_loss += prev_loss
            
            epoch_loss /= len(train_loader)
            
            print(f"Epoch {epoch + 1} finished with Loss: {epoch_loss:.4e}")

            if epoch == 0 or (epoch + 1) % checkpoint_save_interval == 0:
                torch.save(
                    obj={
                        'resnet': model.resnet.state_dict(),
                        'contrastive_head': model.contrastive_head.state_dict(),
                        'optim': optimizer.state_dict(),
                    },
                    f=os.path.join(ckpt_dir, 'MSL', f'epoch_{epoch + 1}.pt')
                )

        print('Pretext training done.\n')
    
    else:
        print('Skipping pretext training.')
        model = ContrastiveModel(in_channels=55)
        ckpt = torch.load(
            f=os.path.join(ckpt_dir, f'epoch_{epochs}.pt')
        )
        model.resnet.load_state_dict(ckpt['resnet'])
        model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
        model = model.to(pretext_device)

    print('Selecting nearest/furthest neighborhoods...\n')

    basic_time_series_repository = TimeSeriesRepository(
        features_len=len(train_dataset),
        features_dim=8,
        num_classes=10,
        temperature=0.4,
    )
    basic_time_series_repository.to(pretext_device)

    augmented_time_series_repository = TimeSeriesRepository(
        features_len=2*len(train_dataset),
        features_dim=8,
        num_classes=10,
        temperature=0.4,
    )
    augmented_time_series_repository.to(pretext_device)

    val_time_series_repository = TimeSeriesRepository(
        features_len=len(val_dataset),
        features_dim=8,
        num_classes=10,
        temperature=0.4,
    )
    val_time_series_repository.to(pretext_device)

    fill_timeseries_repository(
        dataloader=base_loader,
        model=model,
        timeseries_repository=basic_time_series_repository,
        real_augmentation=True,
        timeseries_repository_augmentation=augmented_time_series_repository,
    )

    out_pretext = np.column_stack(
        tup=(
            basic_time_series_repository.features.cpu().numpy(),
            basic_time_series_repository.targets.cpu().numpy(),
        ),
    )

    pretext_save_path = 'carla_loader/msl'
    if not os.path.exists(pretext_save_path):
        os.makedirs(pretext_save_path, exist_ok=True)

    np.save(
        file=os.path.join(pretext_save_path, 'pretext_features_train.npy'),
        arr=out_pretext,
    )

    k_nearest, k_furthest \
    = augmented_time_series_repository.mine_neighborhoods(topk=topk)

    np.save(
        file=os.path.join(pretext_save_path, 'k_nearset_neighbors_train.npy'),
        arr=k_nearest,
    )
    np.save(
        file=os.path.join(
            pretext_save_path,
            'k_furthest_neighbors_train.npy'
        ),
        arr=k_furthest,
    )
    
    print(f'Saved nearset/furthest neighborhoods to {pretext_save_path}.\n')

    print('Selecting nearest/furthest neighborhoods for validation in \
    classification stage...\n')

    fill_timeseries_repository(
        dataloader=val_loader,
        model=model,
        timeseries_repository=val_time_series_repository,
        real_augmentation=False,
        timeseries_repository_augmentation=None,
    )

    out_pretext = np.column_stack(
        tup=(
            val_time_series_repository.features.cpu().numpy(),
            val_time_series_repository.targets.cpu().numpy(),
        ),
    )

    np.save(
        file=os.path.join(pretext_save_path, 'pretext_features_test.npy'),
        arr=out_pretext
    )

    k_nearest, k_furthest \
    = val_time_series_repository.mine_neighborhoods(topk=topk)

    np.save(
        file=os.path.join(pretext_save_path, 'k_nearset_neighbors_tset.npy'),
        arr=k_nearest,
    )
    np.save(
        file=os.path.join(
            pretext_save_path,
            'k_furthest_neighbors_test.npy'
        ),
        arr=k_furthest,
    )


def str2bool(v: str) -> bool:
    return v.lower() in 'true'


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--mode',
        type=str,
        default='pretext',
        help="Mode of CARLA. Either 'pretext' or 'classification'."
    )
    args.add_argument(
        '--dataset',
        type=str,
        help="Name of the dataset."
    )
    args.add_argument(
        '--epochs',
        type=int,
        default=30,
        help="Training epochs. Default 30. \
             If the mode is 'classification, set it to 100."
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="The GPU that you want to use. Default 0."
    )
    args.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help="Batch size. Default 50."
    )
    args.add_argument(
        '--initial-learning-rate',
        type=float,
        default=1e-3,
        help="The initial learning rate. Default 1e-3."
    )
    args.add_argument(
        '--checkpoint-save-interval',
        type=int,
        default=5,
        help="The model and optimizer are saved once in this epoch."
    )
    args.add_argument(
        '--retrain',
        type=str2bool,
        default=False,
        help="Whether to retrain or not. Default False."
    )
    args.add_argument(
        '--retrain-start-epoch',
        type=Optional[int],
        default=None,
        help="If retrain is True, then training is started from this epoch."
    )
    args.add_argument(
        '--skip-training',
        type=str2bool,
        default=False,
        help="In pretext stage, if True, then training is skipped."
    )
    config = args.parse_args()
    
    pretext(
        epochs=config.epochs,
        gpu_num=config.gpu_num,
        batch_size=config.batch_size,
        initial_leraning_rate=config.initial_learning_rate,
        retrain=config.retrain,
        retrain_start_epoch=config.retrain_start_epoch,
        skip_training=config.skip_training,
    )
    