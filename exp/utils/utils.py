import logging
import matplotlib.pyplot as plt
from exp.utils.common_import import *
from exp.data_factory.loader import PerturbationDataset
from exp.models.augmentor import TCNPerturbator


def fix_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return


def set_logging_file(
    exp_name: str,
    log_file_path: str,
    time: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
) -> None:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    format = '%(asctime)s %(message)s'
    logging.basicConfig(
        format=format,
        handlers=[
            logging.FileHandler(
                filename=log_file_path,
                mode=mode,
                encoding=encoding,
            ),
            logging.StreamHandler(),
        ],
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logging.info(f'Logging (File + Stream) Initialized...\n')
    logging.info(f'Experiment name: {exp_name}')
    logging.info(f'Experiment time: {time}\n')

    return


def plot_perturbation(
    time: str,
    data: str,
    subdata: Optional[str],
    window_size: int,
    epoch: int,
    figsize: Tuple[int, int] = (15, 20),
    ylim_upper: int = 10,
) -> None:
    dataset = PerturbationDataset(
        data=data,
        subdata=subdata,
        mode='train',
        window_size=window_size,
    )
    data_dim = dataset.data_dim

    ver_num = data_dim // 5

    save_path = os.path.join('exp', 'figs', 'perturbator', time)
    
    if subdata is not None:
        save_path = os.path.join(save_path, subdata)

    save_path = os.path.join(save_path, f'epoch_{epoch}')
    os.makedirs(save_path, exist_ok=True)

    perturbator = TCNPerturbator(
        window_size=window_size,
        data_dim=data_dim,
        latent_perturbator_factor=2.0,
    )
    perturbator.init_perturbator(
        time=time,
        data=data,
        subdata=subdata,
        epoch=epoch,
    )

    for i in range(len(dataset) // window_size):
        idx = i * window_size
        data = dataset[idx]
        data: Tensor = torch.tensor(data).float().unsqueeze(0)

        _, pert, _, _ = perturbator.forward(data)

        data = data.squeeze(0).detach().numpy()
        pert = pert.squeeze(0).detach().numpy()
        
        fig, axes = plt.subplots(ver_num, 5, figsize=figsize)
        axes = axes.flatten()

        for i in range(data_dim):
            axes[i].plot(data[:, i])
            axes[i].set_title(f"Dim {i+1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_ylim(-ylim_upper, ylim_upper)
            axes[i].set_yticks([-ylim_upper, ylim_upper])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'data_{idx}.png'))
        plt.close()

        fig, axes = plt.subplots(ver_num, 5, figsize=figsize)
        axes = axes.flatten()

        for i in range(data_dim):
            axes[i].plot(pert[:, i])
            axes[i].set_title(f"Dim {i+1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_ylim(-ylim_upper, ylim_upper)
            axes[i].set_yticks([-ylim_upper, ylim_upper])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'neg_pair_{idx}.png'))
        plt.close()

    return
