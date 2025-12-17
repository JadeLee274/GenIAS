import logging
import warnings
import itertools
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
    seed: int,
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


def str2bool(v: str) -> bool:
    """
    Changes string to bool.

    Parameters:
        v: String. Must be either 'True' or 'False'.
    """
    assert v in ['True', 'False'], "string must be either 'True' or 'False'"

    return v.lower() == 'true'


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1
) -> Tensor:
    if eps != 1e-10:
        warnings.warn("'eps' parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(
            input=logits,
            memory_format=torch.legacy_contiguous_format,
        ).exponential_().log()
    )  # ~ Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~ Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            input=logits,
            memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


def plot_causal_matrix(
    cmtx: Tensor,
    class_names: Optional[List[str]] = None,
    figsize: Optional[List[int]] = None,
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
    show_text: bool = True,
    cmap: str = "magma",
) -> Figure:
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = cmtx.shape[0]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    
    figsize[0] = 30 if figsize[0] > 30 else figsize[0]
    figsize[1] = 20 if figsize[1] > 20 else figsize[1]
    
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] < threshold else "black"
        if cmtx.shape[0] < 20 and show_text:
            plt.text(j, i, format(cmtx[i, j], ".2e") if cmtx[i, j] != 0 else ".",
                    horizontalalignment="center", color=color,)

    plt.tight_layout()
    plt.ylabel("Cause")
    plt.xlabel("Effect")

    return figure


def plot_matrix(
    name: str,
    matrix: Matrix,
    plot_dir: str,
    log_step: int,
    figsize: List[int] = [6, 4],
    cmap: str = 'magma',
) -> None:
    if len(matrix.shape) == 3:
        matrix = np.max(matrix, axis=-1)
    
    figure = plot_causal_matrix(
        matrix,
        figsize=figsize,
        show_text=False,
        cmap=cmap
    )
    figure.savefig(os.path.join(plot_dir, f'{name}_{log_step}.pdf'))

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
