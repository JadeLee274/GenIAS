import itertools
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from cuts_plus.utils.imports import *


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    logging.info(f'Experiment time: {time}')
    logging.info(f'Seed: {seed}')

    return


def plot_causal_matrix(
    cmtx: Tensor,
    show_text: bool,
    epoch: int,
    class_names: Optional[List[str]] = None,
    figsize: Optional[List[int]] = None,
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
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
    plt.title(f"Epoch {epoch} causal matrix")
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
    epoch: int,
    figsize: List[int] = [6, 4],
) -> None:
    if len(matrix.shape) == 3:
        matrix = np.max(matrix, axis=-1)
    
    figure: Figure = plot_causal_matrix(
        matrix,
        show_text=False,
        epoch=epoch,
        figsize=figsize,
        epoch=epoch,
    )
    figure.savefig(
        os.path.join(plot_dir, f'{name}_{log_step}_epoch_{epoch}.pdf')
    )

    return

    