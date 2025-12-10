import torch.multiprocessing as mp
from exp.utils.common_import import *


def get_mean_std(x: Matrix) -> Vector:
    """
    Gets the column-wise mean and standard deviations of data.

    Parameters:
        x:   Data.

    Returns:
        mean vector and standard deviation vector of x.
    """

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    
    return mean, std


def mean_std_normalize(x: Matrix) -> Matrix:
    """
    Normalize each comlum of data using its mean and standard deviation.

    Parameters:
        x:   Input data.
        eps: Constant that prevents dividing by zero.
        
    Returns:
        Normlaized data with respect to maen and standard deviation.
    """
    mean = np.mean(a=x, axis=0)
    std = np.std(a=x, axis=0)
    std = np.where(std == 0.0, 1.0, std)

    return (x - mean) / std


def min_max_normalize(x: Matrix) -> Matrix:
    """
    Normalize each column of data using minimum and maximum values of this
    column.

    Parameters:
        x: Input data.
    
    Returns:
        Normalized data with respect to maximum and minimum.
    """
    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)
    return (x - min_x) / (max_x - min_x + 1e-4)


def convert_to_windows(data: Matrix, window_size: int) -> Array:
    windows = []

    for i in range(data.shape[0] - window_size + 1):
        windows.append(data[i: i + window_size])

    return np.array(windows, dtype=np.float32)


def patch(
    x: Union[Matrix, Tensor],
    x_pert: Union[Matrix, Tensor],
    tau: float = 0.05,
) -> Union[Matrix, Tensor]:
    data_dim = x.shape[-1]
    
    if isinstance(x, Matrix):
        x_pert_patched = np.empty_like(x)
    elif isinstance(x, Tensor):
        x_pert_patched = torch.empty_like(x)

    for dim in range(data_dim):
        x_d = x[:, dim]
        x_pert_d = x_pert[:, dim]
        if isinstance(x, Matrix):
            deviation = np.sum((x_d - x_pert_d) ** 2)
            amplitude = np.max(x_d) - np.min(x_d)
        elif isinstance(x, Tensor):
            deviation = torch.sum((x_d - x_pert_d) ** 2)
            amplitude = torch.max(x_d) - torch.min(x_d)

        if deviation > tau * amplitude:
            x_pert_patched[:, dim] = x_pert[:, dim]
        else:
            x_pert_patched[:, dim] = x[:, dim]
    
    return x_pert_patched
