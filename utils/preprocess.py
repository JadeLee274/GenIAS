from utils.common_import import *


def overwrite_nan(data: Matrix) -> Matrix:
    """
    Overwrites NaN-vlaued data with the most recent without-NaN data.

    Parameters:
        data: The data that you want to overwrite.
    """
    for i in range(data.shape[0]):
        if np.any(np.isnan(data[i])):
            data[i] = data[i - 1]
    return data


def drop_anomaly(data: Matrix) -> Matrix:
    """
    Drops anomalies of the training data.

    Parameters:
        data: The data that you want to drop anomaly.

    Retruns:
        The data with dropped anomaly.
    """
    new_data = []
    for i in range(data.shape[0]):
        if data[i, -1] == False:
            new_data.append(data[i])
    new_data = np.stack(new_data)
    return new_data


def overwrite_anomaly(data: Matrix) -> Matrix:
    """
    Overwrites anomalies of the training data with the most recent normal data.

    Parameters:
        data: The data that you want to overwrite.

    Returns:
        The data that the anomaly is overwritten.
    """
    data = data.copy()
    recent_normal = data[0].copy()
    for i in range(1, data.shape[0]):
        if data[i, -1] == True:
            data[i] = recent_normal
        else:
            recent_normal = data[i].copy()
    return data


def mean_std_normalize(x: Matrix, eps: float = 1e-8) -> Matrix:
    """
    Normalize each comlum of data using its mean and standard deviation.

    Parameters:
        x:   Input data.
        eps: Constant that prevents dividing by zero.
        
    Returns:
        Normlaized data with respect to maen and standard deviation.
    """

    for i in range(x.shape[-1]):
        mean = np.mean(x[:, i])
        std = np.std(x[:, i])
        x[:, i] -= mean
        x[:, i] /= (std + eps)
    
    return x


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


def patch(x: Matrix, x_tilde: Matrix, tau: float) -> Matrix:
    """
    Patching function from GenIAS paper, for patching generated anomalies.
    
    Parameters:
        x:       Normal data.
        x_tilde: Generated anomalies through VAE.
        tau:     Coefficient for distance between normal data and anomaly.

    Returns:
        Patched anomaly.

    For each column of generated anomalies, if the amplitude is larger than the
    tau * (distance between the normal data column and the anomaly data colum),
    then the column remains still, converted to normal data otherwise.
    """
    data_dim = x.shape[1]
    x_tilde_patched = torch.empty_like(x) # torch.emty_like(x_tilde)

    for d in range(data_dim):
        x_d = x[:, d]
        x_tilde_d = x_tilde[:, d]
        deviation_d = torch.sum((x_d - x_tilde_d) ** 2)
        amplitude_d = torch.max(x_d) - torch.min(x_d)
        if deviation_d > tau * amplitude_d:
            x_tilde_patched[:, d] = x_tilde[:, d]
        else:
            x_tilde_patched[:, d] = x[:, d]

    return x_tilde_patched
