from exp.utils.common_import import *


def train_val_split(
    train_data: Matrix,
    train_ratio: float,
) -> Tuple[Matrix, Matrix]:
    train_data_len = int(len(train_data) * train_ratio)
    val_data = train_data[train_data_len:].copy()
    train_data = train_data[:train_data_len]
    return train_data, val_data


def downsample_data(data: Matrix, downsample_step: int) -> Matrix:
    assert data.ndim == 2, f"data must be 2-dimensional."
    
    data_downsampled = np.empty(
        [data.shape[0]//downsample_step, data.shape[1]],
        dtype=data.dtype,
    )

    for idx in range(len(data_downsampled)):
        data_downsampled[idx] = np.median(
            data[downsample_step*idx: downsample_step*(idx+1)],
            axis=0,
        )
    
    return data_downsampled


def convert_to_windows(data: Matrix, window_size: int) -> Array:
    windows = []

    for i in range(data.shape[0] - window_size + 1):
        windows.append(data[i: i + window_size])

    return np.array(windows, dtype=np.float32)


def delete_constant_dim_perturbation(x: Matrix, x_pert: Matrix) -> Matrix:
    data_dim = x.shape[-1]
    x_pert_temp = np.empty_like(x)
    
    for d in range(data_dim):
        x_d = x[:, d]
        x_pert_d = x_pert[:, d]
        if np.max(x_d) == np.min(x_d):
            x_pert_temp[:, d] = x_d
        else:
            x_pert_temp[:, d] = x_pert_d
    
    return x_pert_temp


def patch(
    x: Matrix,
    x_pert: Matrix,
    amplitude_list: List[float],
    amplitude_pert_list: List[float],
    non_constant_dim_tau: float,
    constant_dim_tau: float,
) -> Matrix:
    data_dim = x.shape[-1]
    x_pert_temp = np.empty_like(x)

    for dim in range(data_dim):
        x_d = x[:, dim]
        x_pert_d = x_pert[:, dim]
        x_pert_temp_d = x_pert_temp[:, dim]
        amplitude_d = amplitude_list[dim]
        amplitude_pert_d = amplitude_pert_list[dim]

        for i in range(len(x_d)):
            point_i = x_d[i]
            point_i_pert = x_pert_d[i]
            deviation_nonconstant = (point_i - point_i_pert) ** 2
            deviation_nonconstant = np.abs(point_i - point_i_pert)

            # Patching on non-constant dimension
            if amplitude_d != 0:
                if deviation_nonconstant > non_constant_dim_tau * amplitude_d:
                    x_pert_temp_d[i] = point_i_pert
                else:
                    x_pert_temp_d[i] = point_i
            # Patching on constant dimension
            elif amplitude_d == 0:
                if deviation_nonconstant > constant_dim_tau * amplitude_pert_d:
                    x_pert_temp_d[i] = point_i_pert
                else:
                    x_pert_temp_d[i] = point_i
    
    return x_pert_temp
