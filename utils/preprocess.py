import random
from faiss import IndexFlatL2
from utils.common_import import *
device = torch.device('cuda:0')


####################### GenIAS preprocessing functions########################

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


def convert_to_windows(data: Matrix, window_size: int = 200) -> Array:
    windows = []

    for i in range(data.shape[0] - window_size + 1):
        windows.append(data[i: i + window_size])

    return np.array(windows)


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
    x_tilde_patched = np.empty_like(x) # torch.emty_like(x_tilde)

    for d in range(data_dim):
        x_d = x[:, d]
        x_tilde_d = x_tilde[:, d]
        deviation_d = np.sum((x_d - x_tilde_d) ** 2)
        amplitude_d = np.max(x_d) - np.min(x_d)
        if deviation_d > tau * amplitude_d:
            x_tilde_patched[:, d] = x_tilde[:, d]
        else:
            x_tilde_patched[:, d] = x[:, d]

    return x_tilde_patched

##################### CARLA pretext processing functions #####################

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
    
    return mean, std


def noise_transformation(x: Matrix, sigma: float = 0.01) -> Matrix:
    """
    Inject Gaussian noise with to data.
    Used for choosing positive pair of first 10 windows.
    This is a basic form of data augmentation.
    
    Parameters:
        x:     The data that the noise will be injected.
        sigma: The standard deviation of the noise. Default 0.01.
               The mean of the noise is zero.
    """
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)

    return x + noise


class AnomalyInjection(object):
    """
    This is a modified version of SubAnomaly in CARLA code.
    This if for injectind anomaly to the normal window.
    """
    def __init__(self, portion_len: float = 0.99) -> None:
        self.portion_len = portion_len
        return
    
    def inject_anomaly(
        self,
        window: Matrix,
        subsequence_length: Optional[int] = None,
        compression_factor: Optional[int] = None,
        start_idx: Optional[int] = None,
        trend_end: bool = False,
        scale_factor: Optional[int] = None,
        trend_factor: Optional[float] = None,
        shapelet_factor: bool = False,
    ) -> Matrix:
        """
        
        Parameters:
            window:             The window that the anomaly will be injected.

                                The anomaly-injected-subset of the window will
                                be called 'subsequence' in this description.

            subsequence_length: The length of the subsequence. Default None.
                                If not specified, it is determined as one of
                                the integers between 0.1 * (window length) and
                                0.9 * (window length).

            compression_factor: The degraded subsequence will be shortened to
                                1/(this factor) of its length. Default None.
                                If not specified, it is determined as one of
                                2, 3, 4.

            start_idx:          The anomaly injection starts from this index.
                                That is, the subsequence will be started from
                                this index. Default None. If not specified, it
                                is determined as one of the integers between
                                0 and (window length - subsequence length).

            trend_end:          Determines if that the subsequence lasts to
                                the last index of the window. Default False.

            scale_factor:       The subsequence will be scaled by this value.
                                Default None. If not specified, it is 
                                determined as one of the floats between 0.1 and 
                                2.0.

            trend_factor:       If the type of anomalies that will be added is
                                'trend', then it determines the scale of trend
                                anomaly. Default None. If not stated, it is
                                determined as the value of normal distribution
                                with mean 1 and standard deviation 0.5.

            shapelet_factor:    Whether or not to add shapelet anomaly.
                                Default False.
        """
        window = window.copy()

        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
        
        if start_idx is None:
            start_idx = np.random.randint(0, len(window) - subsequence_length)
        
        end_idx = min(start_idx + subsequence_length, window.shape[0])

        if trend_end:
            end_idx = window.shape[0]

        degraded_subsequence = window[start_idx: end_idx]
        degraded_subsequence = np.tile(
            A=degraded_subsequence,
            reps=(compression_factor, 1)
        )
        degraded_subsequence = degraded_subsequence[::compression_factor]
        degraded_subsequence = degraded_subsequence * scale_factor

        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)

        trend_coef = 1
        random_float = np.random.uniform()

        if random_float < 0.5:
            trend_coef = -1
        
        degraded_subsequence = degraded_subsequence + trend_coef * trend_factor

        if shapelet_factor:
            degraded_subsequence = window[start_idx] \
            + (np.random.random_sample(window[start_idx].shape) * 0.1)

        window[start_idx: end_idx] = degraded_subsequence

        return np.squeeze(window)
    
    def __call__(self, x: Matrix) -> Matrix:
        window = x.copy()

        degraded_window = window.copy()

        min_len = int(window.shape[0] * 0.1)
        max_len = int(window.shape[0] * 0.9)

        subsequence_length = np.random.randint(min_len, max_len)
        start_idx = np.random.randint(0, len(window) - subsequence_length)
        
        anomaly_types = [
            'global',
            'contextual',
            'seasonal',
            'shapelet',
            'trend'
        ]

        anomaly_type = random.choice(anomaly_types)

        if window.ndim > 1:
            num_features = window.shape[1]
            num_dims = np.random.randint(num_features//10, num_features//2)
            for _ in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_window = window[:, i].reshape(window.shape[0], 1)

                if anomaly_type == 'global':
                    degraded_window[:, i] = self.inject_anomaly(
                        window=temp_window,
                        subsequence_length=2,
                        compression_factor=1,
                        start_idx=start_idx,
                        scale_factor=8,
                        trend_factor=0,
                    )
                elif anomaly_type == 'contextual':
                    degraded_window[:, i] = self.inject_anomaly(
                        window=temp_window,
                        subsequence_length=4,
                        compression_factor=1,
                        start_idx=start_idx,
                        scale_factor=3,
                        trend_factor=0,
                    )
                elif anomaly_type == 'seasonal':
                    degraded_window[:, i] = self.inject_anomaly(
                        window=temp_window,
                        subsequence_length=subsequence_length,
                        start_idx=start_idx,
                        scale_factor=1,
                        trend_factor=0,
                    )
                elif anomaly_type == 'shapelet':
                    degraded_window[:, i] = self.inject_anomaly(
                        window=temp_window,
                        subsequence_length=subsequence_length,
                        compression_factor=1,
                        start_idx=start_idx,
                        scale_factor=1,
                        trend_factor=0,
                        shapelet_factor=True,
                    )
                elif anomaly_type == 'trend':
                    degraded_window[:, i] = self.inject_anomaly(
                        window=temp_window,
                        subsequence_length=subsequence_length,
                        compression_factor=1,
                        start_idx=start_idx,
                        trend_end=True,
                        scale_factor=1,
                    )
                
        else:
            temp_window = window.reshape(len(window), 1)

            if anomaly_type == 'global':
                degraded_window = self.inject_anomaly(
                    window=temp_window,
                    subsequence_length=2,
                    compression_factor=1,
                    start_idx=start_idx,
                    scale_factor=8,
                    trend_factor=0,
                )
            elif anomaly_type == 'contextual':
                degraded_window = self.inject_anomaly(
                    window=temp_window,
                    subsequence_length=4,
                    compression_factor=1,
                    start_idx=start_idx,
                    scale_factor=3,
                    trend_factor=0,
                    )
            elif anomaly_type == 'seasonal':
                degraded_window = self.inject_anomaly(
                    window=temp_window,
                    subsequence_length=subsequence_length,
                    start_idx=start_idx,
                    scale_factor=1,
                    trend_factor=0,
                )
            elif anomaly_type == 'shapelet':
                degraded_window = self.inject_anomaly(
                    window=temp_window,
                    subsequence_length=subsequence_length,
                    compression_factor=1,
                    start_idx=start_idx,
                    scale_factor=1,
                    trend_factor=0,
                    shapelet_factor=True,
                )
            elif anomaly_type == 'trend':
                degraded_window = self.inject_anomaly(
                    window=temp_window,
                    subsequence_length=subsequence_length,
                    compression_factor=1,
                    start_idx=start_idx,
                    trend_end=True,
                    scale_factor=1,
                    )

        return degraded_window
