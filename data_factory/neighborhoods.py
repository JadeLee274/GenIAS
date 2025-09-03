import random
from torch.utils.data import Dataset
from utils.common_import import *
from .loader import CARLADataset
NPTensor = np.ndarray
MSL_PATH = '../data/MSL'
DEVICE = torch.device('cuda:0')


class NoiseTransformation(object):
    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma
    
    def __call__(self, x: Tensor) -> Tensor:
        if x.device.type == 'cuda':
            x = x.cpu()
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)

        return torch.tensor(
            data=x.numpy() + noise,
            dtype=torch.float32,
            device=DEVICE,
        )
    

class SubAnomaly(object):
    def __init__(self, portion_len: int) -> None:
        self.portion_len = portion_len

    def inject_frequency_anomaly(
        self,
        window: Tensor,
        subsequence_length: Optional[int] = None,
        compression_factor: Optional[int] = None,
        scale_factor: Optional[float] = None,
        trend_factor: Optional[float] = None,
        shapelet_factor: bool = False,
        trend_end: bool = False,
        start_idx: Optional[int] = None,
    ) -> Matrix:
        """
        Injects an anomaly into an multivariate time series window by
        manipulating a subsequence of the window.

        Parameters:
            window:
            subsequence_length:
            compressing_factor:
            scale_factor:
            trend_factor:

        Returns:
            Windows with injected anomaly
        """
        window = window.clone()
        
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
            print('test')

        if start_idx is None:
            start_idx = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_idx + subsequence_length, window.shape[0])

        if trend_end:
            end_index = window.shape[0]

        anomalous_subsequence = window[start_idx:end_index]

        anomalous_subsequence = anomalous_subsequence.repeat(
            compression_factor, 1
        )
        anomalous_subsequence = anomalous_subsequence[::compression_factor]
        anomalous_subsequence = anomalous_subsequence * scale_factor

        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1

        if np.random.uniform() < 0.5: coef = -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_idx] \
            + (torch.rand_like(window[start_idx]) * 0.1)

        window[start_idx:end_index] = anomalous_subsequence

        return np.squeeze(window)
    
    def __call__(self, x: Tensor):
        """
        Adding sub anomaly with user-defined portion
        """
        window = x.clone()
        anomaly_seasonal = window.clone() 
        anomaly_trend = window.clone() 
        anomaly_global = window.clone() 
        anomaly_contextual = window.clone() 
        anomaly_shapelet = window.clone() 
        min_len = int(window.shape[0] * 0.1)
        max_len = int(window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_idx = np.random.randint(0, len(window) - subsequence_length)
        if (window.ndim > 1):
            num_features = window.shape[1]
            num_dims = np.random.randint(
                int(num_features/10),
                int(num_features/2)
            )
            for k in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_win = window[:, i].reshape((window.shape[0], 1))
                anomaly_seasonal[:, i] = self.inject_frequency_anomaly(
                    temp_win,
                    scale_factor=1,
                    trend_factor=0,
                    subsequence_length=subsequence_length,
                    start_idx=start_idx
                )

                anomaly_trend[:, i] = self.inject_frequency_anomaly(
                    temp_win,
                    compression_factor=1,
                    scale_factor=1,
                    trend_end=True,
                    subsequence_length=subsequence_length,
                    start_idx=start_idx
                )

                anomaly_global[:, i] = self.inject_frequency_anomaly(
                    temp_win,
                    subsequence_length=2,
                    compression_factor=1,
                    scale_factor=8,
                    trend_factor=0,
                    start_idx=start_idx
                )

                anomaly_contextual[:, i] = self.inject_frequency_anomaly(
                    temp_win,
                    subsequence_length=4,
                    compression_factor=1,
                    scale_factor=3,
                    trend_factor=0,
                    start_idx=start_idx
                )

                anomaly_shapelet[:, i] = self.inject_frequency_anomaly(
                    temp_win,
                    compression_factor=1,
                    scale_factor=1,
                    trend_factor=0,
                    shapelet_factor=True,
                    subsequence_length=subsequence_length,
                    start_idx=start_idx
                )

        else:
            temp_win = window.reshape((len(window), 1))
            anomaly_seasonal = self.inject_frequency_anomaly(
                temp_win,
                scale_factor=1,
                trend_factor=0,
                subsequence_length=subsequence_length,
                start_idx=start_idx
            )

            anomaly_trend = self.inject_frequency_anomaly(
                temp_win,
                compression_factor=1,
                scale_factor=1,
                trend_end=True,
                subsequence_length=subsequence_length,
                start_idx=start_idx
            )

            anomaly_global = self.inject_frequency_anomaly(
                temp_win,
                subsequence_length=3,
                compression_factor=1,
                scale_factor=8,
                trend_factor=0,
                start_idx=start_idx)

            anomaly_contextual = self.inject_frequency_anomaly(
                temp_win,
                subsequence_length=5,
                compression_factor=1,
                scale_factor=3,
                trend_factor=0,
                start_idx=start_idx)

            anomaly_shapelet = self.inject_frequency_anomaly(
                temp_win,
                compression_factor=1,
                scale_factor=1,
                trend_factor=0,
                shapelet_factor=True,
                subsequence_length=subsequence_length,
                start_idx=start_idx
            )

        anomalies = [
            anomaly_seasonal,
            anomaly_trend,
            anomaly_global,
            anomaly_contextual,
            anomaly_shapelet
        ]

        anomalous_window = random.choice(anomalies)

        return anomalous_window
    

class MSLDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[NoiseTransformation] = None,
        subanomaly: Optional[SubAnomaly] = None,
        mean_data: Optional[Matrix] = None,
        std_data: Optional[Matrix] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.subanomaly = subanomaly

        label = None
        
        if train:
            data = np.load(os.path.join(MSL_PATH, 'MSL_train.npy'))
            label = np.zeros(data.shape[0], dtype=np.int32)
        else:
            data = np.load(os.path.join(MSL_PATH, 'MSL_test.npy'))
            label = np.load(os.path.join(MSL_PATH, 'MSL_test_label.npy'))
        
        self.mean = mean_data
        self.std = std_data

        if np.any(sum(np.isnan(data)) != 0):
            data = np.nan_to_num(data)
        
        if train:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        else:
            self.mean = mean_data
            self.std = std_data
            self.std[self.std == 0.0] = 1.0
            data = (data - self.mean) / self.std
        
        data = np.asarray(data)

        self.data, self.targets = self.convert_to_windows(
            data=data,
            label=label,
            window_size=200,
        )

    def __getitem__(self, idx: int) -> Tuple[Matrix, ]:
        window = torch.from_numpy(self.data[idx]).float().to(DEVICE)

        if len(self.targets) > 0:
            target = torch.tensor(
                data=self.targets[idx].astype(int),
                dtype=torch.long
            ).to(DEVICE)

        return window, target

    def convert_to_windows(
        self,
        data: Matrix,
        label: Union[Vector, Matrix],
        window_size: int
    ) -> Tuple[NPTensor, Vector]:
        windows = []
        window_labels = []

        for i in range(data.shape[0] - window_size):
            window = data[i: i + window_size]
            window_label = 0
            if sum(label[i: i + window_size]) > 0:
                window_label = 1
            
            windows.append(window)
            window_labels.append(window_label)
        
        return np.stack(windows), np.stack(window_labels)
    

class AugmentedDataset(Dataset):
    def __init__(self, dataset: MSLDataset) -> None:
        super().__init__()
        self.samples = []
        transform = dataset.transform
        sub_anomaly = dataset.subanomaly
        dataset.transform = None
        self.datset = dataset
     if isin


        
        



class NeighborsDataset(Dataset):
    def __init__(
        self,
        dataset: CARLADataset,
        transform: Union[Dict[str, Any], NoiseTransformation],
        num_neighbors: int,
        nearset_indices: Vector,
        furthest_indices: Vector,
    ) -> None:
        super().__init__()
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighborhood_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighborhood_transform = transform
        
        all_data = dataset.windows.to(device)
        self.dataset = dataset

        self.nearest_neighbor_indices = nearset_indices[:, :num_neighbors]
        self.furthest_neighbor_indices = furthest_indices[:, -num_neighbors:]

        self.dataset.data = dataset.windows.to(device)
        num_samples = self.dataset.data_len

        nearest_neighbor_index = np.array(
            [np.random.choice(self.nearest_neighbor_indices[i], 1) 
             for i in range(num_samples)]
        )
        furthest_neighbor_index = np.array(
            [np.random.choice(self.furthest_neighbor_indices[i], 1)
             for i in range(num_samples)]
        )

        self.nearest_neighbor = all_data[nearest_neighbor_index]
        self.furthest_neighbor = all_data[furthest_neighbor_index]

        return

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Matrix, Matrix, Matrix, Vector, Vector]:
        anchor = self.dataset.__getitem__(idx)
        nearest_neighbor = self.nearest_neighbor.__getitem__(idx)
        furthest_neighbor = self.furthest_neighbor.__getitem__(idx)
        possible_nearest_neighbors = torch.from_numpy(
            self.nearest_neighbor_indices[idx]
        )
        possible_furthest_neighbors = torch.from_numpy(
            self.furthest_neighbor_indices[idx]
        )

        return anchor, nearest_neighbor, furthest_neighbor, \
        possible_nearest_neighbors, possible_furthest_neighbors 
