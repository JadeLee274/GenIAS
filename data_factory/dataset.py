import random
from torch.utils.data import Dataset, DataLoader
from faiss import IndexFlatL2
from utils.common_import import *
from data_factory.loader import CARLADataset
from carla.model import ContrastiveModel
NPTensor = np.ndarray
MSL_PATH = '/data/seungmin/MSL'
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
    def __init__(self, portion_len: float = 0.99) -> None:
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
                    start_idx=start_idx,
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
        train: bool,
        transform: NoiseTransformation = NoiseTransformation(sigma=0.01),
        subanomaly: SubAnomaly = SubAnomaly(portion_len=0.99),
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
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
        self.dataset = dataset
        self.mean = None
        self.std = None
        
        self.ts_transform = NoiseTransformation(sigma=0.01)
        self.augment_transform = NoiseTransformation(sigma=0.01)
        self.subsequence_anomaly = sub_anomaly
        
        self.create_pairs()

        return
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.samples[idx]

    def create_pairs(self) -> None:
        mean = self.dataset.mean
        mean = torch.tensor(mean, dtype=torch.float32).to(DEVICE)

        std = self.dataset.std
        std = torch.tensor(std, dtype=torch.float32).to(DEVICE)

        for idx in range(len(self.dataset)):
            window, target = self.dataset.__getitem__(idx)
            window = window.clone().detach().to(DEVICE)
            target = target.clone().detach().to(DEVICE)

            if idx > 10:
                positive_pair_idx = np.random.randint(idx - 10, idx)
                positive_pair, _ = self.dataset.__getitem__(positive_pair_idx)
                positive_pair = positive_pair.clone().detach().to(DEVICE)
            else:
                positive_pair = self.augment_transform(window)
            
            negative_pair = self.subsequence_anomaly(window)
            std = torch.where(
                condition=std == 0.0,
                input=torch.tensor(1.0, device=std.device),
                other=std,
            )

            normalized_window = (window - mean) / std
            normalized_positive_pair = (positive_pair - mean) / std
            normalized_negative_pair = (negative_pair - mean) / std

            self.samples.append(
                (
                    normalized_window,
                    normalized_positive_pair,
                    normalized_negative_pair,
                    target
                )
            )

        return    


class NeighborsDataset(Dataset):
    def __init__(
        self,
        dataset: CARLADataset,
        transform: NoiseTransformation,
        num_neighbors: int,
        nearset_indices: Vector,
        furthest_indices: Vector,
    ) -> None:
        super().__init__()
        self.mean = None
        self.std = None

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighborhood_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighborhood_transform = transform
        
        all_data = dataset.windows.to(DEVICE)
        self.dataset = dataset

        self.nearest_neighbor_indices = nearset_indices[:, :num_neighbors]
        self.furthest_neighbor_indices = furthest_indices[:, -num_neighbors:]

        self.dataset.data = dataset.windows.to(DEVICE)
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
    

class TimeSeriesRepository(object):
    def __init__(
        self,
        features_len: int,
        features_dim: int,
        num_classes: int = 10,
        temperature: float = 0.4,
    ) -> None:
        self.features_len = features_len
        self.features_dim = features_dim
        self.features = torch.FloatTensor(self.features_len, self.features_dim)
        self.targets = torch.LongTensor(self.features_len)
        self.temperature = temperature
        self.num_classes = num_classes

        self.ptr = 0

        return

    def mine_neighborhoods(self, topk: int) -> Tuple[Matrix, Matrix]:
        index_searcher = IndexFlatL2(d=self.features_dim)
        index_searcher.add(self.features.cpu().numpy())
        query = np.random.random(self.features_dim)
        _, indices = index_searcher.search(
            query.reshape(1, -1).astype(np.flaot32),
            self.features_len,
        )
        topk = indices.shape[1]
        k_nearest_neighbors = indices.reshape(topk, 1)
        k_furthest_neighbors = indices.reshape(topk, 1)[::-1]

        return k_nearest_neighbors, k_furthest_neighbors
    
    def reset(self) -> None:
        self.ptr = 0
        return
    
    def resize(self, size: int) -> None:
        self.features_len = size * self.features_len
        self.featrues = torch.FloatTensor(self.features_len, self.features_dim)
        self.targets = torch.LongTensor(self.features_len)
        return
    
    def update(self, features: Tensor, targets: Tensor) -> None:
        batch_size = features.size(0)    
        assert (batch_size + self.ptr <= self.features_len)
        
        self.features[self.ptr: self.ptr+batch_size].copy_(features.detach())

        if not torch.is_tensor(targets):
            targets = torch.from_numpy(targets)

        self.targets[self.ptr: self.ptr+batch_size].copy_(targets.detach())
        self.ptr += batch_size

        return
    
    def to(self, device: torch.device) -> None:
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device
        return

    def cpu(self) -> None:
        self.to('cpu')
        return
    
    def cuda(self) -> None:
        self.to('cuda')
        return
    

class SaveAugmentedDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.classes = [
            'Normal',
            'Anomaly',
            'Noise',
            'Point',
            'Subseq',
            'Subseq2'
        ]
        self.targets = target
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ts = self.data[index]

        if len(self.targets) > 0:
            target = int(self.targets[index])
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        out = ts, target

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)
    

@torch.no_grad()
def fill_timeseries_repository(
    dataloader: DataLoader,
    model: ContrastiveModel,
    timeseries_repository: TimeSeriesRepository,
    real_augmentation: bool = False,
    timeseries_repository_augmentation: Optional[TimeSeriesRepository] = None,
) -> None:
    model.eval()
    timeseries_repository.reset()

    if timeseries_repository_augmentation:
        timeseries_repository_augmentation.reset()
    
    if real_augmentation:
        timeseries_repository.resize(size=3)
    
    contrastive_data = torch.tensor([]).to(DEVICE)
    temp_target = torch.tensor([]).to(DEVICE)

    for batch in dataloader:
        window, positive_pair, negative_pair, target = batch
        window = window.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        if window.ndim == 3:
            B, W, F = window.shape
        else:
            B, W = window.shape
            F = 1
        
        output = model.resnet.forward(x=window.reshape(B, F, W))
        timeseries_repository.update(features=output, targets=target)

        if timeseries_repository_augmentation:
            timeseries_repository_augmentation.update(
                features=output,
                targets=target
            )
        
        if real_augmentation:
            contrastive_data = torch.cat((contrastive_data, window), dim=0)
            temp_target = torch.cat((temp_target, target), dim=0)

            target = torch.LongTensor(
                [2]*positive_pair.shape[0]
            ).to(DEVICE, non_blocking=True)

            positive_pair = positive_pair.to(DEVICE, non_blocking=True)
            output = model.resnet.forward(positive_pair.reshape(B, F, W))
            timeseries_repository.update(features=output, targets=target)

            negative_pair = negative_pair.to(DEVICE, non_blocking=True)
            target = torch.LongTensor(
                [4]*negative_pair.shape[0]
            ).to(DEVICE, non_blocking=True)

            contrastive_data = torch.cat(
                tensors=(contrastive_data, negative_pair),
                dim=0,
            )
            temp_target = torch.cat(
                tensors=(temp_target, target),
                dim=0,
            )

            timeseries_repository.update(features=output, targets=target)
            timeseries_repository_augmentation.update(
                features=output,
                targets=target
            )
    
    if real_augmentation:
        contrastive_dataset = SaveAugmentedDataset(
            data=contrastive_data,
            target=temp_target
        )
        temp_loader = DataLoader(
            dataset=contrastive_dataset,
            batch_size=50,
            pin_memory=True,
            shuffle=False,
        )

        pretext_save_path = 'carla_loader/msl'
        if not os.path.exists(pretext_save_path):
            os.makedirs(pretext_save_path, exist_ok=True)

        torch.save(
            obj=temp_loader,
            f=os.path.join(pretext_save_path, 'contrastive_train_dataset.pth'),
        )
             
    return


def dataset(
    transform: NoiseTransformation,
    sub_anomaly: SubAnomaly,
    data: str = 'msl',
    mode: str = 'augment',
) -> Union[AugmentedDataset, NeighborsDataset]:
    if data == 'msl':
        basic_dataset = MSLDataset(
            train=True,
            transform=transform,
            subanomaly=sub_anomaly,
        )
        mean = basic_dataset.mean
        std = basic_dataset.std
    
    if mode == 'augment':
        dataset = AugmentedDataset(dataset=basic_dataset)
    elif mode == 'neighbor':
        nearest_indices = np.load()
        furthest_indices = np.load()
        dataset = NeighborsDataset(
            dataset=basic_dataset,
            nearset_indices=nearest_indices,
            furthest_indices=furthest_indices,
        )
    
    dataset.mean = mean
    dataset.std = std

    return dataset
