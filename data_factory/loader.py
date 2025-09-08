import pandas as pd
from torch.utils.data import Dataset
from genias.tcnvae import VAE
from utils.common_import import *
from utils.preprocess import *
DATA_PATH = '/data/seungmin'
VAE_PATH = '../checkpoints/vae'
RESNET_PATH = '../checkpoints/carla_pretext'
"""
Codes for loading data. This code follows the paper
GenIAS: Generator for Instantiating Anomalies in Time Series,
Darban et al., 2025

Paper link: https://arxiv.org/pdf/2502.08262
"""

############################# Dataset for GenIAS #############################

class GenIASDataset(object):
    """
    Loads data for training VAE of GenIAS process.

    Parameters:
        dataset:            Name  of the dataset.
        window_size:        Length of the sliding window. Default 200.
        mode:               Either train or test. Default train.

        convert_nan:        How to convert the data with NaN value. 
                            Default 'nan_to_zero.'
                            If dataset is 'GECCO_2018' or 'CECCO_2019', 
                            then set it to 'overwrite.'

        anomaly_processing: How to process anomalies of the training dataset. 
                            Default 'drop.'

        train_traio:        The ratio of train set. 
                            For MSL, SMAP, SMD, SWaT, it is unnecessary.
                            For GECCO_2018 and GECCO_2019, set it to 0.5.
    """
    def __init__(
        self,
        dataset: str,
        window_size: int = 200,
        normalize: str = 'min_max',
        mode: str = 'train',
        convert_nan: str = 'nan_to_zero',
        anomaly_processing: str = 'drop',
        train_ratio: Optional[float] = None,
    ) -> None:
        data_path = os.path.join(DATA_PATH, dataset)

        data = None
        labels = None
        anomalies = None

        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(os.path.join(data_path, f'{dataset}_train.npy'))
            elif mode == 'test':
                data = np.load(os.path.join(data_path, f'{dataset}_test.npy'))
                labels = np.load(
                    os.path.join(data_path, f'{dataset}_test_label.npy')
                )
                anomalies = data[labels == 1]
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(data_path, 'SWaT_Normal.csv'))
                data.drop(
                    columns=[' Timestamp', 'Normal/Attack'],
                    inplace=True,
                )
                data = data.values[:, 1:]
            elif mode == 'test':
                data = pd.read_csv(os.path.join(data_path, 'SWaT_Abormal.csv'))
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                anomalies = data[labels == 1]
                anomalies = anomalies[:, :-1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)
        elif 'GECCO' in dataset.upper():
            data = pd.read_csv(
                os.path.join(
                    DATA_PATH,
                    'GECCO',
                    dataset,
                    f'1_{dataset.lower().replace('_', '')}_water_quality.csv'
                )
            )
            data.drop(columns=['Time'], inplace=True)
            data = data.values[:, 1:]
            data = np.array(data, dtype=np.float64)

            if anomaly_processing == 'drop':
                data = drop_anomaly(data)
            elif anomaly_processing == 'overwrite':
                data = overwrite_anomaly(data)
            
            labels = data[:, -1]
            labels = np.where(labels == False, 0, 1)
            data = data[:, :-1]

            train_size = int(data.shape[0] * train_ratio)
            if mode == 'train':
                data = data[:train_size, :]
            elif mode == 'test':
                data = data[train_size:, :]
                labels = labels[train_size:]
        
        if convert_nan == 'overwrite':
            data = overwrite_nan(data)
        elif convert_nan == 'nan_to_zero':
            data = np.nan_to_num(data)

        self.data_shape = data.shape

        self.data = data
        self.labels = labels
        self.anomalies = anomalies

        assert normalize in ['min_max', 'mean_std', 'none'], \
        "'normalize' argument must be either 'min_max' or 'mean_std'."

        self.normalize = normalize
        self.mode = mode
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx: int) -> Union[Matrix, Tuple[Matrix, Matrix]]:
        if self.mode == 'train':
            window = self.data[idx: idx + self.window_size]
            
            if self.normalize == 'min_max':
                window = min_max_normalize(window)
            elif self.normalize == 'mean_std':
                window =  mean_std_normalize(window)

            window = np.float32(window)
            return window
            
        elif self.mode == 'test':
            window = self.data[idx: idx + self.window_size]
            label = self.labels[idx: idx + self.window_size]

            if self.normalize == 'min_max':
                window = min_max_normalize(window)
            elif self.normalize == 'mean_std':
                window = mean_std_normalize(self.data[idx: idx + self.window_size])

            window = np.float32(window)
            label = np.float32(label)
            
            return window, label

############################## Dataset for CARLA ##############################

class PretextDataset(object):
    """
    Dataset for CARLA pretext stage.
    This is the modified version of 'AugmentDatset' of CARLA code.
    It loads anchor, positive pair, negative pair when mode is 'train'; 
    and label, in addition, when mode is 'test'.

    Parameters:
        dataset:     Name of dataset.
        window_size: Window size of data, positive pair, negative pair.
                     Default 200.
        mode:        Whether the dataset is for trainig or test. 
                     Default 'train'. It must be either 'train' or 'test'.
        use_genias:  Whether or not to use GanIAS to create negative pairs.
                     Default True.
    """
    def __init__(
        self,
        dataset: str,
        window_size: int = 200,
        mode: str = 'train',
        use_genias: bool = False,
    ) -> None:
        assert mode in ['train', 'test'], \
        "mode must be either 'train' or 'test'"

        self.dataset = dataset
        self.window_size = window_size
        self.mode = mode
        self.use_genias = use_genias
        self.anomaly_injection = AnomalyInjection()

        data = None
        labels = None
        
        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(
                    os.path.join(DATA_PATH, dataset, f'{dataset}_train.npy'))
            elif mode == 'test':
                data = np.load(
                    os.path.join(DATA_PATH, dataset, f'{dataset}_test.npy')
                )
                labels = np.load(
                    os.path.join(
                        DATA_PATH, dataset, f'{dataset}_test_label.npy'
                    )
                )
                anomalies = data[labels == 1]
        
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(DATA_PATH, 'SWaT_Normal.csv'))
                data.drop(
                    columns=[' Timestamp', 'Normal/Attack'],
                    inplace=True,
                )
                data = data.values[:, 1:]
            elif mode == 'test':
                data = pd.read_csv(
                    os.path.join(DATA_PATH, 'SWaT', 'SWaT_Abormal.csv')
                )
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                anomalies = data[labels == 1]
                anomalies = anomalies[:, :-1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)
        
        self.data = data
        mean, std = get_mean_std(data)
        self.mean = mean
        self.std = std

        self.anchors = convert_to_windows(data=data, window_size=window_size)
        self.data_dim = self.anchors.shape[-1]

        if labels:
            self.labels = convert_to_windows(
                data=labels,
                window_size=window_size
            )
        
        patch_coef = 0.3

        if dataset == 'MSL':
            patch_coef = 0.4
        elif dataset in ['SMAP', 'Yahoo']:
            patch_coef = 0.2
        
        self.get_positive_pairs()
        self.get_negative_pairs(patch_coef=patch_coef)
   
    def get_positive_pairs(self) -> None:
        positive_pairs = []

        for idx in range(self.anchors.shape[0]):
            if idx > 10:
                positive_pair_idx = np.random.randint(idx - 10, idx)
                positive_pair = self.anchors[positive_pair_idx]
            else:
                positive_pair = self.anchors[idx]
                positive_pair = noise_transformation(x=positive_pair)
            
            positive_pairs.append(positive_pair)
        
        self.positive_pairs = np.array(positive_pairs)
        
        return

    def get_negative_pairs(self, patch_coef: float) -> None:
        if self.use_genias:
            vae = VAE(
                window_size=self.window_size,
                data_dim=self.data_dim,
                latent_dim = 100 if self.data_dim != 0 else 50,
                depth=10,
            )

            ckpt = torch.load(
                os.path.join(VAE_PATH, self.dataset, 'epoch_1000.pt')
            )

            vae.load_state_dict(ckpt['model'])

            negative_pairs = np.empty_like(self.data)

            for i in range(self.data.shape[0]//self.window_size + 1):
                subdata = self.data[
                    i * self.window_size: (i + 1) * self.window_size
                ]

                if subdata.shape[0] != self.window_size:
                    subdata_temp = np.empty(
                        shape=(self.window_size, self.data_dim)
                    )
                    subdata_temp[:subdata.shape[0]] = subdata
                    subdata = subdata_temp
                
                subdata = torch.tensor(subdata, dtype=torch.float32)
                subdata = subdata.unsqueeze(0)
                negative_pair = vae.forward(subdata)[-1]
                negative_pair = negative_pair.squeeze(0)
                negative_pair = np.array(negative_pair.detach())

                if subdata.shape[0] != self.window_size:
                    negative_pair = negative_pair[:subdata.shape[0]]
                
                negative_pairs[
                    i * self.window_size: (i + 1) * self.window_size
                ] \
                = negative_pair

            negative_pairs = patch(
                x=self.data,
                x_tilde=negative_pairs,
                tau=patch_coef
            )

            self.negative_pairs = convert_to_windows(negative_pairs)
        
        else:
            negative_pairs = []
            
            for idx in range(self.anchors.shape[0]):
                window = self.anchors[idx]
                negative_pairs.append(
                    self.anomaly_injection(window)
                )
            
            self.negative_pairs = np.array(negative_pairs)

        return

    def __len__(self) -> int:
        return self.anchors.shape[0]
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[
        Tuple[Matrix, Matrix, Matrix],
        Tuple[Matrix, Matrix, Matrix, Matrix],
    ]:
        mean = self.mean
        std = self.std
        std = np.where(std==0.0, 1.0, std)

        if self.mode == 'train':
            anchor = self.anchors[idx]
            anchor = (anchor - mean) / std

            positive_pair = self.positive_pairs[idx]
            positive_pair = (positive_pair - mean) / std

            negative_pair = self.negative_pairs[idx]
            negative_pair = (negative_pair - mean) / std

            return anchor, positive_pair, negative_pair
        
        elif self.mode == 'test':
            anchor = self.anchors[idx]
            anchor = (anchor - mean) / std

            positive_pair = self.positive_pairs[idx]
            positive_pair = (positive_pair - mean) / std

            negative_pair = self.negative_pairs[idx]
            negative_pair = (negative_pair - mean) / std

            label = self.labels[idx]

            return anchor, positive_pair, negative_pair, label        


class ClassificationDataset(object):
    def __init__(
        self,
        dataset: str,
        window_size: int = 200,
        mode: str = 'train',
    ) -> None:
        assert mode in ['train', 'test'], \
        "mode must be either 'train' or 'test'"

        data = None
        labels = None
        
        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(
                    os.path.join(DATA_PATH, dataset, f'{dataset}_train.npy'))
            elif mode == 'test':
                data = np.load(
                    os.path.join(DATA_PATH, dataset, f'{dataset}_test.npy')
                )
                labels = np.load(
                    os.path.join(
                        DATA_PATH, dataset, f'{dataset}_test_label.npy'
                    )
                )
                anomalies = data[labels == 1]
        
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(DATA_PATH, 'SWaT_Normal.csv'))
                data.drop(
                    columns=[' Timestamp', 'Normal/Attack'],
                    inplace=True,
                )
                data = data.values[:, 1:]
            elif mode == 'test':
                data = pd.read_csv(
                    os.path.join(DATA_PATH, 'SWaT', 'SWaT_Abormal.csv')
                )
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                anomalies = data[labels == 1]
                anomalies = anomalies[:, :-1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)

        self.data_dim = data.shape[-1]
        mean, std = get_mean_std(data)
        std = np.where(std==0.0, 1.0, std)
        self.mean = mean
        self.std = std

        self.anchors = convert_to_windows(data=data, window_size=window_size)
        
        neighbors_dir = f'../classification_dataset/{dataset}'

        self.nearest_neighbors = np.load(
            os.path.join(neighbors_dir, 'nearest_neighbors.npy')
        )
        self.furthest_neighbors = np.load(
            os.path.join(neighbors_dir, 'furthest_neighbors.npy')
        )

        if labels:
            self.labels = convert_to_windows(
                data=labels,
                window_size=window_size
            )

        self.mode = mode

        return

    def __len__(self) -> int:
        return self.anchors.shape[0]
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[
        Tuple[NPTensor, NPTensor, NPTensor],
        Tuple[NPTensor, NPTensor, NPTensor, NPTensor],
    ]:
        if self.mode == 'train':
            anchor = self.anchors[idx]
            nearest_neighbor = self.nearest_neighbors[idx]
            furthest_neighbor = self.furthest_neighbors[idx]

            anchor = (anchor - self.mean) / self.std
            nearest_neighbor = (nearest_neighbor - self.mean) / self.std
            furthest_neighbor = (furthest_neighbor - self.mean) / self.std

            return anchor, nearest_neighbor, furthest_neighbor
        
        elif self.mode == 'test':
            anchor = self.anchors[idx]
            nearest_neighbor = self.nearest_neighbors[idx]
            furthest_neighbor = self.furthest_neighbors[idx]

            anchor = (anchor - self.mean) / self.std
            nearest_neighbor = (nearest_neighbor - self.mean) / self.std
            furthest_neighbor = (furthest_neighbor - self.mean) / self.std

            label = self.labels[idx]

            return anchor, nearest_neighbor, furthest_neighbor, label
