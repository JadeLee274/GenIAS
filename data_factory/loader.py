import pandas as pd
from torch.utils.data import Dataset
from genias.tcnvae import VAE
from utils.patch import patch
from utils.common_import import *
from utils.alias import *
DATA_PATH = '/data/seungmin'
VAE_PATH = '/data/home/tmdals274/genias/checkpoints/vae'
"""
Codes for loading data. This code follows the paper
GenIAS: Generator for Instantiating Anomalies in Time Series,
Darban et al., 2025

Paper link: https://arxiv.org/pdf/2502.08262
"""


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


def min_max_normalize(x: Matrix) -> Matrix:
    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)
    return (x - min_x) / (max_x - min_x + 1e-4)


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
        self.mode = mode
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx: int) -> Matrix:
        if self.mode == 'train':
            return min_max_normalize(
                np.float32(self.data[idx: idx + self.window_size])
            )
        elif self.mode == 'test':
            return np.float32(
                min_max_normalize(self.data[idx: idx + self.window_size])
            ), \
            np.float32(self.labels[idx: idx + self.window_size])


class CARLADataset(Dataset):
    def __init__(
        self,
        dataset: str,
        window_size: int = 200,
        mode: str = 'train',
        # convert_nan: str = 'drop',
        # anomaly_processing: str = 'drop',
        # train_ratio: Optional[float] = None,
    ) -> None:
        """
        Training set for CARLA pretext training.

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
        data_path = os.path.join(DATA_PATH, dataset)

        data = None
        
        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(os.path.join(data_path, f'{dataset}_train.npy'))
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(data_path, 'SWaT_Normal.csv'))
                data.drop(
                    columns=[' Timestamp', 'Normal/Attack'],
                    inplace=True,
                )
                data = data.values[:, 1:]
        
        data = torch.tensor(data, dtype=torch.float32)
        self.data = data

        self.data_len = data.shape[0]
        self.get_positive_pairs()
        
        self.window_size = window_size
        self.data_dim = data.shape[1]
        self.latent_dim = 100 if self.data_dim != 1 else 50
        self.dataset = dataset
        
        if dataset == 'MSL':
            patch_coef = 0.4
        elif dataset in ['SMAP', 'Yahoo']:
            patch_coef = 0.2
        
        self.get_negative_pairs(patch_coef=patch_coef)

    
    def __len__(self) -> int:
        return self.data.shape[0] - self.window_size + 1
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        anchor = self.data[idx: idx + self.window_size]
        positive_pair = self.positive_pairs[idx: idx + self.window_size]
        negative_pair = self.negative_pairs[idx: idx + self.window_size]
        return anchor, positive_pair, negative_pair

    def get_positive_pairs(self) -> None:
        positive_pairs = torch.empty_like(self.data)

        for i in range(self.data_len):
            if i == 0:
                random_index = i
            else:
                random_index = np.random.randint(0, i)
            positive_pairs[i] = self.data[random_index]
        self.positive_pairs = positive_pairs

        return None

    def get_negative_pairs(self, patch_coef: float) -> None:
        vae = VAE(
            window_size=self.window_size,
            data_dim=self.data_dim,
            latent_dim=self.latent_dim,
            depth=10,
        )

        ckpt = torch.load(
            os.path.join(VAE_PATH, self.dataset, 'epoch_1000.pt')
        )
        vae.load_state_dict(ckpt['model'])

        negative_pairs = torch.empty_like(self.data)
        data_len = self.data.shape[0]

        for i in range(data_len//self.window_size + 1):
            subdata = self.data[
                i * self.window_size: (i + 1) * self.window_size
            ]
            
            if subdata.shape[0] != self.window_size:
                subdata_temp = torch.empty(self.window_size, self.data_dim)
                subdata_temp[:subdata.shape[0]] = subdata
                subdata = subdata_temp

            subdata = subdata.unsqueeze(0)
            negative_pair = vae.forward(subdata)[-1]
            negative_pair = negative_pair.squeeze(0)

            if subdata.shape[0] != self.window_size:
                negative_pair = negative_pair[:subdata.shape[0]]

            negative_pairs[
                i * self.window_size: (i + 1) * self.window_size,
            ] \
            = negative_pair
        
        negative_pair = patch(
        x=self.data,
        x_tilde=negative_pairs,
        tau=patch_coef
        )
        self.negative_pairs = negative_pairs

        return None
