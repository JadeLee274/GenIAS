import pandas as pd
from utils.common_import import *
from utils.preprocess import *
from genias.tcnvae import VAE
DATA_PATH = '/data/seungmin/'
VAE_PATH = 'checkpoints/vae/'
CLASSIFICATION_PATH = 'temp/classification_temp'


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
        data_name: str,
        subdata: str,
        window_size: int = 200,
        normalize: str = 'mean_std',
        mode: str = 'train',
        convert_nan: str = 'nan_to_zero',
        anomaly_processing: str = 'drop',
        train_ratio: Optional[float] = None,
    ) -> None:
        data_path = None

        if data_name in ['MSL', 'SMAP', 'SMD']:
            data_path = os.path.join(DATA_PATH, data_name + '_SEPARATED', mode)
        else:
            data_path = os.path.join(DATA_PATH, data_name)

        data = None
        labels = None
        anomalies = None

        if data_name in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(
                    os.path.join(data_path, f'{subdata}_train.npy')
                )
            elif mode == 'test':
                data = np.load(
                    os.path.join(data_path, f'{subdata}_test.npy'))
                labels = np.load(
                    os.path.join(data_path, f'{subdata}_test_label.npy')
                )
                anomalies = data[labels == 1]

        elif data_name == 'SWaT':
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
        elif 'GECCO' in data_name.upper():
            data = pd.read_csv(
                os.path.join(
                    DATA_PATH,
                    'GECCO',
                    data_name,
                    f'1_{data_name.lower().replace('_', '')}_water_quality.csv'
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
                window = mean_std_normalize(
                    x=self.data[idx: idx + self.window_size]
                )

            window = np.float32(window)
            label = np.float32(label)
            
            return window, label


############################## Dataset for CARLA ##############################


class PretextDataset(object):
    def __init__(
        self,
        data_name: str,
        subdata: Optional[str] = None,
        window_size: int = 200,
        mode: str = 'train',
        use_genias: bool = False,
    ) -> None:
        self.data_name = data_name
        self.subdata = subdata
        self.window_size = window_size
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode
        self.use_genias = use_genias

        data_dir = os.path.join(
            DATA_PATH, data_name, subdata, mode, f'{subdata}_{mode}.npy'
        )
        self.data = np.load(data_dir)
        self.data_dim = self.data.shape[-1]

        self.mean, self.std = get_mean_std(x=self.data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.anchors = convert_to_windows(
            data=self.data,
            window_size=window_size
        )

        patch_coef = 0.3

        if data_name == 'MSL':
            patch_coef = 0.4
        elif data_name in ['SMAP', 'Yahoo']:
            patch_coef = 0.2

        self.get_pairs(patch_coef=patch_coef)
    
    def __len__(self) -> int:
        return self.anchors.shape[0]
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[Matrix, Matrix, Matrix],
        Tuple[Matrix, Matrix, Matrix, Vector]
    ]:
        if self.mode == 'train':
            anchor = self.anchors[idx]
            positive = self.positive_pairs[idx]
            negative = self.negative_pairs[idx]

            anchor = (anchor - self.mean) / self.std
            positive = (positive - self.mean) / self.std
            negative = (negative - self.mean) / self.std 

            return anchor, positive, negative
        else:
            anchor = self.anchors[idx]
            positive = self.positive_pairs[idx]
            negative = self.negative_pairs[idx]

            anchor = (anchor - self.mean) / self.std
            positive = (positive - self.mean) / self.std
            negative = (negative - self.mean) / self.std

            label = self.labels[idx]

            return anchor, positive, negative, label
    
    def get_pairs(self, patch_coef: float) -> None:
        if self.use_genias:
            vae = VAE(
                window_size=self.window_size,
                data_dim=self.data_dim,
                latent_dim=100,
                depth=10,
            )
            vae_dir = os.path.join(
                VAE_PATH, self.data_name, self.subdata, 'epoch_1000.pt')
            ckpt = torch.load(vae_dir)
            vae.load_state_dict(ckpt['model'])

            # Create positive pairs through TCN-VAE
            positive_pairs = np.empty_like(self.data)

            for i in range(self.data.shape[0]//self.window_size + 1):
                sub_positive = self.data[
                    i * self.window_size: (i + 1) * self.window_size
                ]

                if sub_positive.shape[0] != self.window_size:
                    sub_positive_temp = np.empty(
                        shape=(self.window_size, self.data_dim)
                    )
                    sub_positive_temp[:sub_positive.shape[0]] = sub_positive
                    sub_positive = sub_positive_temp
                
                sub_positive = torch.tensor(sub_positive, dtype=torch.float32)
                sub_positive = sub_positive.unsqueeze(0)
                negative_pair = vae.forward(sub_positive)[-2]
                positive_pair = positive_pair.squeeze(0)
                positive_pair = np.array(positive_pair.detach())

                if sub_positive.shape[0] != self.window_size:
                    positive_pair = positive_pair[:sub_positive.shape[0]]
                
                positive_pairs[
                    i * self.window_size: (i + 1) * self.window_size
                ] \
                = positive_pair
            
            self.positive_pairs = convert_to_windows(positive_pairs)

            # Create negative pairs through TCN-VAE
            negative_pairs = np.empty_like(self.data)

            for i in range(self.data.shape[0]//self.window_size + 1):
                sub_negative = self.data[
                    i * self.window_size: (i + 1) * self.window_size
                ]

                if sub_negative.shape[0] != self.window_size:
                    sub_negative_temp = np.empty(
                        shape=(self.window_size, self.data_dim)
                    )
                    sub_negative_temp[:sub_negative.shape[0]] = sub_negative
                    sub_negative = sub_negative_temp
                
                sub_negative = torch.tensor(sub_negative, dtype=torch.float32)
                sub_negative = sub_negative.unsqueeze(0)
                negative_pair = vae.forward(sub_negative)[-1]
                negative_pair = negative_pair.squeeze(0)
                negative_pair = np.array(negative_pair.detach())

                if sub_negative.shape[0] != self.window_size:
                    negative_pair = negative_pair[:sub_negative.shape[0]]
                
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

            negative_save_dir = os.path.join(
                CLASSIFICATION_PATH, self.data_name, self.subdata
            )
            os.makedirs(negative_save_dir, exist_ok=True)
            np.save(
                file=os.path.join(negative_save_dir, 'negative_pairs.npy'),
                arr=self.negative_pairs
            )
            print('Saving negetive pairs for classification stage.')

        else:
            positive_pairs = []

            for idx in range(self.anchors.shape[0]):
                if idx < 10:
                    positive_pair = self.anchors[idx]
                    positive_pair = noise_transformation(positive_pair)
                else:
                    random_idx = np.random.randint(idx - 10, idx)
                    positive_pair = self.anchors[random_idx]

                positive_pairs.append(positive_pair)
            
            self.positive_pairs = np.array(positive_pairs)

            negative_pairs = []
            anomaly_injection = AnomalyInjection()

            for idx in range(self.anchors.shape[0]):
                negative_pair = self.anchors[idx]
                negative_pair = anomaly_injection(negative_pair)
                negative_pairs.append(negative_pair)
            
            self.negative_pairs = np.array(negative_pairs)
            negative_save_dir = os.path.join(
                CLASSIFICATION_PATH, self.data_name, self.subdata
            )
            os.makedirs(negative_save_dir, exist_ok=True)
            np.save(
                file=os.path.join(negative_save_dir, 'negative_pairs.npy'),
                arr=self.negative_pairs
            )
            print('Saving negetive pairs for classification stage.')

        return


class ClassificationDataset(object):
    def __init__(
        self,
        data_type: str,
        data_name: str,
        mode: str = 'train',
    ) -> None:
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode

        data_dir = os.path.join(
            DATA_PATH, data_type, mode, f'{data_name}_{mode}.npy'
        )
        data = np.load(data_dir)
        self.mean, self.std = get_mean_std(x=data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.data_dim = data.shape[-1]

        if mode == 'test':
            label_dir = os.path.join(
                DATA_PATH, data_type, mode, f'{data_name}_{mode}_label.npy'
            )
            self.label = np.load(label_dir)
        
        if mode == 'train':
            classification_data_dir = os.path.join(
                CLASSIFICATION_PATH, data_type, data_name
            )
            anchors = convert_to_windows(data=data)
            negative_pairs = np.load(
                os.path.join(classification_data_dir, 'negative_pairs.npy')
            )
            self.windows = np.concatenate([anchors, negative_pairs], axis=0)

            anchor_nns = np.load(
                os.path.join(classification_data_dir, 'anchor_nns.npy')
            )
            negative_nns = np.load(
                os.path.join(classification_data_dir, 'negative_nns.npy')
            )
            self.nns = np.concatenate([anchor_nns, negative_nns], axis=0)

            anchor_fns = np.load(
                os.path.join(classification_data_dir, 'anchor_fns.npy')
            )
            negative_fns = np.load(
                os.path.join(classification_data_dir, 'negative_fns.npy')
            )
            self.fns = np.concatenate([anchor_fns, negative_fns], axis=0)

        else:
            self.data = (data - self.mean) / self.std

    def __len__(self) -> int:
        return self.windows.shape[0]
    
    def __getitem__(
        self,
        idx: int
    ) -> Optional[Tuple[Matrix, Array, Array]]:
        if self.mode == 'train':
            window = self.windows[idx]
            nearest_neighbor = self.nns[idx]
            furthest_neighbor = self.fns[idx]

            window = (window - self.mean) / self.std
            nearest_neighbor = (nearest_neighbor - self.mean) / self.std
            furthest_neighbor = (furthest_neighbor - self.mean) / self.std

            return window, nearest_neighbor, furthest_neighbor
        
        else:
            return self.data[idx]
