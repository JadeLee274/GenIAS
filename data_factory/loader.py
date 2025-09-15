import pandas as pd
from utils.common_import import *
from utils.preprocess import *
from genias.tcnvae import VAE
DATA_PATH = '/data/seungmin'


############################# Dataset for GenIAS #############################


class GenIASDataset(object):
    """
    Loads data for training VAE of GenIAS process.

    Parameters:
        dataset:            Name  of the dataset.
        window_size:        Length of the sliding window. Default 200.
    """
    def __init__(
        self,
        dataset: str,
        subdata: str,
        window_size: int = 200,
        normalize: str = 'mean_std',
    ) -> None:
        if dataset in ['MSL', 'SMAP', 'SMD']:
            data_path = f'{DATA_PATH}/{dataset}_SEPARATED/train/{subdata}.npy'
            data = np.load(data_path)
            self.data_dim = data.shape[-1]
        elif dataset == 'SWaT':
            data = pd.read_csv(os.path.join(data_path, 'SWaT_Normal.csv'))
            data.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
            data = data.values[:, 1:]
            self.data_dim = data.shape[-1]

        if normalize == 'mean_std':
            data = mean_std_normalize(data)
        elif normalize == 'min_max':
            data = min_max_normalize(data)

        self.windows = convert_to_windows(data=data, window_size=window_size)

    def __len__(self) -> int:
        return self.windows.shape[0]
    
    def __getitem__(self, idx: int) -> Union[Matrix, Tuple[Matrix, Matrix]]:
        return self.windows[idx]


############################## Dataset for CARLA ##############################


class PretextDataset(object):
    def __init__(
        self,
        dataset: str,
        subdata: Optional[str] = None,
        window_size: int = 200,
        use_genias: bool = False,
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        self.window_size = window_size
        self.use_genias = use_genias

        if dataset in ['MSL', 'SMAP', 'SMD']:
            data_dir = f'{DATA_PATH}/{dataset}_SEPARATED/train/{subdata}.npy'
            self.data = np.load(data_dir)
            self.data_dim = self.data.shape[-1]

        self.mean, self.std = get_mean_std(x=self.data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.anchors = convert_to_windows(
            data=self.data,
            window_size=window_size
        )

        patch_coef = 0.3

        if dataset == 'MSL':
            patch_coef = 0.4
        elif dataset in ['SMAP', 'Yahoo']:
            patch_coef = 0.2

        self.get_pairs(patch_coef=patch_coef)
    
    def __len__(self) -> int:
        return self.anchors.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Matrix, Matrix, Matrix]:
            anchor = self.anchors[idx]
            positive = self.positive_pairs[idx]
            negative = self.negative_pairs[idx]

            anchor = (anchor - self.mean) / self.std
            positive = (positive - self.mean) / self.std
            negative = (negative - self.mean) / self.std 

            return anchor, positive, negative
    
    def get_pairs(self, patch_coef: float) -> None:
        negative_dir = f'classification_dataset/{self.dataset}'   

        if self.use_genias:
            vae = VAE(
                window_size=self.window_size,
                data_dim=self.data_dim,
                latent_dim=100,
                depth=10,
            )
            vae_dir = f'checkpoints/vae/{self.dataset}/{self.subdata}'
            ckpt = torch.load(os.path.join(vae_dir, 'epoch_1000.pt'))
            vae.load_state_dict(ckpt['model'])

            positive_pairs = []
            negative_pairs = []
            
            for idx in range(self.anchors.shape[0]):
                anchor = self.anchors[idx]
                _, _, positive_pair, negative_pair = \
                vae.forward(torch.tensor(anchor).float().unsqueeze(0))
                positive_pair = positive_pair.detach().squeeze(0).numpy()
                negative_pair = negative_pair.detach().squeeze(0).numpy()
                negative_pair = patch(
                    x=anchor,
                    x_tilde=negative_pair,
                    tau=patch_coef,
                )
                positive_pairs.append(positive_pair)
                negative_pairs.append(negative_pair)
            
            self.positive_pairs = np.array(positive_pairs)
            self.negative_pairs = np.array(negative_pairs)

            # Saving negetive pairs for classification stage.
            negative_save_dir = os.path.join(
                negative_dir, self.subdata, 'use_genias',
            )
            os.makedirs(negative_save_dir, exist_ok=True)
            np.save(
                file=os.path.join(negative_save_dir, 'negative_pairs.npy'),
                arr=self.negative_pairs
            )

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
            
            # Saving negative pairs for classification stage.
            negative_save_dir = os.path.join(
                negative_dir, self.subdata, 'without_genias',
            )
            os.makedirs(negative_save_dir, exist_ok=True)
            np.save(
                file=os.path.join(negative_save_dir, 'negative_pairs.npy'),
                arr=self.negative_pairs
            )

        return


class ClassificationDataset(object):
    def __init__(
        self,
        dataset: str,
        subdata: str,
        window_size: int = 200,
        mode: str = 'train',
        use_genias: bool = False,
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode

        if dataset in ['MSL', 'SMAP', 'SMD']:
            data_dir = f'{DATA_PATH}/{dataset}_SEPARATED'
            if mode == 'train':
                data = np.load(f'{data_dir}/train/{subdata}.npy')
            elif mode == 'test':
                data = np.load(f'{data_dir}/test/{subdata}.npy')
                labels = np.load(f'{data_dir}/label/{subdata}.npy')
        
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
        self.mean, self.std = get_mean_std(x=data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)

        classification_dir = f'classification_dataset/{dataset}'
        if use_genias:
            classification_data_dir = os.path.join(
                classification_dir, subdata, 'use_genias',
            )
        else:
            classification_data_dir = os.path.join(
                classification_dir, subdata, 'without_genias',
            )
        
        if mode == 'train':
            anchors = convert_to_windows(data=data, window_size=window_size)
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

        elif mode == 'test':
            self.data = (data - self.mean) / self.std
            self.label = labels.reshape(-1)

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
