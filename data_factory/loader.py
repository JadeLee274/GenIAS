import pandas as pd
from utils.common_import import *
from utils.preprocess import *
from genias.tcnvae import VAE


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
        subdata: Optional[str] = None,
        window_size: int = 200,
        normalize: str = 'mean_std',
    ) -> None:
        if dataset in ['MSL', 'SMAP', 'SMD']:
            data_path = f'data/{dataset}/train/{subdata}.npy'
            data = np.load(data_path)
            self.data_dim = data.shape[-1]
        elif dataset == 'SWaT':
            data = pd.read_csv(f'data/{dataset}/SWaT_Normal.csv')
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
        scheme: str = 'carla',
        mix_step: Optional[int] = None,
        num_pairs: int = 3,
        cut_negative_pairs: bool = True,
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        self.window_size = window_size

        assert scheme in ['carla', 'genias', 'mix', 'genias_multiple'], \
        "'carla', 'genias', 'mix', 'genias_multiple'"

        self.scheme = scheme
        self.mix_step = mix_step
        self.num_pairs = num_pairs
        self.cut_negative_pairs = cut_negative_pairs

        if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            data_dir = f'data/{dataset}/train/{subdata}.npy'
            self.data = np.load(data_dir)
            self.data_dim = self.data.shape[-1]

        self.mean, self.std = get_mean_std(x=self.data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.anchors = convert_to_windows(
            data=self.data,
            window_size=window_size
        )
        self.len = self.anchors.shape[0]

        # Get pairs
        self._get_positive_pairs()
        self._get_negative_pairs()
    
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

    def _get_positive_pairs(self) -> None:
        positive_pairs = []

        for idx in range(self.anchors.shape[0]):
            if self.scheme == 'genias_multiple':
                if idx < 10:
                    positive_pair = self.anchors[idx]
                    positive_pair = [
                        noise_transformation(positive_pair) \
                        for _ in range(self.num_pairs)
                    ]
                    positive_pair = np.array(positive_pair)
                else:
                    random_idx = np.array([i for i in range(idx - 10, idx)])
                    random_idx = np.random.choice(random_idx, self.num_pairs, replace=False)
                    positive_pair = self.anchors[random_idx]
                
                positive_pairs.append(positive_pair)

            else:
                if idx < 10:
                    positive_pair = self.anchors[idx]
                    positive_pair = noise_transformation(positive_pair)
                else:
                    random_idx = np.random.randint(idx - 10, idx)
                    positive_pair = self.anchors[random_idx]

                positive_pairs.append(positive_pair)
        
        self.positive_pairs = np.array(positive_pairs)

        return
    
    def _get_negative_pairs(self) -> None:        
        # Negative pair generation algorithm for CARLA
        anomaly_injection = AnomalyInjection()

        vae = VAE(
            window_size=self.window_size,
            data_dim=self.data_dim,
            latent_dim=100,
            depth=10,
        )
        vae_dir = f'checkpoints/vae/{self.dataset}/{self.subdata}'
        vae_ckpt = torch.load(f'{vae_dir}/epoch_1000.pt')
        vae.load_state_dict(vae_ckpt['model'])
        vae.eval()

        patch_coef = 0.05 # other options include 0.1 and 0.6

        if self.dataset == 'MSL':
            patch_coef = 0.4
        elif self.dataset in ['SMAP', 'Yahoo']:
            patch_coef = 0.2
        
        # Prepare negative pairs
        negative_pairs = []
        for idx in range(self.anchors.shape[0]):
            anchor = self.anchors[idx]

            if self.scheme == 'carla':
                negative_pair = anomaly_injection(anchor)
            
            elif self.scheme == 'genias':
                _, _, _, negative_pair = vae.forward(
                    torch.tensor(anchor).float().unsqueeze(0)
                )
                negative_pair = negative_pair.detach().squeeze(0).numpy()
                negative_pair = patch(
                    x=anchor,
                    x_tilde=negative_pair,
                    tau=patch_coef,
                )
            
            elif self.scheme == 'mix':
                idx_mod_step = idx // self.mix_step
                idx_mode = idx_mod_step // 2
                
                if idx_mode == 0: # CARLA scheme applied
                    negative_pair = anomaly_injection(anchor)
                    
                elif idx_mode == 1: # GenIAS scheme applied
                    _, _, _, negative_pair = vae.forward(
                        torch.tensor(anchor).float().unsqueeze(0)
                    )
                    negative_pair = negative_pair.squeeze(0).detach().numpy()
                    negative_pair = patch(
                        x=anchor,
                        x_tilde=negative_pair,
                        tau=patch_coef,
                    )
                        
            elif self.scheme == 'genias_multiple':
                negative_pair = []

                for _ in range(self.num_pairs):
                    _, _, _, negative = vae.forward(
                        torch.tensor(anchor).float().unsqueeze(0)
                    )
                    negative = negative.detach().squeeze(0).numpy()
                    negative = patch(
                        x=anchor,
                        x_tilde=negative,
                        tau=patch_coef,
                    )
                    negative_pair.append(negative)

                negative_pair = np.array(negative_pair)

            negative_pairs.append(negative_pair)
                
        self.negative_pairs = np.array(negative_pairs)

        # Saving negetive pairs for classification stage.
        negative_save_dir = f'classification_dataset/{self.dataset}'

        if self.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            negative_save_dir = f'{negative_save_dir}/{self.subdata}'

        negative_save_dir = f'{negative_save_dir}/{self.scheme}'
        os.makedirs(negative_save_dir, exist_ok=True)

        if self.cut_negative_pairs:
            random_choose = np.random.randint(0, self.negative_pairs.shape[1])
            self._negative_pairs = self.negative_pairs[:, random_choose]
        else:
            self._negative_pairs = self.negative_pairs

        np.save(
            file=f'{negative_save_dir}/negative_pairs.npy',
            arr=self._negative_pairs,
        )
    
    def _cut_negative_pairs(self) -> None:
        self.negative_pairs = self._negative_pairs

        return


class ClassificationDataset(object):
    def __init__(
        self,
        dataset: str,
        subdata: Optional[str] = None,
        window_size: int = 200,
        mode: str = 'train',
        scheme: str = 'carla',
        num_pairs: int = 3,
        cut_negative_pairs: bool = True,
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata

        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode

        assert scheme in ['carla', 'genias', 'mix', 'genias_multiple'], \
        "'carla', 'genias', 'mix', 'genias_multiple'"

        if dataset in ['MSL', 'SMAP', 'SMD']:
            data_dir = f'data/{dataset}'
            if mode == 'train':
                data = np.load(f'{data_dir}/train/{subdata}.npy')
            elif mode == 'test':
                data = np.load(f'{data_dir}/test/{subdata}.npy')
                labels = np.load(f'{data_dir}/label/{subdata}.npy')
        
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(f'data/{dataset}/SWaT_Normal.csv')
                data.drop(
                    columns=[' Timestamp', 'Normal/Attack'],
                    inplace=True,
                )
                data = data.values[:, 1:]
            elif mode == 'test':
                data = pd.read_csv(f'data/{dataset}/SWaT_Abormal.csv')
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)
        
        self.data_dim = data.shape[-1]
        self.mean, self.std = get_mean_std(x=data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)

        classification_dir = f'classification_dataset/{dataset}'

        if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            classification_dir = f'{classification_dir}/{subdata}'
        
        classification_dir = f'{classification_dir}/{scheme}'
        
        if mode == 'train':
            anchors = convert_to_windows(data=data, window_size=window_size)
            negative_pairs = np.load(
                f'{classification_dir}/negative_pairs.npy'
            )
            if scheme == 'genias_multiple':
                windows = []
                for idx in range(anchors.shape[0]):
                    windows.append(anchors[idx])
                
                for idx in range(negative_pairs.shape[0]):
                    if cut_negative_pairs:
                        windows.append(negative_pairs[idx])
                    else:
                        for idx in range(negative_pairs.shape[0]):
                            nega_1, nega_2, nega_3 = np.split(negative_pairs[idx], 3)
                            windows.append(nega_1.squeeze(0))
                            windows.append(nega_2.squeeze(0))
                            windows.append(nega_3.squeeze(0))
                        
                self.windows = np.array(windows)
                print(self.windows.shape)
            
            else:
                self.windows = np.concatenate([anchors, negative_pairs], axis=0)

            anchor_nn_indices = np.load(
                f'{classification_dir}/anchor_nn_indices.npy'
            )
            negative_nn_indices = np.load(
                f'{classification_dir}/negative_nn_indices.npy'
            )
            anchor_nns = []
            
            for idx in range(anchors.shape[0]):
                anchor_nn_idx = anchor_nn_indices[idx]
                anchor_nn_idx = np.random.choice(anchor_nn_idx) # newley added for experimental purpose
                anchor_nns.append(self.windows[anchor_nn_idx])
            
            anchor_nns = np.array(anchor_nns)

            negative_nns = []
            
            if scheme == 'genias_multiple':
                if cut_negative_pairs:
                    for idx in range(negative_pairs.shape[0]):
                        negative_nn_idx = negative_nn_indices[idx]
                        negative_nn_idx = np.random.choice(negative_nn_idx)
                        negative_nns.append(self.windows[negative_nn_idx])
                else:
                    for idx in range(negative_pairs.shape[0] * num_pairs):
                        negative_nn_idx = negative_nn_indices[idx]
                        negative_nn_idx = np.random.choice(negative_nn_idx)
                        negative_nns.append(self.windows[negative_nn_idx])
            else:
                for idx in range(negative_pairs.shape[0]):
                    negative_nn_idx = negative_nn_indices[idx]
                    negative_nn_idx = np.random.choice(negative_nn_idx) # newley added for experimental purpose
                    negative_nns.append(self.windows[negative_nn_idx])

            negative_nns = np.array(negative_nns)

            self.nns = np.concatenate([anchor_nns, negative_nns], axis=0)
            
            anchor_fn_indices = np.load(
                f'{classification_dir}/anchor_fn_indices.npy'
            )
            negative_fn_indices = np.load(
                f'{classification_dir}/negative_fn_indices.npy'
            )

            anchor_fns = []

            for idx in range(anchors.shape[0]):
                anchor_fn_idx = anchor_fn_indices[idx]
                anchor_fn_idx = np.random.choice(anchor_fn_idx) # newley added for experimental purpose
                anchor_fns.append(self.windows[anchor_fn_idx])

            anchor_fns = np.array(anchor_fns)

            negative_fns = []
            
            if scheme == 'genias_multiple':
                if cut_negative_pairs:
                    for idx in range(negative_pairs.shape[0]):
                        negative_fn_idx = negative_fn_indices[idx]
                        negative_fn_idx = np.random.choice(negative_fn_idx)
                        negative_fns.append(self.windows[negative_fn_idx])
                else:
                    for idx in range(negative_pairs.shape[0] * num_pairs):
                        negative_fn_idx = negative_fn_indices[idx]
                        negative_fn_idx = np.random.choice(negative_fn_idx)
                        negative_fns.append(self.windows[negative_fn_idx])
            else:
                for idx in range(negative_pairs.shape[0]):
                    negative_fn_idx = negative_fn_indices[idx]
                    negative_fn_idx = np.random.choice(negative_fn_idx) # newley added for experimental purpose
                    negative_fns.append(self.windows[negative_fn_idx])
                    
            negative_fns = np.array(negative_fns)

            self.fns = np.concatenate([anchor_fns, negative_fns], axis=0)

        elif mode == 'test':
            self.windows = convert_to_windows(
                data=data, window_size=window_size
            )
            labels = convert_to_windows(data=labels, window_size=window_size)
            window_labels = []
            for label in labels:
                if np.sum(label) > 0:
                    window_labels.append(1)
                else:
                    window_labels.append(0)
            self.labels = np.array(window_labels).reshape(-1)

        return

    def __len__(self) -> int:
        return self.windows.shape[0]
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[Tuple[Matrix, Array, Array], Matrix]:
        if self.mode == 'train':
            window = self.windows[idx]
            nearest_neighbor = self.nns[idx]
            furthest_neighbor = self.fns[idx]

            window = (window - self.mean) / self.std
            nearest_neighbor = (nearest_neighbor - self.mean) / self.std
            furthest_neighbor = (furthest_neighbor - self.mean) / self.std

            return window, nearest_neighbor, furthest_neighbor
        
        else:
            return self.windows[idx]
