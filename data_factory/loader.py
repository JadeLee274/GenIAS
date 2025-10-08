import pandas as pd
from utils.common_import import *
from utils.preprocess import *
from genias.tcnvae import VAE


############################# Dataset for GenIAS #############################


class GenIASDataset(object):
    """
    Loads data for training VAE of GenIAS process.

    Parameters:
        dataset:            Name of the dataset.
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
            data_path = os.path.join('data',dataset,'train',f'{subdata}.npy')
            data = np.load(data_path)
            self.data_dim = data.shape[-1]

        elif dataset == 'SWaT':
            data_path = os.path.join('data', dataset, 'SWaT_Normal.csv')
            data = pd.read_csv(data_path)
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
    """
    Dataset for pretext stage of CARLA.

    Parameters:
        dataset:     Name of the dataset.
        subdata:     If dataset consists of more than one subdata, the name
                     of the subdata.
        window_size: Window size. Default 200.
        use_genias:  Whether or not to use GenIAS scheme for making positive 
                     and negative pairs. Default False.
        device:      CUDA or CPU device where the genias VAE operates.
    """
    def __init__(
        self,
        dataset: str,
        subdata: Optional[str] = None,
        window_size: int = 200,
        scheme: str = 'carla',
        device: str = 'cpu',
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        self.window_size = window_size
        self.scheme = scheme
        self.device = device

        data_dir = os.path.join('data', dataset)

        if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            data_dir = os.path.join(data_dir, 'train', f'{subdata}.npy')
            self.data = np.load(data_dir)
            self.data_dim = self.data.shape[-1]

        elif dataset == 'SWaT':
            data_dir = os.path.join(data_dir, 'SWaT_Normal.csv')
            data = pd.read_csv(data_dir)
            data.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
            self.data = data.values[:, 1:]
            self.data_dim = data.shape[-1]

        self.mean, self.std = get_mean_std(x=self.data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.anchors = convert_to_windows(
            data=self.data,
            window_size=window_size
        )

        # Get positive pairs
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

        # Get negative pairs
        self._get_negative_pairs()

        return
    
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
    
    def _get_negative_pairs(self) -> None:
        negative_dir = os.path.join('classification_dataset', self.dataset)
        vae_dir = os.path.join('checkpoints', 'vae', self.dataset)
        
        if self.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            negative_dir = os.path.join(negative_dir, self.subdata)
            vae_dir = os.path.join(vae_dir,  self.subdata)
        
        negative_dir = os.path.join(negative_dir, self.scheme)
        os.makedirs(negative_dir, exist_ok=True)

        negative_pairs = []

        if self.scheme == 'carla':
            anomaly_injection = AnomalyInjection()
            for anchor in self.anchors:
                negative_pair = anomaly_injection(anchor)
                negative_pairs.append(negative_pair)

        elif self.scheme == 'genias':
            vae = VAE(
                window_size=self.window_size,
                data_dim=self.data_dim,
                latent_dim=100,
                depth=10,
            ).to(self.device)
            vae_ckpt = torch.load(
                os.path.join(vae_dir, 'epoch_1000.pt'),
                map_location=self.device
                )
            vae.load_state_dict(vae_ckpt['model'])
            vae.eval()

            if self.dataset == 'MSL':
                patch_coef = 0.4
            elif self.dataset in ['SMAP', 'Yahoo']:
                patch_coef = 0.2
            else:
                patch_coef = np.random.choice([0.05, 0.1, 0.2, 0.4, 0.6])

            negative_pairs = []
            
            for anchor in self.anchors:
                anchor = torch.tensor(anchor).float().to(self.device)
                negative_pair = vae.forward(anchor.unsqueeze(0))[-1]
                negative_pair = negative_pair.detach().cpu().squeeze(0).numpy()
                negative_pair = patch(
                    x=anchor.detach().cpu().numpy(),
                    x_tilde=negative_pair,
                    tau=patch_coef,
                )
                negative_pairs.append(negative_pair)
        self.negative_pairs = np.array(negative_pairs)
        np.save(
            file=os.path.join(negative_dir, 'negative_pairs.npy'),
            arr=self.negative_pairs
        )

        return


class ClassificationDataset(object):
    """
    Dataset for Classification stage of CARLA.

    Parameters:
        dataset:     Name of the dataset.
        subdata:     If the dataset consists of more than one subdata, the
                     name of subdata.
        window_size: Window size of anchor. Default 200.
        mode:        If 'train', anchors, negative pairs, nearest/furthest
                     neighborhoods are loaded for training. If 'test', only
                     the raw test dataset (with window conversion not applied) 
                     and the test label are loaded.
        use_genias:  Whether or not the pretext stage where the negative pairs,
                     nearest/furthest neighborhoods at the pretext stage was
                     generated by GenIAS scheme or not. Default False.
    """
    def __init__(
        self,
        dataset: str,
        subdata: str,
        window_size: int = 200,
        mode: str = 'train',
        scheme: str = 'carla',
        neighborhood_choice: str = 'random_choice',
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode
        
        data_dir = os.path.join('data', dataset)

        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(
                    os.path.join(data_dir, 'train', f'{subdata}.npy')
                )
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
            elif mode == 'test':
                data = np.load(
                    os.path.join(data_dir, 'test', f'{subdata}.npy')
                )
                self.mean, self.std = get_mean_std(x=np.load(
                    os.path.join(data_dir, 'train', f'{subdata}.npy')
                ))
                self.std = np.where(self.std == 0.0, 1.0, self.std)
                labels = np.load(
                    os.path.join(data_dir, 'label', f'{subdata}.npy')    
                )
        
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Normal.csv'))
                data.drop(columns=[' Timestamp', 'Normal/Attack'],inplace=True)
                data = data.values[:, 1:]
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
            elif mode == 'test':
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Normal.csv'))
                data.drop(columns=[' Timestamp', 'Normal/Attack'],inplace=True)
                data = data.values[:, 1:]
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Abormal.csv'))
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)

        self.data_dim = data.shape[-1]
        classification_dir = os.path.join('classification_dataset', dataset)

        if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
            classification_dir = os.path.join(classification_dir, subdata)
        
        classification_dir = os.path.join(classification_dir, scheme)
        
        if mode == 'train':
            anchors = convert_to_windows(data=data, window_size=window_size)
            negative_pairs = np.load(
                os.path.join(classification_dir, 'negative_pairs.npy')
            )
            self.windows = np.concatenate([anchors, negative_pairs], axis=0)

            anchor_nn_indices = np.load(
                os.path.join(classification_dir, 'anchor_nn_indices.npy')
            )
            negative_nn_indices = np.load(
                os.path.join(classification_dir, 'negative_nn_indices.npy')
            )
            
            anchor_nns = []
            negative_nns = []

            for idx in range(anchors.shape[0]):
                anchor_nn_idx = anchor_nn_indices[idx]

                if neighborhood_choice == 'random_choice':
                    anchor_nn_idx = np.random.choice(anchor_nn_idx)
                
                anchor_nns.append(self.windows[anchor_nn_idx])

            for idx in range(negative_pairs.shape[0]):
                negative_nn_idx = negative_nn_indices[idx]

                if neighborhood_choice == 'random_choice':
                    negative_nn_idx = np.random.choice(negative_nn_idx)
                
                negative_nns.append(self.windows[negative_nn_idx])
            
            anchor_nns = np.array(anchor_nns)
            negative_nns = np.array(negative_nns)
            self.nns = np.concatenate([anchor_nns, negative_nns], axis=0)

            anchor_fn_indices = np.load(
                os.path.join(classification_dir, 'anchor_fn_indices.npy')
            )
            negative_fn_indices = np.load(
                os.path.join(classification_dir, 'negative_fn_indices.npy')
            )

            anchor_fns = []
            negative_fns = []

            for idx in range(anchors.shape[0]):
                anchor_fn_idx = anchor_fn_indices[idx]

                if neighborhood_choice == 'random_choice':
                    anchor_fn_idx = np.random.choice(anchor_fn_idx)
                
                anchor_fns.append(self.windows[anchor_fn_idx])

            for idx in range(negative_pairs.shape[0]):
                negative_fn_idx = negative_fn_indices[idx]

                if neighborhood_choice == 'random_choice':
                    negative_fn_idx = np.random.choice(negative_fn_idx)
                
                negative_fns.append(self.windows[negative_fn_idx])
            
            anchor_fns = np.array(anchor_fns)
            negative_fns = np.array(negative_fns)
            self.fns = np.concatenate([anchor_fns, negative_fns], axis=0)

        elif mode == 'test':
            self.windows = convert_to_windows(
                data=data,
                window_size=window_size,
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
            return (self.windows[idx] - self.mean) / self.std


############################## Dataset for DeepSVDD ############################


class DeepSVDDDataset(object):
    """
    Dataset for Classification stage of CARLA.

    Parameters:
        dataset:     Name of the dataset.
        subdata:     If the dataset consists of more than one subdata, the
                     name of subdata.
        window_size: Window size of anchor. Default 200.
        mode:        If 'train', anchors, negative pairs, nearest/furthest
                     neighborhoods are loaded for training. If 'test', only
                     the raw test dataset (with window conversion not applied) 
                     and the test label are loaded.
        scheme:      Which scheme to use the test data. It should be chosen 
                     either ['true', 'carla', 'genias'].
    """
    def __init__(
        self,
        dataset: str,
        subdata: str,
        window_size: int = 200,
        mode: str = 'train',
        scheme: str = 'carla',
        device: str = 'cuda:0'
    ) -> None:
        self.dataset = dataset
        self.subdata = subdata
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode
        
        data_dir = os.path.join('data', dataset)
        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(
                    os.path.join(data_dir, 'train', f'{subdata}.npy')
                )
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
            elif mode == 'test':
                data = np.load(
                    os.path.join(data_dir, 'test', f'{subdata}.npy')
                )
                self.mean, self.std = get_mean_std(x=np.load(
                    os.path.join(data_dir, 'train', f'{subdata}.npy')
                ))
                self.std = np.where(self.std == 0.0, 1.0, self.std)
                labels = np.load(
                    os.path.join(data_dir, 'label', f'{subdata}.npy')    
                )

        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Normal.csv'))
                data.drop(columns=[' Timestamp', 'Normal/Attack'],inplace=True)
                data = data.values[:, 1:]
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
            elif mode == 'test':
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Normal.csv'))
                data.drop(columns=[' Timestamp', 'Normal/Attack'],inplace=True)
                data = data.values[:, 1:]
                self.mean, self.std = get_mean_std(x=data)
                self.std = np.where(self.std == 0.0, 1.0, self.std)
                data = pd.read_csv(os.path.join(data_dir, 'SWaT_Abormal.csv'))
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)

        data_dim = data.shape[-1]
        self.windows = convert_to_windows(data=data, window_size=window_size)

        if mode == 'test': 
            labels = convert_to_windows(data=labels, window_size=window_size)
            window_labels = []
            
            for label in labels:
                if np.sum(label) > 0:
                    window_labels.append(1)
                else:
                    window_labels.append(0)
            
            self.labels = np.array(window_labels).reshape(-1)
            is_anomaly = np.where(self.labels > 0.9)[0]
            
            if scheme == 'carla':
                anomaly_injection = AnomalyInjection()
                for idx in is_anomaly:
                    window = self.windows[idx]
                    window = anomaly_injection(window)
                    self.windows[idx] = window
                
            elif scheme == 'genias':
                vae = VAE(
                    window_size=window_size,
                    data_dim=data_dim,
                    latent_dim=100,
                    depth=10,
                ).to(device)
                
                vae_dir = os.path.join('checkpoints', 'vae', self.dataset)
                
                if self.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
                    vae_dir = os.path.join(vae_dir,  self.subdata)
                vae_ckpt = torch.load(
                    os.path.join(vae_dir, 'epoch_1000.pt'),
                    map_location=device
                    )
                vae.load_state_dict(vae_ckpt['model'])
                vae.eval()

                if self.dataset == 'MSL':
                    patch_coef = 0.4
                elif self.dataset in ['SMAP', 'Yahoo']:
                    patch_coef = 0.2
                else:
                    patch_coef = np.random.choice([0.05, 0.1, 0.2, 0.4, 0.6])

                for idx in is_anomaly:
                    window = torch.from_numpy(self.windows[idx]).float().to(device)
                    window_anomaly = vae.forward(window.unsqueeze(0))[-1]
                    window_anomaly = window.detach().cpu().squeeze(0).numpy()
                    window = patch(
                        x=window.detach().cpu().numpy(),
                        x_tilde=window_anomaly,
                        tau=patch_coef,
                    )
                    self.windows[idx] = window
        
        return

    def __len__(self) -> int:
        return self.windows.shape[0]
    
    def __getitem__(
        self,
        idx: int
        ) -> Union[Matrix, Tuple[Matrix, Array]]:
        if self.mode == 'train':
            window = self.windows[idx]
            window = (window - self.mean) / self.std
            return window
        elif self.mode == 'test':
            window = self.windows[idx]
            window = (window - self.mean) / self.std
            return window, self.labels[idx]
        else:
            NotImplementedError