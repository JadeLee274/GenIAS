from tqdm import tqdm
from torch.utils.data import DataLoader
from exp.utils.common_import import *
from exp.utils.preprocess import *
from exp.models.augmentor import *


class PerturbationDataset(object):
    def __init__(
        self,
        data: str,
        subdata: Optional[str],
        mode: str,
        window_size: int,
    ) -> None:
        self.mode = mode

        data_path = os.path.join('exp', 'data', data)
        if data in ['MSL', 'SMAP', 'SMD']:
            train_data_path = os.path.join(data_path, 'train')
            test_data_path = os.path.join(data_path, 'test')
            label_path = os.path.join(data_path, 'label')
            train_data = np.load(
                os.path.join(train_data_path, f'{subdata}.npy')
            )
            test_data = np.load(
                os.path.join(test_data_path, f'{subdata}.npy')
            )
            test_labels = np.load(os.path.join(label_path, f'{subdata}.npy'))
        
        self.mean, self.std = get_mean_std(x=train_data)

        self.train_data = convert_to_windows(
            data=train_data,
            window_size=window_size,
        )
        self.test_data = convert_to_windows(
            data=test_data,
            window_size=window_size,
        )
        self.data_dim = self.train_data.shape[-1]
        self.n_nodes = self.train_data.shape[-1]

        test_labels = convert_to_windows(
            data=test_labels,
            window_size=window_size,
        )
        self.test_labels = []
        
        for label in test_labels:
            if np.sum(label) > 0:
                self.test_labels.append(1)
            else:
                self.test_labels.append(0)
                
        self.test_labels = np.array(self.test_labels, dtype=np.int32)
        self.test_labels = self.test_labels.reshape(-1)

        return
    
    def __len__(self) -> int:
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'test':
            return self.test_data.shape[0]

    def __getitem__(self, idx: int) -> Matrix:
        if self.mode == 'train':
            window = (self.train_data[idx] - self.mean) / self.std
            return window
        
        elif self.mode == 'test':
            window = (self.test_data[idx] - self.mean) / self.std
            return window
        

class PretextDataset(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        positive_augmentor_time: str,
        perturbator_time: str,
        processor_num: int = 0,
        apply_patch: bool = False,
    ) -> None:
        data_path = os.path.join('exp', 'data', data)
        if data in ['MSL', 'SMAP', 'SMD']:
            train_data_path = os.path.join(data_path, 'train')
            train_data = np.load(
                os.path.join(train_data_path, f'{subdata}.npy')
            )
        
        mean, std = get_mean_std(x=train_data)
        train_data = (train_data - mean) / std
            
        self.anchors = convert_to_windows(
            data=train_data,
            window_size=window_size,
        )
        data_dim = self.anchors.shape[-1]
        self.data_dim = data_dim
        n_nodes = data_dim
        
        # Make positive pairs
        pretidction_step = 1
        noise_injection_step = window_size - pretidction_step
        
        processor = torch.device(f'cuda:{processor_num}')

        positive_augmentor = PositiveAugmentor(
            noise_injection_step=noise_injection_step,
            prediction_step=pretidction_step,
            noise_level=0.1,
            n_nodes=n_nodes,
        )
        positive_augmentor.init_causal_discoverer(
            time=positive_augmentor_time,
            data=data,
            subdata=subdata,
        )
        positive_augmentor = positive_augmentor.to(processor)
        causality_matrix = positive_augmentor.causal_discoverer.causality_mtx
        positive_augmentor.eval()

        anchor_loader = DataLoader(
            dataset=self.anchors,
            batch_size=200,
            shuffle=False,
        )

        self.positive_pairs = torch.tensor([], dtype=torch.float32)
        self.positive_pairs = self.positive_pairs.to(processor)

        for x in anchor_loader:
            x: Tensor = x.to(processor)
            positive_pair = positive_augmentor.forward(
                x=x,
                causality_matrix=causality_matrix,
            )
            self.positive_pairs = torch.cat(
                tensors=[self.positive_pairs, positive_pair],
                dim=0,
            )
        
        self.positive_pairs = self.positive_pairs.detach().cpu().numpy()

        # Make negative pairs
        perturbator = TCNPerturbator(
            window_size=window_size,
            data_dim=data_dim,
            latent_perturbator_factor=2.0,
        )
        perturbator.init_perturbator(
            time=perturbator_time,
            data=data,
            subdata=subdata,
        )
        perturbator = perturbator.to(processor)
        perturbator.eval()

        self.negative_pairs = torch.tensor([], dtype=torch.float32)
        self.negative_pairs = self.negative_pairs.to(processor)
        
        for x in anchor_loader:
            x: Tensor = x.to(processor)
            _, negative_pair, _, _ = perturbator.forward(x=x)
            negative_pair = positive_augmentor.forward(
                x=negative_pair,
                causality_matrix=causality_matrix,
            )
            if apply_patch:
                negative_pair = patch(x=x, x_pert=negative_pair)
            self.negative_pairs = torch.cat(
                tensors=[self.negative_pairs, negative_pair],
                dim=0,
            )
        
        self.negative_pairs = self.negative_pairs.detach().cpu().numpy()

        # Save negative pairs for classification stage
        classification_data_save_dir = os.path.join(
            'exp', 'data', 'classification_data', data
        )

        if data in ['MSL', 'SMAP', 'SMD']:
            classification_data_save_dir = os.path.join(
                classification_data_save_dir, subdata
            )
        
        os.makedirs(classification_data_save_dir, exist_ok=True)
        
        np.save(
            file=os.path.join(
                classification_data_save_dir, f'negative_pairs_{time}.npy'
            ),
            arr=self.negative_pairs,
        )
    
        return
        
    def __len__(self) -> int:
        return self.anchors.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Matrix, Matrix, Matrix]:
        anchor = self.anchors[idx]
        positive_pair = self.positive_pairs[idx]
        negaitve_pair = self.negative_pairs[idx]

        return anchor, positive_pair, negaitve_pair


class ClassificationDatasaet(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        mode: str = 'train',
    ) -> None:
        assert mode in ['train', 'test'], \
        "mode must be either 'train' or 'test"
        self.mode = mode

        data_dir = os.path.join('exp', 'data', data)

        if data in ['MSL', 'SMAP', 'SMD']:
            self.train_data = np.load(
                os.path.join(data_dir, 'train', f'{subdata}.npy')
            )
            self.test_data = np.load(
                os.path.join(data_dir, 'test', f'{subdata}.npy')
            )
            self.test_labels = np.load(
                os.path.join(data_dir, 'label', f'{subdata}.npy')
            )
        
        self.data_dim = self.train_data.shape[-1]
        mean, std = get_mean_std(x=self.train_data)
        
        classification_data_dir = os.path.join(
            'exp', 'data', 'classification_data', data
        )
        if data in ['MSL', 'SMAP', 'SMD']:
            classification_data_dir = os.path.join(
                classification_data_dir, subdata
            )
        
        if mode == 'train':
            anchors = convert_to_windows(
                data=self.train_data,
                window_size=window_size,
            )
            negative_pairs: Array = np.load(
                os.path.join(
                    classification_data_dir,
                    f'negative_pairs_{time}.npy'
                )
            )
            self.anchors = (anchors - mean) / std
            self.negative_pairs = negative_pairs
            self.windows = np.concatenate([anchors, negative_pairs], axis=0)

            # Load indices for nearest neighborhoods.
            anchor_nn_indices = np.load(
                os.path.join(
                    classification_data_dir, f'anchor_nn_indices_{time}.npy'
                )
            )
            negative_nn_indices = np.load(
                os.path.join(
                    classification_data_dir, f'negative_nn_indices_{time}.npy'
                )
            )

            # Make arrays consisting of nearest neighborhood of anchors.
            anchor_nns = []

            for idx in range(anchors.shape[0]):
                anchor_nn_idx = anchor_nn_indices[idx]
                anchor_nn_idx = np.random.choice(anchor_nn_idx)
                anchor_nns.append(self.windows[anchor_nn_idx])
            
            anchor_nns = np.array(anchor_nns, dtype=np.float32)

            # Make arrays consisting of nearest neighborhood of negative pairs.
            negative_nns = []

            for idx in range(negative_pairs.shape[0]):
                negative_nn_idx = negative_nn_indices[idx]
                negative_nn_idx = np.random.choice(negative_nn_idx)
                negative_nns.append(self.windows[negative_nn_idx])

            negative_nns = np.array(negative_nns, dtype=np.float32)
        
            # Make arrays consisting of nearest neighborhoods.
            self.nns = np.concatenate([anchor_nns, negative_nns], axis=0)

            # Load indices for furthest neighborhoods.
            anchor_fn_indices = np.load(
                os.path.join(
                    classification_data_dir,
                    f'anchor_fn_indices_{time}.npy'
                )
            )
            negative_fn_indices = np.load(
                os.path.join(
                    classification_data_dir,
                    f'negative_fn_indices_{time}.npy'
                )
            )

            # Make arrays consisting of furthest neighborhood of anchors.
            anchor_fns = []

            for idx in range(anchors.shape[0]):
                anchor_fn_idx = anchor_fn_indices[idx]
                anchor_fn_idx = np.random.choice(anchor_fn_idx)
                anchor_fns.append(self.windows[anchor_fn_idx])

            anchor_fns = np.array(anchor_fns, dtype=np.float32)

            # Make arrays consisting of furthest neighborhood of negative pairs.
            negative_fns = []

            for idx in range(negative_pairs.shape[0]):
                negative_fn_idx = negative_fn_indices[idx]
                negative_fn_idx = np.random.choice(negative_fn_idx)
                negative_fns.append(self.windows[negative_fn_idx])
            
            negative_fns = np.array(negative_fns, dtype=np.float32)

            self.fns = np.concatenate([anchor_fns, negative_fns], axis=0)

        elif mode == 'test':
            self.windows = convert_to_windows(
                data=self.test_data,
                window_size=window_size,
            )
            labels = convert_to_windows(
                data=self.test_labels,
                window_size=window_size,
            )
            window_labels = []
            
            for label in labels:
                if np.sum(label) > 0:
                    window_labels.append(1)
                else:
                    window_labels.append(0)
            
            self.test_labels = np.array(window_labels)

        return
    
    def __len__(self) -> int:
        return self.windows.shape[0]
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[Tuple[Matrix, Matrix, Matrix], Matrix]:
        if self.mode == 'train':
            window = self.windows[idx]
            nearest_neighbor = self.nns[idx]
            furthest_neighbor = self.fns[idx]
            return window, nearest_neighbor, furthest_neighbor
        
        elif self.mode == 'test':
            window = self.windows[idx]
            return window
