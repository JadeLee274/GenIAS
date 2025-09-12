import pandas as pd
from utils.common_import import *
from utils.preprocess import *
DATA_PATH = '/data/seungmin/'
CLASSIFICATION_PATH = 'temp/classification_temp'


class PretextDataset(object):
    def __init__(
        self,
        data_type: str = 'MSL_SEPARATED',
        data_name: str = 'M-6',
        window_size: int = 200,
        mode: str = 'train',
    ) -> None:
        assert mode in ['train', 'test'], "mode is either 'train' or 'test'"
        self.mode = mode

        data_dir = os.path.join(
            DATA_PATH, data_type, mode, f'{data_name}_{mode}.npy')
        data = np.load(data_dir)
        self.data_dim = data.shape[-1]

        if mode == 'test':
            label_dir = os.path.join(
                DATA_PATH, data_type, mode, f'{data_name}_{mode}_label.npy'
            )
            label = np.load(label_dir)
            self.labels = convert_to_windows(label)
        
        self.mean, self.std = get_mean_std(x=data)
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        self.anchors = convert_to_windows(data=data, window_size=window_size)

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
            CLASSIFICATION_PATH, data_type, data_name
        )
        os.makedirs(negative_save_dir, exist_ok=True)
        np.save(
            file=os.path.join(negative_save_dir, 'negative_pairs.npy'),
            arr=self.negative_pairs
        )

        return
    
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
                classification_data_dir, 'negative_pairs.npy'
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
