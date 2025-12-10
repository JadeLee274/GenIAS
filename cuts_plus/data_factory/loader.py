from sklearn.preprocessing import StandardScaler
from cuts_plus.utils.imports import *


def get_mean_std(x: Matrix) -> Vector:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def train_val_split(train: Matrix, train_ratio: float) -> Tuple[Matrix, Matrix]:
    train_len = int(len(train) * train_ratio)
    val = train[train_len:].copy()
    train = train[:train_len]
    return train, val


def convert_to_windows(x: Matrix, window_size: int) -> Array:
    windows = []
    
    for idx in range(x.shape[0] - window_size + 1):
        windows.append(x[idx: idx+window_size])

    windows = np.array(windows)
    return windows


def get_labels(label: Vector, window_size: int) -> Vector:
    labels = []
    for i in range(label.shape[0] - window_size + 1):
        if np.sum(label[i: i+window_size]) != 0:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels).reshape(-1)
    return labels


class Loader(object):
    def __init__(
        self,
        dataset: str,
        subdata: Optional[str],
        mode: str,
        window_size: int,
        train_ratio: float,
    ) -> None:
        assert mode in ['train', 'val', 'test'], "'train', 'val', 'test'"
        assert train_ratio < 1.0, "train_ratio must be smaller than 1.0"
        self.mode = mode

        scaler = StandardScaler()

        data_dir = os.path.join('cuts_plus', 'data', dataset)
        if dataset in ['MSL', 'SMAP', 'SMD']:
            
            train = np.load(os.path.join(data_dir, 'train', f'{subdata}.npy'))
            test = np.load(os.path.join(data_dir, 'test', f'{subdata}.npy'))
            label = np.load(os.path.join(data_dir, 'label', f'{subdata}.npy'))
        
        # mean, std = get_mean_std(x=train)
        # train = (train - mean) / std
        # test = (test - mean) / std

        train, val = train_val_split(train=train, train_ratio=train_ratio)

        train = scaler.fit_transform(train)
        val = scaler.transform(val)
        test = scaler.transform(test)

        self.train = convert_to_windows(x=train, window_size=window_size)
        self.val = convert_to_windows(x=val, window_size=window_size)
        
        self.test = convert_to_windows(x=test, window_size=window_size)
        self.labels = get_labels(label=label, window_size=window_size)

        self.data_dim = self.train.shape[-1]

        return
    
    def __len__(self) -> int:
        if self.mode == 'train':
            return self.train.shape[0]
        elif self.mode == 'val':
            return self.val.shape[0]
        elif self.mode == 'test':
            return self.test.shape[0]
        
    def __getitem__(self, idx: int) -> Union[Array, Tuple[Array, Vector]]:
        if self.mode == 'train':
            return np.float32(self.train[idx])
        elif self.mode == 'val':
            return np.float32(self.val[idx])
        elif self.mode == 'test':
            assert self.test.shape[0] == self.labels.shape[0], \
            "test data length and labels length mismatch"
            return np.float32(self.test[idx]), self.labels[idx]
        