import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from exp.utils.common_import import *
from exp.models.cuts_plus import CUTS_Plus_Net
from exp.models.augmentor import *
from exp.utils.preprocess import *


class CUTSplusDataset(object):
    def __init__(
        self,
        dataset: str,
        subdata: Optional[str],
        mode: str,
        window_size: int,
        train_ratio: float,
        downsample: bool,
        downsample_step: int,
    ) -> None:
        assert mode in ['train', 'val', 'test'], "'train', 'val', 'test'"
        assert train_ratio < 1.0, "train_ratio must be smaller than 1.0"
        
        self.mode = mode

        scaler = StandardScaler()

        data_path = os.path.join('exp', 'data', dataset)
        
        if dataset in ['MSL', 'SMAP', 'SMD']:
            train_data = np.load(
                os.path.join(data_path, 'train', f'{subdata}.npy')
            )
            test_data = np.load(
                os.path.join(data_path, 'test', f'{subdata}.npy')
            )
            test_labels = np.load(
                os.path.join(data_path, 'label', f'{subdata}.npy')
            )
        
        elif dataset == 'SWaT':
            train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
            train_data = train_data.iloc[:, :-1].to_numpy()

            test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))

            test_labels = test_data.iloc[:, -1] == 1
            test_labels = test_labels.to_numpy().astype(int)

            test_data = test_data.iloc[:, :-1].to_numpy()
        
        elif dataset == 'WADI':
            train_data = pd.read_csv(
                os.path.join(data_path, 'WADI_14days_new.csv')
            )
            train_data = train_data.dropna(axis='columns', how='all').dropna()
            train_data = train_data.iloc[:, 3:].to_numpy()

            test_data = pd.read_csv(
                os.path.join(data_path, 'WADI_attackdataLABLE.csv'),
                header=1,
            )
            test_data = test_data.dropna(axis='columns', how='all').dropna()

            test_labels = test_data.iloc[:, -1] == -1
            test_labels = test_labels.to_numpy().astype(int)

            test_data = test_data.iloc[:, 3:-1].to_numpy()

        train_data, val_data = train_val_split(
            train_data=train_data,
            train_ratio=train_ratio,
        )

        if downsample:
            train_data = downsample_data(
                data=train_data,
                downsample_step=downsample_step,
            )
            val_data = downsample_data(
                data=val_data,
                downsample_step=downsample_step,
            )
            test_data = downsample_data(
                data=test_data,
                downsample_step=downsample_step,
            )
            test_labels = downsample_data(
                data=test_labels,
                downsample_step=downsample_step,
            )

        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

        self.train_data: Array = convert_to_windows(
            data=train_data,
            window_size=window_size,
        )
        self.val_data: Array = convert_to_windows(
            data=val_data,
            window_size=window_size,
        )
        self.test_data: Array = convert_to_windows(
            data=test_data,
            window_size=window_size,
        )
        test_labels = convert_to_windows(
            data=test_labels,
            window_size=window_size,
        )

        labels = []

        for i in range(test_labels.shape[0] - window_size + 1):
            if np.sum(test_labels[i: i+window_size]) != 0:
                labels.append(1)
            else:
                labels.append(0)
                
        self.test_labels = np.array(labels).reshape(-1)

        self.data_dim = self.train_data.shape[-1]

        return
    
    def __len__(self) -> int:
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'val':
            return self.val_data.shape[0]
        elif self.mode == 'test':
            return self.test_data.shape[0]
        
    def __getitem__(self, idx: int) -> Union[Array, Tuple[Array, Vector]]:
        if self.mode == 'train':
            return np.float32(self.train_data[idx])
        elif self.mode == 'val':
            return np.float32(self.val_data[idx])
        elif self.mode == 'test':
            assert self.test_data.shape[0] == self.test_labels.shape[0], \
            "test data length and labels length mismatch"
            return np.float32(self.test_data[idx]), self.test_labels[idx]


class PositiveAugmentor(nn.Module):
    def __init__(
        self,
        noise_injection_step: int,
        prediction_step: int,
        noise_level: float,
        n_nodes: int,
    ) -> None:
        super().__init__()
        self.noise_injection_step = noise_injection_step
        self.prediction_step = prediction_step
        self.noise_level = noise_level
        self.causal_discoverer = CUTS_Plus_Net(n_nodes=n_nodes, data_dim=1)

        return
    
    def init_causal_discoverer(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        seed: int,
    ) -> None:
        cuts_plus_dir = os.path.join(
            os.getcwd(), 'exp', 'checkpoints', data
        )
        # if not os.path.exists(cuts_plus_dir):
        #     cuts_plus_dir = os.path.join(
        #         os.getcwd(), 'cuts_plus', 'checkpoints', data
        #     )

        if subdata is not None:
            cuts_plus_dir = os.path.join(cuts_plus_dir, subdata)
            
        cuts_plus_dir = os.path.join(cuts_plus_dir, time, 'best_model.pt')
        cuts_plus_ckpt = torch.load(cuts_plus_dir)

        assert cuts_plus_ckpt['seed'] == seed, \
        f"CUTS+ seed {cuts_plus_ckpt['seed']} and current seed {seed} mismatch"
        
        self.causal_discoverer.load_state_dict(cuts_plus_ckpt['model'])
        self.causal_discoverer.eval()
        
        return

    def forward(self, x: Tensor, causality_matrix: Matrix) -> Tensor:
        x_positive = x.clone()
        causality_matrix = (causality_matrix > 0.5).int()
        
        start_node = random.choice(range(len(causality_matrix)))
        self.start_node = start_node
        
        effects = [
            i for i, val in enumerate(causality_matrix[start_node]) if val == 1
        ]
        self.effects = effects

        x_positive[:, self.noise_injection_step, start_node] \
        += torch.randn(x.size(0)).to(x.device) * self.noise_level

        with torch.no_grad():
            graph = (self.causal_discoverer.causality_mtx > 0.5).float()
            graph = graph[None].expand(x.size(0), -1, -1)
            graph_output = self.causal_discoverer.forward(
                x=x_positive[:, :self.noise_injection_step],
                fwd_graph=graph,
            ).transpose(1, 2)
        
        x_positive[:, -self.prediction_step, effects] \
        = graph_output[:, -self.prediction_step, effects]

        return x_positive


class PerturbationDataset(object):
    def __init__(
        self,
        data: str,
        subdata: Optional[str],
        mode: str,
        window_size: int,
        downsample: bool = False,
        downsample_step: int = 5,
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

        elif data == 'SWaT':
            train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
            train_data = train_data.iloc[:, :-1].to_numpy()

            test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))

            test_labels = test_data.iloc[:, -1] == 1
            test_labels = test_labels.to_numpy().astype(int)

            test_data = test_data.iloc[:, :-1].to_numpy()
        
        elif data == 'WADI':
            train_data = pd.read_csv(
                os.path.join(data_path, 'WADI_14days_new.csv')
            )
            train_data = train_data.dropna(axis='columns', how='all').dropna()
            train_data = train_data.iloc[:, 3:].to_numpy()

            test_data = pd.read_csv(
                os.path.join(data_path, 'WADI_attackdataLABLE.csv'),
                header=1,
            )
            test_data = test_data.dropna(axis='columns', how='all').dropna()

            test_labels = test_data.iloc[:, -1] == -1
            test_labels = test_labels.to_numpy().astype(int)

            test_data = test_data.iloc[:, 3:-1].to_numpy()
        
        if downsample:
            train_data = downsample_data(
                data=train_data,
                downsample_step=downsample_step,
            )
            test_data = downsample_data(
                data=test_data,
                downsample_step=downsample_step,
            )
            test_labels = downsample_data(
                data=test_labels,
                downsample_step=downsample_step,
            )
        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

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
            window = self.train_data[idx]
            window = self.scaler.transform(window)
            return window
        
        elif self.mode == 'test':
            window = self.test_data[idx]
            window = self.scaler.transform(window)
            return window
        

class PretextDataset(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        seed: int,
        perturbator_mode: str,
        positive_augmentor_time: str,
        perturbator_time: str,
        processor_num: int,
        downsample: bool,
        downsample_step: int,
        apply_patch: bool,
        patch_after: str,
        deviation_mode: str,
        non_constant_dim_tau: float,
        constant_dim_tau: float,
        save_negative_pairs: bool,
        plot_perturbation: bool,
    ) -> None:
        assert perturbator_mode in ['plad', 'tcn_perturbator'], \
        "perturbator_mode has to be either 'plad' or 'tcn_perturbator'"

        data_path = os.path.join('exp', 'data', data)

        if data in ['MSL', 'SMAP', 'SMD']:
            train_data_path = os.path.join(data_path, 'train')
            train_data = np.load(
                os.path.join(train_data_path, f'{subdata}.npy')
            )
        
        elif data == 'SWaT':
            train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
            train_data = train_data.iloc[1:, 1:-1].to_numpy()
            
            test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))
            test_data = test_data.iloc[1: 1:-1].to_numpy()
        
        elif data == 'WADI':
            train_data = pd.read_csv(
                os.path.join(data_path, 'WADI_14days_new.csv')
            )
            train_data = train_data.dropna(axis='columns', how='all').dropna()
            train_data = train_data.iloc[:, 3:].to_numpy()

            test_data = pd.read_csv(
                os.path.join(data_path, 'WADI_attackLABLE.csv'),
                header=1,
            )
            test_data = test_data.dropna(axis='columns', how='all').dropna()
            test_data = test_data.iloc[:, 3:-1].to_numpy()
        
        if downsample:
            train_data = downsample_data(
                data=train_data,
                downsample_step=downsample_step,
            )
            test_data = downsample_data(
                data=test_data,
                downsample_step=downsample_step,
            )
        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        # Anchors are normalized for augmentation.
        self.anchors = convert_to_windows(
            data=train_data,
            window_size=window_size,
        )

        data_dim = self.anchors.shape[-1]
        self.data_dim = data_dim
        n_nodes = data_dim
        
        # Initialize positive augmentor.
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
            seed=seed,
        )
        positive_augmentor = positive_augmentor.to(processor)
        causality_matrix = positive_augmentor.causal_discoverer.causality_mtx
        positive_augmentor.eval()
        
        # Initialize negative augmentor(perturbator).
        if perturbator_mode == 'plad':
            latent_dim = 100 if data_dim > 1 else 50
            perturbator = PLAD(
                window_size=window_size,
                data_dim=data_dim,
                latent_dim=latent_dim,
            )
            perturbator.init_plad(
                time=perturbator_time,
                data=data,
                subdata=subdata,
                epoch=50,
            )
        
        elif perturbator_mode == 'tcnperturbator':
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

        assert perturbator.seed == seed, \
        f"perturbator seed {perturbator.seed} and current seed {seed} mismatch"
        
        perturbator = perturbator.to(processor)
        perturbator.eval()

        anchor_loader = DataLoader(
            dataset=self.anchors,
            batch_size=len(self.anchors),
            shuffle=False,
        )

        # Make positive pairs.
        anchors = next(iter(anchor_loader)).to(processor)
        assert len(anchors) == len(self.anchors), \
        f"len(anchors) {len(anchors)} != len(self.anchors) {len(self.anchors)}"
        
        # Positive pairs are (sort of) normalized.
        self.positive_pairs = positive_augmentor.forward(
            x=anchors,
            causality_matrix=causality_matrix,
        ).detach().cpu().numpy()

        # Negative pairs are also normalized.
        if perturbator_mode == 'plad':
            negative_pairs, _ = perturbator.forward(x=anchors)
        
        elif perturbator_mode == 'tcn_perturbator':
            _, negative_pairs, _, _ = perturbator.forward(x=anchors)

        if apply_patch:
            assert patch_after in ['perturbator', 'positive_augmentor'], \
            "patch after 'perturbator' or 'positive_augmentor'"

        if (apply_patch and patch_after == 'perturbator'):
            negative_pairs_without_patch = negative_pairs
            negative_pairs = negative_pairs.detach().cpu().numpy()
            negative_pairs_temp = np.empty_like(negative_pairs)

            for i in range(len(anchors)):
                anchor = self.anchors[i]
                negative_pair = negative_pairs[i]
                amplitude_list = []
                amplitude_pert_list = []

                for dim in range(data_dim):
                    x_d = anchor[..., dim]
                    x_pert_d = negative_pair[..., dim]
                    
                    amplitude_d = np.max(x_d) - np.min(x_d)
                    amplitude_pert_d = np.max(x_pert_d) - np.min(x_pert_d)

                    amplitude_list.append(amplitude_d)
                    amplitude_pert_list.append(amplitude_pert_d)
                
                negative_pairs_temp[i] = patch(
                    x=anchor,
                    x_pert=negative_pair,
                    amplitude_list=amplitude_list,
                    amplitude_pert_list=amplitude_pert_list,
                    deviation_mode=deviation_mode,
                    non_constant_dim_tau=non_constant_dim_tau,
                    constant_dim_tau=constant_dim_tau,
                )
            
            negative_pairs = negative_pairs_temp
            
            negative_loader = DataLoader(
                dataset=negative_pairs,
                batch_size=len(negative_pairs),
                shuffle=False,
            )

            negative_pairs = next(iter(negative_loader)).to(processor)
            assert len(negative_pairs) == len(self.anchors), \
            f"len(negative_pairs) {len(anchors)} != len(self.anchors) {len(self.anchors)}"

            negative_pairs_without_patch = positive_augmentor.forward(
                x=negative_pairs_without_patch,
                causality_matrix=causality_matrix,
            ).detach().cpu().numpy()
        
            negative_pairs = positive_augmentor.forward(
                x=negative_pairs,
                causality_matrix=causality_matrix,
            ).detach().cpu().numpy()

            self.negative_pairs = negative_pairs
            negative_pairs_with_patch = negative_pairs

        elif (apply_patch and patch_after == 'positive_augmentor'):
            negative_pairs = positive_augmentor.forward(
                x=negative_pairs,
                causality_matrix=causality_matrix,
            ).detach().cpu().numpy()
            negative_pairs_without_patch = negative_pairs
            negative_pairs_temp = np.empty_like(negative_pairs)

            for i in range(len(anchors)):
                anchor = self.anchors[i]
                negative_pair = negative_pairs[i]
                amplitude_list = []
                amplitude_pert_list = []

                for dim in range(data_dim):
                    x_d = anchor[..., dim]
                    x_pert_d = negative_pair[..., dim]
                    
                    amplitude_d = np.max(x_d) - np.min(x_d)
                    amplitude_pert_d = np.max(x_pert_d) - np.min(x_pert_d)

                    amplitude_list.append(amplitude_d)
                    amplitude_pert_list.append(amplitude_pert_d)
                
                negative_pairs_temp[i] = patch(
                    x=anchor,
                    x_pert=negative_pair,
                    amplitude_list=amplitude_list,
                    amplitude_pert_list=amplitude_pert_list,
                    deviation_mode=deviation_mode,
                    non_constant_dim_tau=non_constant_dim_tau,
                    constant_dim_tau=constant_dim_tau,
                )
            
            self.negative_pairs = negative_pairs_temp
            negative_pairs_with_patch = negative_pairs_temp
        
        else:
            self.negative_pairs = positive_augmentor.forward(
                x=negative_pairs,
                causality_matrix=causality_matrix,
            ).detach().cpu().numpy()
            negative_pairs_without_patch = self.negative_pairs

        classification_data_save_dir = os.path.join(
            'exp', 'data', 'classification_data', data
        )

        if data in ['MSL', 'SMAP', 'SMD']:
            classification_data_save_dir = os.path.join(
                classification_data_save_dir, subdata
            )
        
        os.makedirs(classification_data_save_dir, exist_ok=True)

        if save_negative_pairs:
            np.save(
                file=os.path.join(
                    classification_data_save_dir, f'negative_pairs_{time}.npy'
                ),
                arr=self.negative_pairs,
            )
        
        if plot_perturbation:
            plot_dir = os.path.join(os.getcwd(), 'plots', data)

            if apply_patch:
                plot_dir = os.path.join(
                    plot_dir,
                    f'patch_after_{patch_after}',
                    f'nonconstant_dim_tau_{non_constant_dim_tau}',
                    f'constant_dim_tau_{constant_dim_tau}'
                )
            
            if subdata is not None:
                plot_dir = os.path.join(plot_dir, subdata)
            
            os.makedirs(plot_dir, exist_ok=True)

            num_cols = 5
            num_rows = data_dim // num_cols

            if data_dim % num_cols != 0:
                num_rows = (data_dim // num_cols) + 1

            for i in range(len(self.anchors)//window_size):
                len_ratio = int(num_rows // num_cols)
                
                if num_rows % num_cols != 0:
                    len_ratio = int(num_rows // num_cols) + 1
                
                figsize = (20, 10*len_ratio)
                
                fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
                idx = i * window_size

                for j in range(0, data_dim):
                    ax[j//5, j%5].plot(
                        self.anchors[idx][:, j],
                        color='red',
                        linestyle='--',
                        label='anchor',
                    )
                    ax[j//5, j%5].plot(
                        negative_pairs_without_patch[idx][:, j],
                        color='blue',
                        linestyle=':',
                        label='negative pair without patch',
                    )
                    if apply_patch:
                        ax[j//5, j%5].plot(
                            negative_pairs_with_patch[idx][:, j],
                            color='green',
                            label='negative pair with patch',
                        )
                
                handles, labels = ax[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right')
                fig.savefig(
                    os.path.join(plot_dir, f'comparison_patch_{idx}.png')
                )
                plt.close()
    
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
        downsample: bool,
        downsample_step: int,
        mode: str = 'train',
    ) -> None:
        assert mode in ['train', 'test'], \
        "mode must be either 'train' or 'test'"
        self.mode = mode

        data_path = os.path.join('exp', 'data', data)

        if data in ['MSL', 'SMAP', 'SMD']:
            train_data = np.load(
                os.path.join(data_path, 'train', f'{subdata}.npy')
            )
            test_data = np.load(
                os.path.join(data_path, 'test', f'{subdata}.npy')
            )
            test_labels = np.load(
                os.path.join(data_path, 'label', f'{subdata}.npy')
            )
        
        elif data == 'SWaT':
            train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
            train_data = train_data.iloc[1:, 1:-1].to_numpy()
            
            test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))

            test_labels = test_data.iloc[1:, -1] == 1
            test_labels = test_data.to_numpy().astype(int)

            test_data = test_data.iloc[1: 1:-1].to_numpy()
        
        elif data == 'WADI':
            train_data = pd.read_csv(
                os.path.join(data_path, 'WADI_14days_new.csv')
            )
            train_data = train_data.dropna(axis='columns', how='all').dropna()
            train_data = train_data.iloc[:, 3:].to_numpy()

            test_data = pd.read_csv(
                os.path.join(data_path, 'WADI_attackLABLE.csv'),
                header=1,
            )
            test_data = test_data.dropna(axis='columns', how='all').dropna()

            test_labels = test_data.iloc[:, -1] == -1
            test_labels = test_labels.to_numpy().astype(int)

            test_data = test_data.iloc[:, 3:-1].to_numpy()

        if downsample:
            train_data = downsample_data(
                data=train_data,
                downsample_step=downsample_step,
            )
            test_data = downsample_data(
                data=test_data,
                downsample_step=downsample_step,
            )
            test_labels = downsample_data(
                data=test_labels,
                downsample_step=downsample_step,
            )
        
        scaler = StandardScaler()
        scaler.fit(train_data)

        # Normalize train/test data
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        self.data_dim = train_data.shape[-1]
        
        classification_data_path = os.path.join(
            'exp', 'data', 'classification_data', data
        )
        if data in ['MSL', 'SMAP', 'SMD']:
            classification_data_path = os.path.join(
                classification_data_path, subdata
            )
        
        if mode == 'train':
            # Anchors are normalized within this dataset.
            anchors = convert_to_windows(
                data=train_data,
                window_size=window_size,
            )
            # Negative pairs are already normallized in the pretext stage.
            negative_pairs: Array = np.load(
                os.path.join(
                    classification_data_path,
                    f'negative_pairs_{time}.npy'
                )
            )
            self.windows = np.concatenate([anchors, negative_pairs], axis=0)

            # Load indices for nearest neighborhoods.
            anchor_nn_indices = np.load(
                os.path.join(
                    classification_data_path, f'anchor_nn_indices_{time}.npy'
                )
            )
            negative_nn_indices = np.load(
                os.path.join(
                    classification_data_path, f'negative_nn_indices_{time}.npy'
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
                    classification_data_path,
                    f'anchor_fn_indices_{time}.npy'
                )
            )
            negative_fn_indices = np.load(
                os.path.join(
                    classification_data_path,
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
                data=test_data,
                window_size=window_size,
            )
            labels = convert_to_windows(
                data=test_labels,
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
