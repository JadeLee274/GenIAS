import torch
import numpy as np
from exp.models import TCNPerturbator
from exp.data_factory.loader import PerturbationDataset, PositiveAugmentor
from exp.utils.utils import str2bool
from sklearn.decomposition import PCA
from openTSNE import TSNE
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help="Name of dataset.",
    )
    parser.add_argument(
        '--sub-dataset',
        type=str,
        required=True,
        help="Name of sub-dataset.",
    )
    parser.add_argument(
        '--num-sample',
        type=int,
        default=100,
        help="Number of anomaly samples for t-SNE visualization. Default 100.",
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help="Length of window. Default 10.",
    )
    parser.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="GPU number. Default 0."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed. Default 42.'
    )
    parser.add_argument(
        '--time-or-dir',
        type=str,
        choices=['time', 'dir'],
        default='time',
        help="Whether to load pre-trained model by timestamp or directory. " \
             "Default 'time'.",
    )
    parser.add_argument(
        '--load-positive-augmentor',
        type=str,
        help="The timestamp (or directory) of pre-trained CUTS+.",
    )
    parser.add_argument(
        '--load-perturbator',
        type=str,
        help="The timestamp (or directory) of pre-trained perturbator.",
    )
    parser.add_argument(
        '--downsample',
        type=str2bool,
        default=False,
        help="Whether to downsample data or not. Default False."
    )
    parser.add_argument(
        '--downsample-step',
        type=int,
        default=5,
        help="Step size of downsampling. Default 5.",
    )
    parser.add_argument(
        '--deviation-mode',
        type=str,
        default='abs',
        help="The mode of deviation for patching. Default 'abs'."
    )
    parser.add_argument(
        '--non-constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to non-constant dim.",
    )
    parser.add_argument(
        '--constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to constant dim."
    )
    parser.add_argument(
        '--pca-dim',
        type=int,
        default=50,
        help="Dimension for PCA. Note; data are flattened " \
            "so the actual dimension is window_size * data_dim. Default 50.",
    )
    parser.add_argument(
        '--anomaly-or-normal',
        type=str,
        choices=['anomaly', 'normal'],
        default='anomaly',
        help="Whether to use anomaly or normal data for injection. Default 'anomaly'.",
    )
    args = parser.parse_args()

    # (1) Load dataset
    dataset = PerturbationDataset(
        data=args.data,
        subdata=args.sub_dataset,
        mode='test',
        window_size=args.window_size,
        downsample=args.downsample,
        downsample_step=args.downsample_step
    )
    anomaly_data_index = np.where(dataset.test_labels==1)[0]
    len_data = min(len(anomaly_data_index), args.num_sample)
    if len(anomaly_data_index) < args.num_sample:
        print(
            'The number of anomalies in test data is less than num_sample.' \
            ' All anomalies are used for t-SNE visualization.'
            )
    else:
        anomaly_data_index = np.random.choice(
            anomaly_data_index, size=args.num_sample, replace=False
            )
    normal_data = dataset.train_data[np.random.choice(
        len(dataset.train_data), size=len_data, replace=False
    )]
    anomaly_data = dataset.test_data[anomaly_data_index]
    data_dim = anomaly_data.shape[-1]

    # (2) Load pretrained models
    prediction_step = 1
    noise_injection_step = args.window_size - prediction_step
    
    processor = torch.device(f'cuda:{args.gpu_num}')

    positive_augmentor = PositiveAugmentor(
        noise_injection_step=noise_injection_step,
        prediction_step=prediction_step,
        noise_level=0.1,
        n_nodes=data_dim,
    )
    perturbator = TCNPerturbator(
        window_size=args.window_size,
        data_dim=data_dim,
        latent_perturbator_factor=2.0,
    )
    if args.time_or_dir == 'time':
        positive_augmentor.init_causal_discoverer(
            time=args.load_positive_augmentor,
            data=args.data,
            subdata=args.subdata,
            seed=args.seed,
        )
        perturbator.init_perturbator(
            time=args.load_perturbator,
            data=args.data,
            subdata=args.subdata,
        )
    elif args.time_or_dir == 'dir':
        cuts_plus_dir = os.path.join(
            args.load_positive_augmentor, args.data, args.sub_dataset, 'cuts_plus.pt'
            )
        cuts_plus_ckpt = torch.load(cuts_plus_dir, weights_only=True)
        positive_augmentor.causal_discoverer.load_state_dict(
            cuts_plus_ckpt['model']
            )
        
        perturbator_dir = os.path.join(
            args.load_perturbator, args.data, args.sub_dataset, 'perturbator.pt'
        )
        perturbator_ckpt = torch.load(perturbator_dir, weights_only=True)
        perturbator.load_state_dict(perturbator_ckpt['perturbator'])

    positive_augmentor = positive_augmentor.to(processor)
    causality_matrix = positive_augmentor.causal_discoverer.causality_mtx
    positive_augmentor.eval()
    perturbator = perturbator.to(processor)
    perturbator.eval()

    # (3) Inject anomalies
    if args.anomaly_or_normal == 'anomaly':
        inject_data = torch.from_numpy(anomaly_data).to(processor)
    else:
        inject_data = torch.from_numpy(normal_data).to(processor)
    _, negative_pairs_perturbator, _, _ = perturbator.forward(x=inject_data)
    # Clone for not early patching
    negative_pairs_without_patch = negative_pairs_perturbator.clone()
    # Sample 1: neg -> perturbator
    negative_pairs_perturbator_numpy = negative_pairs_perturbator.detach().cpu().numpy()
    
    amplitude_d= torch.max(inject_data, dim=1)[0] - torch.min(inject_data, dim=1)[0] # (B, F)
    amplitude_pert_d= torch.max(negative_pairs_perturbator, dim=1)[0] - \
        torch.min(negative_pairs_perturbator, dim=1)[0] # (B, F)
    if args.deviation_mode == 'abs':
        deviation = torch.abs(inject_data - negative_pairs_perturbator) # (B, T, F)
    elif args.deviation_mode == 'square':
        deviation = (inject_data - negative_pairs_perturbator) ** 2 # (B, T, F)
    else:
        raise ValueError("Deviation mode should be 'abs' or 'square'")
    threshold = torch.where(
        amplitude_d != 0,
        args.non_constant_dim_tau * amplitude_d,
        args.constant_dim_tau * amplitude_pert_d
        ) # (B, F)
    threshold = threshold[:, None, :].expand(-1, args.window_size, -1) # (B, T, F)
    negative_pairs_perturbator_patch_tensor = torch.where(
        deviation > threshold, negative_pairs_perturbator, inject_data
        )
    # Sample 2: neg -> perturbator -> patch
    negative_pairs_perturbator_patch = negative_pairs_perturbator_patch_tensor.detach().cpu().numpy()
    
    # Sample 3: neg -> perturbator -> CUTS+
    negative_pairs_perturbator_cuts = positive_augmentor.forward(
        x=negative_pairs_without_patch,
        causality_matrix=causality_matrix,
    ).detach().cpu().numpy()

    negative_pairs_cuts_patch = negative_pairs_without_patch.clone()

    # Sample 4: neg -> perturbator -> patch -> CUTS+
    negative_pairs_perturbator_patch_cuts = positive_augmentor.forward(
        x=negative_pairs_perturbator_patch_tensor,
        causality_matrix=causality_matrix,
    ).detach().cpu().numpy()
    
    # Sample 5: neg -> perturbator -> CUTS+ -> patch
    inject_data = inject_data.detach().cpu().numpy()
    amplitude_d = np.max(inject_data, axis=1) - np.min(inject_data, axis=1) # (B, F)
    amplitude_pert_d = np.max(negative_pairs_perturbator_cuts, axis=1) - \
        np.min(negative_pairs_perturbator_cuts, axis=1) # (B, F)
    if args.deviation_mode == 'abs':
        deviation = np.abs(inject_data - negative_pairs_perturbator_cuts) # (B, T, F)
    elif args.deviation_mode == 'square':
        deviation = (inject_data - negative_pairs_perturbator_cuts) ** 2 # (B, T, F)
    else:
        raise ValueError("Deviation mode should be 'abs' or 'square'")
    threshold = np.where(
        amplitude_d != 0,
        args.non_constant_dim_tau * amplitude_d,
        args.constant_dim_tau * amplitude_pert_d
        ) # (B, F)
    threshold = np.tile(threshold[:, None, :], (1, args.window_size, 1)) # (B, T, F)
    negative_pairs_perturbator_cuts_patch = np.where(
        deviation > threshold, negative_pairs_perturbator_cuts, inject_data
        )
    
    # (4) t-SNE visualization
    os.makedirs('tsne/image', exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 16))
    tsne_dataset = [
        anomaly_data, negative_pairs_perturbator_numpy,
        negative_pairs_perturbator_cuts, negative_pairs_perturbator_patch,
        negative_pairs_perturbator_patch_cuts, negative_pairs_perturbator_cuts_patch
        ]
    tsne_label = [
        'True Anomaly', 'Perturbator', 'Perturbator + CUTS+',
        'Perturbator + Patching', 'Perturbator + Patching + CUTS+',
        'Perturbator + CUTS+ + Patching'
    ]
    
    tsne_dataset = np.concatenate(
        [data.reshape(data.shape[0], -1) for data in tsne_dataset]
        , axis=0
        )  # (6*B, T, F)
    pca_dim = int(min(args.pca_dim, tsne_dataset.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    tsne_dataset_pca = pca.fit_transform(tsne_dataset)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        initialization='pca',
        random_state=args.seed,
        n_jobs=1
    )
    tsne_embedding = tsne.fit(tsne_dataset_pca)
    tsne_embedding = np.asarray(tsne_embedding)[np.newaxis, :, :].reshape(6, len_data, 2)
    
    for i in range(6):
        ax.scatter(tsne_embedding[i, :, 0], tsne_embedding[i, :, 1], label=tsne_label[i])
    ax.legend()
    fig.suptitle(f't-SNE analysis for the {args.data} {args.sub_dataset}')
    plt.savefig(f'tsne/image/{args.data}_{args.sub_dataset}.png')