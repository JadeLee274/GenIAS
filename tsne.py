from sklearn.decomposition import PCA
from openTSNE import TSNE
from exp import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Name of dataset.',
    )
    parser.add_argument(
        '--subdata',
        type=str,
        required=True,
        help="Name of subdata."    
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
        help="Window length. Default 10."
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
        help="Seed. Default 42."
    )
    parser.add_argument(
        '--perturbator-mode',
        type=str,
        choices=['plad', 'tcn_perturbator'],
        default='plad',
        help="Mode of perturbator. Default 'plad'.",
    )
    parser.add_argument(
        '--positive-augmentor-time',
        type='str',
        help="The timestamp (or directory) of pre-trained CUTS+.",
    )
    parser.add_argument(
        '--perturbator-time',
        type=str,
        help="The timestamp (or directory) of pre-trained perturbator."
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
        choices=['abs', 'square'],
        default='abs',
        help="The mode of deviation for patching. Default 'abs'.",
    )
    parser.add_argument(
        '--non-constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to non-constant dim.",
    )
    parser.add_argument(
        '--constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to constant dim.",
    )
    parser.add_argument(
        '--pca-dim',
        type=int,
        default=50,
        help="Dimension for PCA. Default 50." \
             "Note: data are flattened so the actual dimension is " \
                    "window_size * data_dim.",
    )
    parser.add_argument(
        '--anomaly-or-normal',
        type=str,
        choices=['anomaly', 'normal'],
        default='anomaly',
        help="Whether to use abnormal or normal data for injection." \
             "Default 'anomaly'.",
    )
    config = parser.parse_args()

    # [Step 1] Load dataset.
    dataset = PerturbationDataset(
        data=config.data,
        subdata=config.subdata,
        mode='test',
        window_size=config.window_size,
        downsample=config.downsample,
        downsample_step=config.downsample_step,
    )
    anomaly_data_indices = np.where(dataset.test_labels==1)[0]
    len_data = min(len(anomaly_data_indices), config.num_sample)
    
    if len(anomaly_data_indices) < config.num_sample:
        print(
            'The number of anomalies in test data is less than # samples.' \
            'All anomalies are uesd for t-SNE visualization.'
        )
    else:
        anomaly_data_indices = np.random.choice(
            a=anomaly_data_indices,
            size=config.num_sample,
            replace=False,
        )
    
    normal_data = dataset.train_data[
        np.random.choice(
            a=len(dataset.train_data),
            size=len_data,
            replace=False,
        )
    ]
    anomaly_data = dataset.test_data[anomaly_data_indices]
    data_dim = anomaly_data.shape[-1]

    # [Step 2] Load pretrained models.
    prediction_step = 1
    noise_injection_step = config.window_size - prediction_step

    processor = torch.device(f'cuda:{config.gpu_num}')
    
    positive_augmentor = PositiveAugmentor(
        noise_injection_step=noise_injection_step,
        prediction_step=prediction_step,
        noise_level=0.1,
        n_nodes=data_dim,
    )

    if config.perturbator_mode == 'plad':
        perturbator = PLAD(
            window_size=config.window_size,
            data_dim=data_dim,
            latent_dim=100 if data_dim > 1 else 50,
        )
    elif config.perturbator_mode == 'tcn_perturbator':
        perturbator = TCNPerturbator(
            window_size=config.window_size,
            data_dim=data_dim,
            latent_perturbator_factor=2.0,
        )
    
    positive_augmentor.init_causal_discoverer(
        time=config.positive_augmentor_time,
        data=config.data,
        subdata=config.subdata,
        seed=config.seed,
    )

    if config.perturbator_mode == 'plad':
        perturbator.init_plad(
            time=config.perturbator_time,
            data=config.data,
            subdata=config.subdata,
        )
    elif config.perturbator_mode == 'tcn_perturbator':
        perturbator.init_perturbator(
            time=config.perturbator_time,
            data=config.data,
            subdata=config.subdata,
        )
    
    positive_augmentor = positive_augmentor.to(processor)
    causality_matrix = positive_augmentor.causal_discoverer.causality_mtx
    positive_augmentor.eval()
    perturbator = perturbator.to(processor)
    perturbator.eval()

    # [Step 3] Inject anomalies.
    if config.anomaly_or_normal == 'anomaly':
        inject_data = torch.from_numpy(anomaly_data).to(processor)
    else:
        inject_data = torch.from_numpy(normal_data).to(processor)
    
    if config.perturbator_mode == 'plad':
        negative_pairs_perturbator, _ = perturbator.forward(x=inject_data)
    elif config.perturbator_mode == 'tcn_perturbator':
        _, negative_pairs_perturbator, _, _ = perturbator.forward(x=inject_data)
    
    negative_pairs_without_patch = negative_pairs_perturbator.clone()

    # Sample 1: negative -> perturbator
    negative_pairs_perturbator_numpy = negative_pairs_without_patch.detach().cpu().numpy()

    amplitude_d = torch.max(inject_data, dim=1)[0] - torch.min(inject_data, dim=1)[0]
    amplitude_pert_d = torch.max(negative_pairs_perturbator, dim=1)[0] \
                       - torch.min(negative_pairs_perturbator, dim=1)[0]

    if config.deviation_mode == 'abs':
        deviation = torch.abs(inject_data - negative_pairs_perturbator)
    elif config.deviation_mode == 'square':
        deviation = (inject_data - negative_pairs_perturbator) ** 2
    else:
        raise ValueError("Deviation mode should be 'abs' or 'square'.")
    
    threshold = torch.where(
        amplitude_d != 0,
        config.non_constant_dim_tau * amplitude_d,
        config.constant_dim_tau * amplitude_pert_d,
    )

    threshold = threshold[:, None, :].expand(-1, config.window_size, -1)

    negative_pairs_perturbator_patch_tensor = torch.where(
        deviation > threshold, negative_pairs_perturbator, inject_data,
    )

    # Sample 2: negative -> perturbator -> patch
    negative_pairs_perturbator_patch \
    = negative_pairs_perturbator_patch_tensor.detach().cpu().numpy()

    # Sample 3: negative -> perturbator -> CUTS+
    negative_pairs_perturbator_cuts = positive_augmentor.forward(
        x=negative_pairs_without_patch,
        causality_matrix=causality_matrix,
    ).detach().cpu().numpy()

    negative_pairs_cuts_patch = negative_pairs_without_patch.clone()

    # Sample 4: negative -> perturbator -> patch -> CUTS+
    negative_pairs_perturbator_patch_cuts = positive_augmentor.forward(
        x=negative_pairs_perturbator_patch_tensor,
        causality_matrix=causality_matrix,
    ).detach().cpu().numpy()

    # Sample 5: negative -> perturbator -> CUTS+ -> patch
    inject_data = inject_data.detach().cpu().numpy()
    amplitude_d = np.max(inject_data, axis=1) - np.min(inject_data, axis=1)
    amplitude_pert = \
        np.max(negative_pairs_perturbator_cuts, axis=1) \
        - np.min(negative_pairs_perturbator_cuts, axis=1)
    
    if config.deviation_mode == 'abs':
        deviation = np.abs(inject_data - negative_pairs_perturbator_cuts)
    elif config.deviation_mode == 'square':
        deviation = (inject_data - negative_pairs_perturbator_cuts) ** 2
    else:
        raise ValueError("Deviation mode should be 'abs' or 'square'.")
    
    threshold = np.where(
        amplitude_d != 0,
        config.non_constant_dim_tau * amplitude_d,
        config.constant_dim_tau * amplitude_pert_d,
    )
    threshold = np.tile(threshold[:, None, :], (1, config.window_size, 1))
    negative_pairs_perturbator_cuts_patch = np.where(
        deviation > threshold, negative_pairs_perturbator_cuts, inject_data,
    )

    # [Step 4] t-SNE visualization
    os.makedirs(os.path.join('tsne', 'image'), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 16))
    tsne_dataset = [
        anomaly_data,
        negative_pairs_perturbator_numpy,
        negative_pairs_perturbator_cuts,
        negative_pairs_perturbator_patch,
        negative_pairs_perturbator_patch_cuts,
        negative_pairs_perturbator_cuts_patch,
    ]

    tsne_labels = [
        'True Anomaly',
        'Perturbator',
        'Perturbator + CUTS+',
        'Perturbator + Patching',
        'Perturbator + Patching + CUTS+',
        'Perturbator + CUTS+ + Patching',
    ]

    tsne_dtaset = np.concatenate(
        [data.reshape(data.shape[0], -1) for data in tsne_dataset],
        axis=0,
    )

    pca_dim = int(min(config.pca_dim, tsne_dataset.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=config.seed)
    tsne_dataset_pca = pca.fit_transform(tsne_dataset)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        initialization='pca',
        random_state=config.seed,
        n_jobs=1,
    )
    tsne_embedding = tsne.fit(tsne_dataset_pca)
    tsne_embedding = np.asarray(
        tsne_embedding[np.newaxis, :, :]
    ).reshape(6, len_data, 2)

    for i in range(6):
        ax.scatter(
            tsne_embedding[i, :, 0],
            tsne_embedding[i, :, 1],
            label=tsne_labels[i],
        )
    
    ax.legend()
    fig.suptitle(f't-SNE analysis for the {config.data} {config.subdata}')
    plt.savefig(
        os.path.join('tsne', 'image', f'{config.data}_{config.subdata}.png')
    )
