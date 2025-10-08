import torch
import numpy as np
from genias import VAE
from utils import patch, AnomalyInjection
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, required=True
)
parser.add_argument(
    '--sub-dataset', type=str, required=True
)
parser.add_argument(
    '--window-mode', default=False, type=bool,
    help='If true, then the only anomaly window is inputted' \
        'if false, then all dataset will be inputted.'
)
parser.add_argument(
    '--device', default='cuda:0', type=str,
)

args = parser.parse_args()
window_size = 200

# dataset_train = np.load(
#     os.path.join('data', args.dataset, 'train', args.sub_dataset + '.npy')
# )
dataset_test = np.load(
    os.path.join('data', args.dataset, 'test', args.sub_dataset + '.npy')
)
vae = VAE(
    window_size=window_size,
    data_dim=dataset_test.shape[1],
    latent_dim=100,
    depth=10,
).to(args.device)
ckpt = torch.load(
    os.path.join(
        'checkpoints/vae/', args.dataset, args.sub_dataset, 'epoch_1000.pt'
        ),
    map_location=args.device, weights_only=True
    )['model']
vae.load_state_dict(ckpt)

time_len = dataset_test.shape[0]

os.makedirs('image/t-SNE', exist_ok=True)
if args.window_mode == True:
    dataset_label = np.load(
        os.path.join('data', args.dataset, 'label', args.sub_dataset + '.npy')
    )
    anomaly_index = np.where(np.isclose(
        dataset_label, np.ones_like(dataset_label)
        ))[0]
    
    start_time = np.random.choice(anomaly_index)
    fig, ax = plt.subplots(figsize=(15, 16))
    tsne = TSNE()
    vector = dataset_test[start_time:start_time + window_size]
    # vector = dataset_train[start_time:start_time + window_size]
    negative_carla = AnomalyInjection().inject_anomaly(vector)
    vector = torch.from_numpy(vector).to(args.device).to(torch.float32)
    _, _, _, negative_genias = vae.forward(vector[None, ...])
    vector = vector.detach().cpu().numpy()
    negative_genias = negative_genias.detach().cpu().numpy().squeeze(0)
    negative_genias = patch(vector, negative_genias, 0.4)
    negative_genias = patch(vector, negative_genias, 0.4)
    # true = dataset_test[start_time:start_time + window_size]
    # vector_embedded = tsne.fit_transform(vector)
    negative_carla_embedded = tsne.fit_transform(negative_carla)
    negative_genias_embedded = tsne.fit_transform(negative_genias)
    true_embedded = tsne.fit_transform(vector)
    # ax.scatter(vector_embedded[:, 0], vector_embedded[:, 1], label='True dataset')
    ax.scatter(negative_carla_embedded[:, 0], negative_carla_embedded[:, 1], label='Anomaly injection (CARLA)')
    ax.scatter(negative_genias_embedded[:, 0], negative_genias_embedded[:, 1], label='Anomaly injection (GenIAS)')
    ax.scatter(true_embedded[:, 0], true_embedded[:, 1], label='Anomaly')
    ax.legend()
    fig.suptitle(f'GenIAS t-SNE analysis for the {args.dataset} {args.sub_dataset}')
    plt.savefig(f'image/t-SNE/{args.dataset}_{args.sub_dataset}_{start_time}.png')

else:
    NotImplementedError
# else:
#     fig, ax = plt.subplots(figsize=(21, 22))
#     negative_carla = np.empty_like(dataset_train)
#     negative_genias = np.empty_like(dataset_train)
#     tsne = TSNE()
#     for i in range(time_len // window_size):
#         start_time = i * window_size
#         vector = dataset_train[start_time:start_time + window_size]
#         negative_carla[start_time:start_time+window_size] = AnomalyInjection().inject_anomaly(vector)
#         vector = torch.from_numpy(vector).to(args.device).to(torch.float32)
#         _, _, _, negative_genias_window = vae.forward(vector[None, ...])
#         vector = vector.detach().cpu().numpy()
#         negative_genias_window = negative_genias_window.detach().cpu().numpy().squeeze(0)
#         negative_genias_window = patch(vector, negative_genias_window, 0.4)
#         negative_genias[start_time:start_time + window_size] = negative_genias_window
#     start_time = time_len - window_size
#     plot_start_time = (i + 1) * window_size - start_time
#     vector = dataset_train[start_time:start_time + window_size]
#     negative_carla[(i + 1) * window_size:start_time+window_size] = AnomalyInjection().inject_anomaly(vector)[plot_start_time:]
#     vector = torch.from_numpy(vector).to(args.device).to(torch.float32)
#     _, _, _, negative_genias_window = vae.forward(vector[None, ...])
#     vector = vector.detach().cpu().numpy()
#     negative_genias_window = negative_genias_window.detach().cpu().numpy().squeeze(0)
#     negative_genias_window = patch(vector, negative_genias_window, 0.4)
#     negative_genias[(i + 1) * window_size:] = negative_genias_window[plot_start_time:]
#     train_embedded = tsne.fit_transform(dataset_train)
#     negative_carla_embedded = tsne.fit_transform(negative_carla)
#     negative_genias_embedded = tsne.fit_transform(negative_genias)
#     true_embedded = tsne.fit_transform(dataset_test)
#     ax.scatter(train_embedded[:, 0], train_embedded[:, 1], label='True dataset')
#     ax.scatter(negative_carla_embedded[:, 0], negative_carla_embedded[:, 1], label='Anomaly injection (CARLA)')
#     ax.scatter(negative_genias_embedded[:, 0], negative_genias_embedded[:, 1], label='Anomaly injection (GenIAS)')
#     ax.scatter(true_embedded[:, 0], true_embedded[:, 1], label='Anomaly')
#     ax.legend()
#     fig.suptitle(f'GenIAS t-SNE analysis for the {args.dataset} {args.sub_dataset}')
#     plt.savefig(f'image/t-SNE/{args.dataset}_{args.sub_dataset}.png')