import math
from exp import *


def plot_data(
    data: str,
    subdata: Optional[str],
    downsample: bool,
    downsample_step: int,
) -> None:
    save_dir = os.path.join('plot_data', data)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if subdata is not None:
        if downsample:
            save_dir = os.path.join(
                save_dir, f'{subdata}_downsample_step_{downsample_step}.png'
            )
        else:
            save_dir = os.path.join(save_dir, f'{subdata}.png')
    
    else:
        if downsample:
            save_dir = os.path.join(
                save_dir, f'{data}_downsample_step_{downsample_step}.png'
            )
        else:
            save_dir = os.path.join(save_dir, f'{data}.png')
    
    if data in ['MSL', 'SMAP', 'SMD']:
        train_data_dir = os.path.join(
            'exp', 'data', data, 'train', f'{subdata}.npy'
        )
        train_data = np.load(train_data_dir)
    
    elif data == 'SWaT':
        train_data_dir = os.path.join('exp', 'data', data, 'swat_train2.csv')
        train_data = pd.read_csv(train_data_dir)
        train_data = train_data.iloc[:, :-1].to_numpy()
    
    elif data == 'WADI':
        train_data_dir = os.path.join(
            'exp', 'data', data, 'WADI_14days_new.csv'
        )
        train_data = pd.read_csv(train_data_dir)
        train_data = train_data.dropna(axis='columns', how='all').dropna()
        train_data = train_data.iloc[:, 3:].to_numpy()
    
    if downsample:
        train_data = downsample_data(
            data=train_data,
            downsample_step=downsample_step
        )
    
    data_dim = train_data.shape[-1]
    num_cols = 2
    num_rows = math.ceil(data_dim / num_cols)
    
    figsize = (25, 20 * num_cols)

    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    for i in range(0, data_dim):
        ax[i//num_cols, i%num_cols].plot(train_data[:, i])
    
    fig.savefig(save_dir)
    plt.close()

    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--data',
        type=str,
    )
    args.add_argument(
        '--downsample',
        type=str2bool,
        default=False,
    )
    args.add_argument(
        '--downsample-step',
        type=int,
        default=5,
    )
    config = args.parse_args()

    if config.data in ['MSL', 'SMAP', 'SMD']:
        subdata_list = os.listdir(
            os.path.join('exp', 'data', config.data, 'train')
        )
        subdata_list = sorted(subdata_list)
        subdata_list = [f.replace('.npy', '') for f in subdata_list]

        for subdata in subdata_list:
            plot_data(
                data=config.data,
                subdata=subdata,
                downsample=config.downsample,
                downsample_step=config.downsample_step,
            )

    elif config.data in ['SWaT', 'WADI']:
        plot_data(
            data=config.data,
            subdata=None,
            downsample=config.downsample,
            downsample_step=config.downsample_step,
        )
