from cuts_plus import *


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--exp-name',
        type=str,
        required=True,
        help="Experiment name."
    )
    args.add_argument(
        '--data',
        type=str,
        required=True,
        help="Name of dataset.",
    )
    args.add_argument(
        '--window-size',
        type=int,
        default=10,
        help="Length of window. Default 10.",
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="GPU number. Default 0."
    )
    args.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size. Default 100.'
    )
    args.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help="Initial learning rate. Default 1e-3."
    )
    args.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Training epochs. Default 100."
    )
    args.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help="Model checkpoint save interval. Default 5."
    )
    args.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed. Default 42.'
    )
    config = args.parse_args()

    set_seed(config.seed)
    
    time = datetime.datetime.now()
    time = time.strftime('%m-%d-%H:%M')

    set_logging_file(
        exp_name=config.exp_name,
        log_file_path=os.path.join(
            'log', 'cuts_plus', config.data, f'{time}.log'
        ),
        time=time,
        seed=config.seed,
    )

    if config.data in ['MSL', 'SMAP', 'SMD']:
        data_dir = os.path.join('cuts_plus', 'data', config.data, 'train')
        data_list = sorted(os.listdir(data_dir))
        data_list = [subdata.replace('.npy', '') for subdata in data_list]
    
        for subdata in data_list:

            logging.info(
                f'Training CUTS+ on {config.data} {subdata}...\n'
            )

            cuts_plus_trainer = CUTS_PLUS_Trainer(
                time=time,
                data=config.data,
                subdata=subdata,
                window_size=config.window_size,
                batch_size=config.batch_size,
            )
            cuts_plus_trainer.train()
