from exp import *


def main(
    time: str,
    task: str,
    data: str,
    subdata: Optional[str],
    model_mode: str,
    window_size: int,
    batch_size: int,
    gpu_num: int,
    positive_augmentor_time: str,
    perturbator_time: str,
) -> None:
    assert task in [
        'perturbation',
        'pretext',
        'calssification',
        'pretext_classification'
    ], \
    "'perturbation', 'pretext', 'classification', pretext_classification"

    assert model_mode in ['linear', 'tcn', 'bidirectional']

    if task == 'perturbation':
        logging.info(
            f'Training perturbator model on {data} {subdata}...\n'
        )
        # if model_mode == 'linear':
        #     trainer = LinearPerturbatorTrainer(
        #         time=time,
        #         data=data,
        #         subdata=subdata,
        #         batch_size=batch_size,
        #         window_size=window_size,
        #         delta_min=0.1,
        #         delta_max=0.2,
        #         prior_var=0.5,
        #         recon_loss_weight=1.0,
        #         perturbation_loss_weight=0.1,
        #         zero_perturbation_loss_weight=0.01,
        #         nonzero_perturbation_loss_bound=1.0,
        #         zero_perturbation_loss_bound=1.0,
        #         kld_loss_weight=0.1,
        #         gpu_num=gpu_num,
        #         learning_rate=1e-4,
        #         epochs=100,
        #         save_interval=5,
        #     )
        if model_mode == 'tcn':
            trainer = TCNPerturbatorTrainer(
                time=time,
                data=data,
                subdata=subdata,
                batch_size=batch_size,
                window_size=window_size,
                recon_pert_mse_delta_min=0.1,
                pert_mse_delta_min=0.2,
                prior_var=0.5,
                recon_loss_weight=1.0,
                pert_loss_weight=0.1,
                zero_pert_loss_weight=0.01,
                kld_loss_weight=0.1,
                discriminator_loss_weight=1.0,
                gpu_num=gpu_num,
                learning_rate=1e-4,
                epochs=50,
                save_interval=5,
            )
        elif model_mode == 'bidirectional':
            trainer = BidirectionalPerturbatorTrainer(
                time=time,
                data=data,
                subdata=subdata,
                batch_size=batch_size,
                window_size=window_size,
                recon_pert_mse_delta_min=0.1,
                pert_mse_delta_min=0.2,
                prior_var=0.5,
                sigma_pert_factor=2.0,
                recon_loss_weight=1.0,
                pert_loss_weight=0.1,
                zero_pert_loss_weight=0.01,
                kld_loss_weight=0.1,
                discriminator_loss_weight=5.0,
                gpu_num=gpu_num,
                learning_rate=1e-4,
                epochs=100,
                save_interval=5,
            )

        trainer.train()
        trainer.eval()

    elif task == 'pretext':
        logging.info(
            f'Training pretext model on {data} {subdata}...\n'
        )
        trainer = PretextTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            positive_augementor_time=positive_augmentor_time,
            perturbator_time=perturbator_time,
            epochs=30,
            batch_size=batch_size,
            learning_rate=1e-3,
            gpu_num=gpu_num,
            num_neighborhoods=5,
        )
        trainer.train()
        trainer.select_neighbors()

    elif task == 'classification':
        logging.info(
            f'Training classification model on {data} {subdata}...\n'
        )
        trainer = ClassificationTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            gpu_num=gpu_num,
            batch_size=batch_size,
            learning_rate=1e-2,
        )
        trainer.train()
        trainer.inference()
    
    elif task == 'pretext_classification':
        logging.info(
            f'Training classification model on {data} {subdata}...\n'
        )
        pretext_trainer = PretextTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            positive_augementor_time=positive_augmentor_time,
            perturbator_time=perturbator_time,
            epochs=30,
            batch_size=batch_size,
            learning_rate=1e-3,
            gpu_num=gpu_num,
            num_neighborhoods=5,
        )
        pretext_trainer.train()
        pretext_trainer.select_neighbors()

        classification_trainer = ClassificationTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            gpu_num=gpu_num,
            epochs=100,
            batch_size=batch_size,
            learning_rate=1e-2,
        )
        classification_trainer.train()
        classification_trainer.inference()

    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--exp-name',
        type=str,
        required=True,
        help="Experiment name."
    )
    args.add_argument(
        '--task',
        type=str,
        required=True,
        help="Task name."
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
        default=100,
        help="Length of window. Default 100.",
    )
    args.add_argument(
        '--model-mode',
        type=str,
        help="Model mode. Either 'linear' or 'tcn'."
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

    fix_seed_all(seed=config.seed)
    
    time = datetime.datetime.now()
    time = time.strftime('%m-%d-%H:%M')

    set_logging_file(
        exp_name=config.exp_name,
        log_file_path=os.path.join(
            'log', config.task, config.data, 'exp', f'{time}.log'
        ),
        time=time,
    )

    if config.data in ['MSL', 'SMAP', 'SMD']:
        data_dir = os.path.join('exp', 'data', config.data, 'train')
        data_list = sorted(os.listdir(data_dir))
        data_list = [subdata.replace('.npy', '') for subdata in data_list]

        for subdata in data_list:
            main(
                time=time,
                task=config.task,
                data=config.data,
                subdata=subdata,
                model_mode=config.model_mode,
                window_size=config.window_size,
                batch_size=config.batch_size,
                gpu_num=config.gpu_num,
                positive_augmentor_time='12-08-16:39',
                perturbator_time='12-08-20:23',
            )
    