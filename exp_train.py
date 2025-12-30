from exp import *


def main(
    time: str,
    task: str,
    data: str,
    subdata: Optional[str],
    window_size: int,
    batch_size: int,
    gpu_num: int,
    epochs: int,
    perturbator_mode: str,
    positive_augmentor_time: Optional[str],
    perturbator_time: Optional[str],
    downsample: bool,
    downsample_step: int,
    seed: int,
    apply_patch: bool,
    patch_after: str,
    deviation_mode: str,
    non_constant_dim_tau: float,
    constant_dim_tau: float,
) -> None:
    assert task in [
        'cuts_plus',
        'plad',
        'perturbation',
        'pretext_classification',
        'all'
    ], "'cuts_plus', 'perturbator', 'pretext_classification', 'all'"

    if task == 'cuts_plus':
        trainer = CUTSplusTrainer(
            time=time,
            data=data,
            subdata=subdata,
            downsample=downsample,
            downsample_step=downsample_step,
            batch_size=batch_size,
            window_size=window_size,
            seed=seed,
        )
        if subdata in ['C-1', 'A-1', 'machine-1-1'] or data in ['SWaT', 'WADI']:
            logging.info('Experiment setup:')
            logging.info(f'- Batch size: {trainer.batch_size}')
            logging.info(f'- Window size: {trainer.window_size}')
            logging.info(f'- Downsample: {trainer.downsample}')

            if downsample:
                logging.info(f'- Downsamle step: {trainer.downsample_step}')

            logging.info(f'- Data prediction: {trainer.predict_data}')
            logging.info(f'- Graph discovery: {trainer.discover_graph}')
            logging.info(f'- Prediction start lr: {trainer.pred_start_lr}')
            logging.info(f'- Prediction end lr: {trainer.pred_end_lr}')
            logging.info(f'- Gumbel start tau: {trainer.gumbel_start_tau}')
            logging.info(f'- Gumbel end tau: {trainer.gumbel_end_tau}')
            logging.info(f'- Graph start lambda: {trainer.graph_start_lambda}')
            logging.info(f'- Graph end lambda: {trainer.graph_end_lambda}')
            logging.info(f'- Graph start lr: {trainer.graph_start_lr}')
            logging.info(f'- Graph end lr: {trainer.graph_end_lr}')
            logging.info(f'- Seed: {trainer.seed}\n')
        
        if data in ['MSL', 'SMAP', 'SMD']:
            logging.info(f'Training CUTS+ model on {data} {subdata}...\n')
        elif data in ['SWaT', 'WADI']:
            logging.info(f'Training CUTS+ model in {data}...\n')
        
        trainer.train()

        return
    
    elif task == 'plad':
        if data in  ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'Training perturbator model on {data} {subdata}...\n'
            )
        elif data in ['SWaT', 'WADI']:
            logging.info(f'Training perturbator model in {data}...\n')
        
        trainer = PLADTrainer(
            time=time,
            data=data,
            subdata=subdata,
            downsample=downsample,
            downsample_step=downsample_step,
            batch_size=batch_size,
            window_size=window_size,
            epochs=epochs,
            seed=seed,
        )
        if subdata in ['C-1', 'A-1', 'machine-1-1'] or data in ['SWaT', 'WADI']:
            logging.info('Experiment setup:')
            logging.info(f'- Batch size: {trainer.batch_size}')
            logging.info(f'- Window size: {trainer.window_size}')
            logging.info(f'- Downsample: {trainer.downsample}')

            if downsample:
                logging.info(f'- Downsample step: {trainer.downsample_step}')

            logging.info(f'- Learning rate: {trainer.optim_learning_rate}')
            logging.info(f'- Weight decay: {trainer.optim_weight_decay}')
            logging.info(f'- Train epochs: {trainer.epochs}')
            logging.info(f'- Seed: {trainer.seed}\n')
        
        trainer.train()
        trainer.eval()

        return
        
    elif task == 'perturbation':
        if data in  ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'Training perturbator model on {data} {subdata}...\n'
            )
        elif data in ['SWaT', 'WADI']:
            logging.info(f'Training perturbator model in {data}...\n')
        
        trainer = TCNPerturbatorTrainer(
            time=time,
            data=data,
            subdata=subdata,
            batch_size=batch_size,
            window_size=window_size,
            downsample=downsample,
            downsample_step=downsample_step,
            seed=seed,
            epochs=epochs,
        )
        
        if subdata in ['C-1', 'A-1', 'machine-1-1'] or data in ['SWaT', 'WADI']:
            logging.info('Experiment setup:')
            logging.info(f'- Batch size: {trainer.batch_size}')
            logging.info(f'- Window size: {trainer.window_size}')
            logging.info(f'- Downsample: {trainer.downsample}')

            if downsample:
                logging.info(f'- Downsample step: {trainer.downsample_step}')
            
            logging.info(
                f'- Recon delta min: {trainer.recon_pert_mse_delta_min}'
            )
            logging.info(f'- Pert delta min: {trainer.pert_mse_delta_min}')
            logging.info(f'- Prior var: {trainer.prior_var}')
            logging.info(f'- Recon loss weight: {trainer.recon_loss_weight}')
            logging.info(
                f'- Zero pert loss weight: {trainer.zero_pert_loss_weight}'
            )
            logging.info(f'- KLD loss weight: {trainer.kld_loss_weight}')
            logging.info(
                f'- Discriminator loss weight: {trainer.discriminator_loss_weight}'
            )
            logging.info(f'- Learning rate: {trainer.learning_rate}')
            logging.info(f'- Epochs: {trainer.epochs}')
            logging.info(f'- Seed: {seed}\n')

        trainer.train()
        trainer.eval()

        return

    elif task == 'pretext_classification':
        if data in  ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'Training classifier model on {data} {subdata}...\n'
            )
        elif data in ['SWaT', 'WADI']:
            logging.info(f'Training classifier model in {data}...\n')
        
        if subdata in ['C-1', 'A-1', 'machine-1-1'] or data in ['SWaT', 'WADI']:
            logging.info('Experiment setup:')
            logging.info(f'- Data: {data}')
            logging.info(
                f'- Positive augmentor timestamp: {positive_augmentor_time}')
            logging.info(f'- Perturbator timestamp: {perturbator_time}')
            logging.info(f'- Downsample: {downsample}')

            if downsample:
                logging.info(f'- Downsample step: {downsample_step}')
            
            logging.info(f'- Apply patch: {apply_patch}')

            if apply_patch:
                logging.info(f'- Patch after: {patch_after}')
                logging.info(f'- Deviation mode: {deviation_mode}')
                logging.info(f'- Non-constant dim tau: {non_constant_dim_tau}')
                logging.info(f'- Constant dim tau: {constant_dim_tau}')
                
            logging.info(f'- Seed: {seed}\n')
        
        pretext_trainer = PretextTrainer(
            time=time,
            data=data,
            subdata=subdata,
            seed=seed,
            window_size=window_size,
            perturbator_mode=perturbator_mode,
            positive_augementor_time=positive_augmentor_time,
            perturbator_time=perturbator_time,
            downsample=downsample,
            downsample_step=downsample_step,
            epochs=30,
            batch_size=batch_size,
            learning_rate=1e-3,
            gpu_num=gpu_num,
            num_neighborhoods=5,
            apply_patch=apply_patch,
            patch_after=patch_after,
            deviation_mode=deviation_mode,
            non_constant_dim_tau=non_constant_dim_tau,
            constant_dim_tau=constant_dim_tau,
        )
        
        pretext_trainer.train()
        pretext_trainer.select_neighbors()

        classification_trainer = ClassificationTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            downsample=downsample,
            downsample_step=downsample_step,
            gpu_num=gpu_num,
            epochs=100,
            batch_size=batch_size,
            learning_rate=1e-2,
        )
        classification_trainer.train()
        f1, tp, fp, fn, aucpr = classification_trainer.inference()

        return f1, tp, fp, fn, aucpr
    
    elif task == 'all':
        if subdata in ['C-1', 'A-1', 'machine-1-1'] or data in ['SWaT', 'WADI']:
            logging.info('Experiment setup:')
            logging.info(f'- Seed: {seed}')
            logging.info(f'- Downsample: {downsample}')

            if downsample:
                logging.info(f'- Downsample step: {downsample_step}\n')
        
        cuts_plus_trainer = CUTSplusTrainer(
            time=time,
            data=data,
            subdata=subdata,
            downsample=downsample,
            downsample_step=downsample_step,
            batch_size=100,
            window_size=window_size,
            seed=seed,
            gpu_num=gpu_num,
        )
        cuts_plus_trainer.train()

        perturbation_trainer = TCNPerturbatorTrainer(
            time=time,
            data=data,
            subdata=subdata,
            batch_size=256,
            window_size=window_size,
            downsample=downsample,
            downsample_step=downsample_step,
            seed=seed,
            gpu_num=gpu_num,
        )
        perturbation_trainer.train()
        perturbation_trainer.eval()

        pretext_trainer = PretextTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            positive_augementor_time=time,
            perturbator_time=time,
            downsample=downsample,
            downsample_step=downsample_step,
            epochs=30,
            batch_size=256,
            learning_rate=1e-3,
            gpu_num=gpu_num,
            num_neighborhoods=5,
            apply_patch=apply_patch,
            patch_after=patch_after,
            deviation_mode=deviation_mode,
            non_constant_dim_tau=non_constant_dim_tau,
            constant_dim_tau=constant_dim_tau,
        )
        pretext_trainer.train()
        pretext_trainer.select_neighbors()

        classification_trainer = ClassificationTrainer(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            downsample=downsample,
            downsample_step=downsample_step,
            gpu_num=gpu_num,
            epochs=100,
            batch_size=256,
            learning_rate=1e-2,
        )
        classification_trainer.train()
        f1, tp, fp, fn, aucpr = classification_trainer.inference()

        return f1, tp, fp, fn, aucpr


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
        '--subdata',
        type=str,
        help="Name of subdata. Only for MSL, SMAP, SMD when you want to run " \
             "this script on specific subdata. If it is None, then this script" \
             "runs on the entire subset."
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
        '--epochs',
        type=int,
        help="Train epochs."
    )
    args.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed. Default 42.'
    )
    args.add_argument(
        '--perturbator-mode',
        type=str,
        help="The mode of perturbator. Either 'plad' or 'tcn_perturbator'",
    )
    args.add_argument(
        '--positive-augmentor-time',
        type=str,
        help="The timestamp of pre-trained CUTS+.",
    )
    args.add_argument(
        '--perturbator-time',
        type=str,
        help="The timestamp of pre-trained perturbator.",
    )
    args.add_argument(
        '--downsample',
        type=str2bool,
        default=False,
        help="Whether to downsample data or not. Default False."
    )
    args.add_argument(
        '--downsample-step',
        type=int,
        default=5,
        help="Step size of downsampling. Default 5.",
    )
    args.add_argument(
        '--apply-patch',
        type=str2bool,
        default=True,
        help="Apply patching algorithm for pretext. Default True.",
    )
    args.add_argument(
        '--patch-after',
        type=str,
        default='positive_augmentor',
        help="Patch will be applied after perturbator or positive_augmentor."
    )
    args.add_argument(
        '--deviation-mode',
        type=str,
        default='abs',
        help="The mode of deviation for patching. Default 'abs'."
    )
    args.add_argument(
        '--non-constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to non-constant dim.",
    )
    args.add_argument(
        '--constant-dim-tau',
        type=float,
        help="Determines threshold when applying patch to constant dim."
    )
    config = args.parse_args()

    assert config.task in [
        'cuts_plus',
        'plad',
        'perturbation',
        'pretext_classification',
    ], "'plad', 'cuts_plus', 'perturbation', 'pretext_classification'"

    assert config.data in ['MSL', 'SMAP', 'SMD', 'SWaT', 'WADI'], \
    "'MSL', 'SMAP', 'SMD', 'SWaT', 'WADI'"

    if config.perturbator_mode is not None:
        assert config.perturbator_mode in ['plad', 'tcn_perturbator']

    if config.task == 'cuts_plus':
        batch_size = 100
    else:
        batch_size = 256
        if config.task == 'pretext_classification':
            assert config.positive_augmentor_time is not None, \
                   "positive augmentor timestamp required."
            assert config.perturbator_time is not None, \
                   "perturbator timestamp required."
            assert config.non_constant_dim_tau is not None, \
                   "tau for non-constant-valued dimension needed."
            assert config.constant_dim_tau is not None, \
                   "tau for constant-valued dimension needed."
    
    fix_seed_all(seed=config.seed)
    
    time = datetime.datetime.now()
    time = time.strftime('%m%d_%H%M')

    if config.task == 'cuts_plus':
        set_logging_file(
            exp_name=config.exp_name,
            log_file_path=os.path.join(
                'log', config.task, config.data, f'{time}.log'
            ),
            time=time,
            seed=config.seed,
        )
    else:
        set_logging_file(
            exp_name=config.exp_name,
            log_file_path=os.path.join(
                'log', config.task, config.data, 'exp', f'{time}.log'
            ),
            time=time,
            seed=config.seed,
        )

    if config.data in ['MSL', 'SMAP', 'SMD']:
        if config.subdata is None:
            data_dir = os.path.join('exp', 'data', config.data, 'train')
            data_list = sorted(os.listdir(data_dir))
            data_list = [subdata.replace('.npy', '') for subdata in data_list]

            f1_list = []
            tp_list = []
            fp_list = []
            fn_list = []
            aucpr_list = []
            
            if config.task == 'pretext_classification':
                for subdata in data_list:
                    f1, tp, fp, fn, aucpr = main(
                        time=time,
                        task=config.task,
                        data=config.data,
                        subdata=subdata,
                        window_size=config.window_size,
                        batch_size=batch_size,
                        gpu_num=config.gpu_num,
                        epochs=config.epochs,
                        perturbator_mode=config.perturbator_mode,
                        positive_augmentor_time=config.positive_augmentor_time,
                        perturbator_time=config.perturbator_time,
                        downsample=config.downsample,
                        downsample_step=config.downsample_step,
                        seed=config.seed,
                        apply_patch=config.apply_patch,
                        patch_after=config.patch_after,
                        deviation_mode=config.deviation_mode,
                        non_constant_dim_tau=config.non_constant_dim_tau,
                        constant_dim_tau=config.constant_dim_tau,
                    )
                    f1_list.append(f1)
                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)
                    aucpr_list.append(aucpr)
                
                f1_list = np.array(f1_list)
                tp_list = np.array(tp_list)
                fp_list = np.array(fp_list)
                fn_list = np.array(f1_list)
                aucpr_list = np.array(aucpr_list)

                best_f1 = np.max(f1_list)
                precision, recall, f1_micro = mirco_f1(
                    tp_list=tp_list,
                    fp_list=fp_list,
                    fn_list=fn_list,
                )
                aucpr_mean = np.mean(aucpr_list)
                aucpr_std = np.std(aucpr_list)
                f1_macro = macro_f1(f1_list=f1_list)

                logging.info('Scores')
                logging.info(f'- Best F1: {round(best_f1, 4)}')
                logging.info(f'- Micro F1: {round(f1_micro, 4)}')
                logging.info(f'- Precision: {round(precision, 4)}')
                logging.info(f'- Recall: {round(recall, 4)}')
                logging.info(f'- AUC-PR mean: {round(aucpr_mean, 4)}')
                logging.info(f'- AUC-PR std: {round(aucpr_std, 4)}')
                logging.info(f'- Macro F1: {round(f1_macro, 4)}')
    
            else:
                for subdata in data_list:
                    main(
                        time=time,
                        task=config.task,
                        data=config.data,
                        subdata=subdata,
                        window_size=config.window_size,
                        batch_size=batch_size,
                        gpu_num=config.gpu_num,
                        epochs=config.epochs,
                        perturbator_mode=config.perturbator_mode,
                        positive_augmentor_time=config.positive_augmentor_time,
                        perturbator_time=config.perturbator_time,
                        downsample=config.downsample,
                        downsample_step=config.downsample_step,
                        seed=config.seed,
                        apply_patch=config.apply_patch,
                        patch_after=config.patch_after,
                        deviation_mode=config.deviation_mode,
                        non_constant_dim_tau=config.non_constant_dim_tau,
                        constant_dim_tau=config.constant_dim_tau,
                    )

        else:
            if config.task == 'pretext_classification':
                f1, tp, fp, fn, aucpr = main(
                    time=time,
                    task=config.task,
                    data=config.data,
                    subdata=config.subdata,
                    window_size=config.window_size,
                    batch_size=batch_size,
                    gpu_num=config.gpu_num,
                    epochs=config.epochs,
                    perturbator_mode=config.perturbator_mode,
                    positive_augmentor_time=config.positive_augmentor_time,
                    perturbator_time=config.perturbator_time,
                    downsample=config.downsample,
                    downsample_step=config.downsample_step,
                    seed=config.seed,
                    apply_patch=config.apply_patch,
                    patch_after=config.patch_after,
                    deviation_mode=config.deviation_mode,
                    non_constant_dim_tau=config.non_constant_dim_tau,
                    constant_dim_tau=config.constant_dim_tau,
                )
                precision = tp / (tp + fp)
                recall = tp / (tp / fn)

                logging.info('Scores:')
                logging.info(f'- F1: {round(f1, 4)}')
                logging.info(f'- Precision: {round(precision, 4)}')
                logging.info(f'- Recall: {round(recall, 4)}')
                logging.info(f'- AUC-PR: {round(aucpr, 4)}')
                logging.info(f'- True positives: {round(tp, 4)}')
                logging.info(f'- False positives: {round(fp, 4)}')
                logging.info(f'- False negatives: {round(fn, 4)}')

            else:
                main(
                    time=time,
                    task=config.task,
                    data=config.data,
                    subdata=config.subdata,
                    window_size=config.window_size,
                    batch_size=batch_size,
                    gpu_num=config.gpu_num,
                    epochs=config.epochs,
                    perturbator_mode=config.perturbator_mode,
                    positive_augmentor_time=config.positive_augmentor_time,
                    perturbator_time=config.perturbator_time,
                    downsample=config.downsample,
                    downsample_step=config.downsample_step,
                    seed=config.seed,
                    apply_patch=config.apply_patch,
                    patch_after=config.patch_after,
                    deviation_mode=config.deviation_mode,
                    non_constant_dim_tau=config.non_constant_dim_tau,
                    constant_dim_tau=config.constant_dim_tau,
                )
            
    elif config.data in ['SWaT', 'WADI']:
        if config.task == 'pretext_classification':
            f1, tp, fp, fn, aucpr = main(
                time=time,
                task=config.task,
                data=config.data,
                subdata=None,
                window_size=config.window_size,
                batch_size=batch_size,
                gpu_num=config.gpu_num,
                epochs=config.epochs,
                perturbator_mode=config.perturbator_mode,
                positive_augmentor_time=config.positive_augmentor_time,
                perturbator_time=config.pertrbator_time,
                downsample=config.downsample,
                downsample_step=config.downsample_ste,
                seed=config.seed,
                apply_patch=config.apply_patch,
                patch_after=config.patch_after,
                deviation_mode=config.deviation_mode,
                non_constant_dim_tau=config.non_constant_dim_tau,
                constant_dim_tau=config.constant_dim_tau,
            )

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            logging.info('Scores:')
            logging.info(f'- F1: {round(f1, 4)}')
            logging.info(f'- Precision: {round(precision, 4)}')
            logging.info(f'- Recall: {round(recall, 4)}')
            logging.info(f'- AUC-PR: {round(aucpr, 4)}')
            logging.info(f'- True positives: {round(tp, 4)}')
            logging.info(f'- False positives: {round(fp, 4)}')
            logging.info(f'- False negatives: {round(fn, 4)}')
        
        else:
            main(
                time=time,
                task=config.task,
                data=config.data,
                subdata=None,
                window_size=config.window_size,
                batch_size=batch_size,
                gpu_num=config.gpu_num,
                epochs=config.epochs,
                perturbator_mode=config.perturbator_mode,
                positive_augmentor_time=config.positive_augmentor_time,
                perturbator_time=config.perturbator_time,
                downsample=config.downsample,
                downsample_step=config.downsample_step,
                seed=config.seed,
                apply_patch=config.apply_patch,
                patch_after=config.patch_after,
                deviation_mode=config.deviation_mode,
                non_constant_dim_tau=config.non_constant_dim_tau,
                constant_dim_tau=config.constant_dim_tau,
            )
