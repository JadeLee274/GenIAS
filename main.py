import argparse
from datetime import datetime
from genias import *


def main(
    exp_name: str,
    dataset: str,
    task: str,
    start_subdata: Optional[str] = None,
    pretext_timestamp: Optional[str] = None,
    pretext_scheme: str = 'carla',
    inject_different_anomalies: bool = False,
    use_pretrained_vae: bool = True,
    mix_step: int = 50,
    cut_negative_pairs: bool = True,
    vae_depth: int = 10,
    vae_prior_var: float = 0.5,
    vae_recon_weight: float = 1.0,
    vae_pert_weight: float = 0.1,
    vae_zero_pert_weight: float = 0.01,
    vae_kld_weight: float = 0.1,
    batch_size: int = 50,
    gpu_num: int = 0,
    seed: int = 42,
) -> None:
    """
    Training code.

    Parameters:
        exp_name: 
    """
    assert task in [
        'pretext',
        'classification',
        'pretext_classification',
        'vae_train',    
    ], "'pretext', 'classification', 'pretext_classification', 'vae_train'"
    
    if task != 'vae_train':
        assert pretext_scheme in [
            'carla',
            'carla_modified',
            'genias',
            'mix',
            'genias_multiple'
        ], "'carla', 'carla_modified', 'genias', 'mix', 'genias_multiple'"

    fix_seed_all(seed=seed)

    if task == 'classification':
        if not use_pretrained_vae:
            now = datetime.now()
            timestamp = now.strftime("%m%d_%H%M")
        else:
            timestamp = pretext_timestamp
    else:
        now = datetime.now()
        timestamp = now.strftime("%m%d_%H%M")
        
    log_dir = os.path.join('log', task, dataset)

    if task != 'vae_train':
        log_dir = os.path.join(log_dir, pretext_scheme)

    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, f'{timestamp}.log')

    set_logging_filehandler(log_file_path=log_file_path)

    logging.info(f'Experiment: {exp_name.replace('_', ' ')}\n')
    logging.info(f'Settings:')
    logging.info(f'- Task: {task}')
    logging.info(f'- Date: {timestamp.replace('_', ' ')}')
    logging.info(f'- Dataset: {dataset}')

    if start_subdata is not None:
        logging.info(f'- Starts from {start_subdata}')

    if task == 'classification':
        logging.info(f'- Timestamp of pretext used: {timestamp}')

    if task != 'vae_train':
        logging.info(f'- Pretext scheme: {pretext_scheme}')

    if pretext_scheme == 'mix':
        logging.info(f'- Mix step: {mix_step}')
    
    elif pretext_scheme == 'multiple_genias':
        logging.info(
            f'- Cut negative pairs: {cut_negative_pairs}')

    logging.info(f'- GPU number: {gpu_num}')
    logging.info(f'- Seed: {seed}\n')

    best_f1_list = []
    best_tp_list = []
    best_fp_list = []
    best_fn_list = []
    auc_pr_list = []

    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        data_dir = os.path.join(
            'genias', 'data', 'dataset', dataset, 'train'
        )
        data_list = sorted(os.listdir(data_dir))
        data_list = [data.replace('.npy', '') for data in data_list]

        if start_subdata is not None:
            start_idx = data_list.index(start_subdata)
            data_list = data_list[start_idx:]

        for subdata in data_list:
            if task == 'pretext':
                pretext(
                    dataset=dataset,
                    timestamp=timestamp,
                    subdata=subdata,
                    scheme=pretext_scheme,
                    inject_different_anomalies=inject_different_anomalies,
                    use_pretrained_vae=use_pretrained_vae,
                    mix_step=mix_step,
                    batch_size=batch_size,
                    gpu_num=gpu_num,
                    cut_negative_pairs=cut_negative_pairs,
                )

            elif task == 'classification':
                best_f1_score, best_tp, best_fp, best_fn, auc_pr = \
                    classification(
                        dataset=dataset,
                        timestamp=timestamp,
                        subdata=subdata,
                        scheme=pretext_scheme,
                        gpu_num=gpu_num,
                        batch_size=batch_size,
                    )
                logging.info(f'- True Positives: {best_tp}')
                logging.info(f'- False Positives: {best_fp}')
                logging.info(f'- False Negatives: {best_fn}\n')

            elif task == 'pretext_classification':
                pretext(
                    dataset=dataset,
                    timestamp=timestamp,
                    subdata=subdata,
                    scheme=pretext_scheme,
                    inject_different_anomalies=inject_different_anomalies,
                    use_pretrained_vae=use_pretrained_vae,
                    mix_step=mix_step,
                    batch_size=batch_size,
                    gpu_num=gpu_num,
                    cut_negative_pairs=cut_negative_pairs,
                )
                best_f1_score, best_tp, best_fp, best_fn, auc_pr = \
                    classification(
                        dataset=dataset,
                        timestamp=timestamp,
                        subdata=subdata,
                        scheme=pretext_scheme,
                        gpu_num=gpu_num,
                        batch_size=batch_size,
                    )
                best_f1_list.append(best_f1_score)
                best_tp_list.append(best_tp)
                best_fp_list.append(best_fp)
                best_fn_list.append(best_fn)
                auc_pr_list.append(auc_pr)

                logging.info(f'- True Positives: {best_tp}')
                logging.info(f'- False Positives: {best_fp}')
                logging.info(f'- False Negatives: {best_fn}\n')
        
            elif task == 'vae_train':
                vae_train(
                    dataset=dataset,
                    timestamp=timestamp,
                    subdata=subdata,
                    depth=vae_depth,
                    gpu_num=gpu_num,
                    prior_var=vae_prior_var,
                    recon_weight=vae_recon_weight,
                    pert_weight=vae_pert_weight,
                    zero_pert_weight=vae_zero_pert_weight,
                    kld_weight=vae_kld_weight,
                )

        if task in ['classification', 'pretext_classification']:
            best_f1_list = np.array(best_f1_list)
            best_tp_list = np.array(best_tp_list)
            best_fp_list = np.array(best_fp_list)
            best_fn_list = np.array(best_fn_list)
            auc_pr_list = np.array(auc_pr_list)

            f1_score_best = np.max(best_f1_list)
            precision, recall, f1_micro = mirco_f1(
                tp_list=best_tp_list,
                fp_list=best_fp_list,
                fn_list=best_fn_list
            )
            auc_pr_mean = np.mean(auc_pr_list)
            auc_pr_std = np.std(auc_pr_list)
            f1_macro = macro_f1(f1_list=best_f1_list)

            logging.info('Scores')
            logging.info(f'- Best F1: {round(f1_score_best, 4)}')
            logging.info(f'- Micro F1: {round(f1_micro, 4)}')
            logging.info(f'- Precision: {round(precision, 4)}')
            logging.info(f'- Recall: {round(recall, 4)}')
            logging.info(f'- AUC-PR mean: {round(auc_pr_mean, 4)}')
            logging.info(f'- AUC-PR std: {round(auc_pr_std, 4)}')
            logging.info(f'- Macro F1: {round(f1_macro, 4)}')

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
        '--dataset',
        type=str,
        required=True,
        help="Name of the dataset."
    )
    args.add_argument(
        '--task',
        type=str,
        required=True,
        help="If 'pretext', then only pretext of CARLA is run."\
        "If 'classification', then only classification of CARLA is run."\
        "If 'pretext_classification', then the entire CARLA is run."\
        "If 'train_vae', then VAE is trained for GenIAS pretext scheme."
    )
    args.add_argument(
        '--pretext-timestamp',
        type=str,
        help="If task is 'classification', then the pretrained pretext model"\
        "is needed. The model trained at this timestamp is loaded."
    )
    args.add_argument(
        '--start-subdata',
        type=Optional[str],
        help='The task starts from this subdata.'
    )
    args.add_argument(
        '--pretext-scheme',
        type=str,
        help="If 'carla', customized algorithm generates anomaly."\
        "If 'carla_modified', modified CARLA anomaly algorithm is applied."\
        "If 'genias', the VAE generates anomaly."\
        "If 'mix', the algorithm and VAE alternately generate anomaly."\
        "If 'genias_multiple', the multiple anomalies from VAE is used."
    )
    args.add_argument(
        '--inject-different-anomalies',
        type=str2bool,
        default=False,
        help="For the 'carla' and 'carla_modified' schemes."\
        "Applies different types of anomalies to each dimension of window."\
        "Default False."
    )
    args.add_argument(
        '--use-pretrained-vae',
        type=str2bool,
        default=True,
        help="The pretrained VAE is used for pretext scheme using VAE. "\
        "Default True. If False, then VAE is trained before pretext."\
        "In this case, you may need to customize 'vae_depth' and 'batch_size'."
    )
    args.add_argument(
        '--mix-step',
        type=int,
        default=50,
        help="When pretext scheme is 'mix', then the algorithm and VAE"\
        "alternatively generates anomaly by this step. Default 50."
    )
    args.add_argument(
        '--cut-negative-pairs',
        type=str2bool,
        default=True,
        help="When pretext scheme is 'genias_multiple', the one negative pair"\
        "is used for each anchor in classification stage. Default True."
    )
    args.add_argument(
        '--vae-depth',
        type=int,
        default=10,
        help="The depth of encoder and decoder of VAE for 'vae_train' task."\
        "Default 10."
    )
    args.add_argument(
        '--vae-prior-var',
        type=float,
        default=0.5,
        help="The prior variance of latent space of VAE. Default 0.5."
    )
    args.add_argument(
        '--vae-recon-weight',
        type=float,
        default=1.0,
        help="The weight of reconstruction loss of VAE. Default 1.0."
    )
    args.add_argument(
        '--vae-pert-weight',
        type=float,
        default=0.1,
        help="The weight of perturbation loss of VAE. Default 0.1."
    )
    args.add_argument(
        '--vae-zero-pert-weight',
        type=float,
        default=0.01,
        help="The weight of zero perturbation loss of VAE. Default 0.01."
    )
    args.add_argument(
        '--vae-kld-weight',
        type=float,
        default=0.1,
        help="The weight of KL-divergence loss of VAE. Default 0.1."
    )
    args.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help="Batch size. Default 50. When task is 'train_vae', then set it"\
        "to 100."
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help="The task is run on this gpu. Default 0."
    )
    args.add_argument(
        '--seed',
        type=int,
        default=42,
        help="The fixed seed. Default 42."
    )
    config = args.parse_args()
    main(
        exp_name=config.exp_name,
        dataset=config.dataset,
        task=config.task,
        start_subdata=config.start_subdata,
        pretext_timestamp=config.pretext_timestamp,
        pretext_scheme=config.pretext_scheme,
        inject_different_anomalies=config.inject_different_anomalies,
        use_pretrained_vae=config.use_pretrained_vae,
        mix_step=config.mix_step,
        cut_negative_pairs=config.cut_negative_pairs,
        vae_depth=config.vae_depth,
        vae_prior_var=config.vae_prior_var,
        vae_recon_weight=config.vae_recon_weight,
        vae_pert_weight=config.vae_pert_weight,
        vae_zero_pert_weight=config.vae_zero_pert_weight,
        vae_kld_weight=config.vae_kld_weight,
        batch_size=config.batch_size,
        gpu_num=config.gpu_num,
        seed=config.seed,
    )