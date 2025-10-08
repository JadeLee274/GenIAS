import torch
import logging
from utils import *
from data_factory import DeepSVDDDataset
import argparse
from deepsvdd import *

args = argparse.ArgumentParser()
args.add_argument(
    '--exp-name',
    type=str,
    help='Experiment name.'
)
args.add_argument(
    '--dataset',
    type=str,
    help="Dataset. Either 'MSL' or 'SMAP'.",
)
args.add_argument(
    '--subdata',
    type=str,
    help="Dataset. Either 'MSL_SEPARATED' or 'SMAP_SEPARATED'.",
)
args.add_argument(
    '--scheme',
    type=str,
    default='carla',
    help="Whether to use carla or genias to create pairs. Default 'carla'."
)
args.add_argument(
    '--epoch',
    type=int,
    default=1000
)
args.add_argument(
    '--use-wandb',
    type=str2bool, 
    default=False,
    help="Whether to use wandb log or not. Default False."
)
args.add_argument(
    '--batch-size',
    type=int,
    default=50,
    help="Batch size. Default 50."
)
args.add_argument(
    '--gpu-num',
    type=int,
    default=0,
    help="gpu number. Default 0.",
)
args.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Fixed seed. Default 42.',
)
args.add_argument(
    '--multiprocess',
    type=bool,
    default=False,
)
args.add_argument(
    '--nu',
    type=float,
    default=0.1
)
args.add_argument(
    '--device',
    type=str,
    default='cuda:0'
)
args.add_argument(
    '--load-epoch',
    type=int,
    default=0
)

def main():
        
    config = args.parse_args()

    fix_seed_all(config.seed)
    set_logging_filehandler(
        log_file_path=f'log/deepsvdd/{config.dataset}/{config.scheme}/{config.exp_name}'
    )

    dataset = DeepSVDDDataset(
        dataset=config.dataset, subdata=config.subdata, mode='train',
        scheme=config.scheme
    )

    window_size, data_dim = dataset.windows[0].shape

    model = SVDD(
        data_dim=data_dim, window_size=window_size, depth=5
    )

    trainer = Trainer(model=model, device=config.device)
    
    ckpt_path = 'checkpoints/deepsvdd'
    os.makedirs(ckpt_path, exist_ok=True)
    if config.load_epoch < config.epoch:
        trainer.train(
            dataset=dataset, total_epoch=config.epoch,
            load_epoch=config.load_epoch
            )
        dataset = DeepSVDDDataset(
            dataset=config.dataset, subdata=config.subdata, 
            mode='test', scheme=config.scheme
        )
        trainer.test(dataset)
    else:
        c = torch.load(os.path.join(ckpt_path, f'c_{config.load_epoch}.pt'))
        R = torch.load(os.path.join(ckpt_path, f'R_{config.load_epoch}.pt'))
        trainer.c, trainer.R = c, R
        dataset = DeepSVDDDataset(
            dataset=config.dataset, subdata=config.subdata, 
            mode='test', scheme=config.scheme
        )
        model.load_state_dict(torch.load(os.path.join(
            ckpt_path, f'model_{config.load_epoch}.pt'
            )))
        trainer.test(dataset, model)

if __name__ == '__main__':
    main()
