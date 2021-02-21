import argparse
import numpy as np 
from config_manager import get_config


def get_param(known=None):
    args = argparse.ArgumentParser()
    
    args.add_argument('--name', type=str, required=True)

    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--resume', action='store_true')    
    args.add_argument('--abspath', type=str, default='/root/datasets')
    args.add_argument('--config_mode', type=str, default='')
    args.add_argument('--doa_loss', type=str, default='MSE', 
                      choices=['MAE', 'MSE', 'MSLE', 'MMSE'])
    args.add_argument('--model', type=str, default='initial_seldnet_v1', 
                      choices=['initial_seldnet', 'initial_seldnet_v1', 
                      'initial_seldnet_v2', 'initial_seldnet_v3', 
                      'attention_seldnet', 'attention_seldnet_v1',
                      'initial_mmse_seldnet'])
    
    
    # training
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--loss_weight', type=str, default='1,1000')
    args.add_argument('--patience', type=int, default=100)
    args.add_argument('--freq_mask_size', type=int, default=8)
    args.add_argument('--time_mask_size', type=int, default=24)
    args.add_argument('--maxstep', type=int, default=75)
    args.add_argument('--inf', action='store_true')

    # metric
    args.add_argument('--lad_doa_thresh', type=int, default=20)

    if known is None:
        known = []
    config = args.parse_known_args(known)[0]
    config.name = config.model + '_' + config.doa_loss + '_' + config.name
    config = get_config(config.name, config, mode=config.config_mode)

    return config


if __name__ == '__main__':
    import sys
    config = get_param(sys.argv[1:])
    print(config)
