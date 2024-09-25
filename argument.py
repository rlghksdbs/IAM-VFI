import argparse

def add_argparse():
    parser = argparse.ArgumentParser()

    ## Wandb
    parser.add_argument('--project-name', default='test', type=str, help='wandb project name')
    parser.add_argument('--wandb-name', default=None, type=str, help='wandb_name')
    parser.add_argument('--wandb-model', action='store_true', help='save wandb model')
    parser.add_argument('--wandb-image', action='store_true', help='save wandb image')

    ## Training
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--log-path', type=str, default='train_single')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='minibatch size')
    parser.add_argument('--val-batch-size', default=32, type=int, help='minibatch size')
    parser.add_argument('--datasets-path', default='./data', type=str, help='dataset root path')
    parser.add_argument('--datasets', default=['vimeo_triplet'], nargs='+', type=str, help='vimeo_triplet')

    parser.add_argument('--patch-size-x', default=256, type=int, help='x patch size')
    parser.add_argument('--patch-size-y', default=256, type=int, help='y patch size')
    parser.add_argument('--init-lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--min-lr', default=3e-5, type=float, help='minimum learning rate for cosine annealing')
    parser.add_argument('--warm-epoch', default=5, type=int, help='warm epoch for cosine annealing')
    parser.add_argument('--lr-scheduler', default='cosineWarmup', type=str, help='can be [cosineWarmup, step]')
    parser.add_argument('--lr-step-gamma', default=0.5, type=float, help='(lr step) multiplication factor(ratio)')
    parser.add_argument('--lr-step-num', default=5, type=float, help='(lr step) the number of step during total training step')
    parser.add_argument('--rotate90', action='store_true', help='rotate 90 degree for augmentation')
    parser.add_argument('--rotate', action='store_true', help='rotate 1-89 degree for augmentation')
    parser.add_argument('--world-size', default=1, type=int, help='world size')
    
    parser.add_argument('--contrast', default=0.5, type=float, help='contrast strength for augmentaiton (0~1)')
    parser.add_argument('--brightness', default=0.5, type=float, help='brightness strength for augmentaiton (0~1)')    
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma strength for augmentaiton (0~1)')
    parser.add_argument('--hue', default=0.1, type=float, help='hue strength for augmentaiton (0~1)')
    parser.add_argument('--saturation', default=0.1, type=float, help='saturation strength for augmentaiton (0~1)')
    parser.add_argument('--zoom', default=0., type=float, help='zoom strength for augmentaiton (0~1)')
    parser.add_argument('--aug-prob', default=0.0, type=float, help='augmentation probability from contrast to zoom')
    parser.add_argument('--cut-mix', default=0.0, type=float, help='augmentation probability from contrast to zoom')
    parser.add_argument('--arbitrary', action='store_true', help='enable arbitrary timestep(on/off)')
    
    parser.add_argument('--pkl_model_path', default=None, help='pkl folder path')
    parser.add_argument('--pkl_name', default='flownet.pkl', help='dataset path')
    parser.add_argument('--save-img', action='store_true', help='enable arbitrary timestep(on/off)')
    parser.add_argument('--output-path', default='../output/RIFE', help='Output dir path of total dataset')

    # SITI argument
    parser.add_argument('--split_data_path', default='./patch_data', help='split data path')
    parser.add_argument('--ti_type', default='easy', help='ti dataset type')
    parser.add_argument('--ti_threshold1', default=15, help='ti easy threshold')
    parser.add_argument('--ti_threshold2', default=30, help='ti medium threshold')

    parser.add_argument('--classifier', action='store_true', help='train classifier')
    parser.add_argument('--classifier_intra_v2', action='store_true', help='train classifier')

    parser.add_argument('--easy_pkl_model_path', default='./ckpt/flownet_easy.pkl', help='pkl folder path')
    parser.add_argument('--medium_pkl_model_path', default='./ckpt/flownet_medium.pkl', help='pkl folder path')
    parser.add_argument('--hard_pkl_model_path', default='./ckpt/flownet_hard.pkl', help='pkl folder path')

    # Test Dataset Path
    parser.add_argument('--hd_data_path', default='../data/09_HD_dataset/', help='dataset path')
    parser.add_argument('--snu_data_path', default='../data/10_SNU-FILM/', help='dataset path')
    parser.add_argument('--ultra_data_path', default='../data/08_4KUltraVideo/', help='dataset path')
    parser.add_argument('--xiph_2k_4k_data_path', default='../data/18_Xiph', help='xiph 2k 4k dataset path')

    args = parser.parse_args()

    return args
