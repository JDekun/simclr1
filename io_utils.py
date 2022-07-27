import numpy as np
import random
import argparse
import torch
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(script):
    parser = argparse.ArgumentParser(description='Train SimCLR' %(script))
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    # parser.add_argument('--amp', action='store_true', help='引用--amp时为True，没有引用时为False')
    parser.add_argument('--amp', default=True, type=str2bool, help='amp')
    parser.add_argument('--results_path', default="results", type=str, help='results/')
    parser.add_argument('--datasets_path', default="../../input", type=str, help='../../input')
    parser.add_argument('--amp_level', default='O2', type=str, help='amp_level')
    parser.add_argument('--use_checkpoint', default=False, type=str2bool, help='use_checkpoint')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume')
    parser.add_argument('--resume_path', default='results/128_0.5_200_512_10.pth', type=str, help='resume_path')
    parser.add_argument('--start_epoch', default=1, type=int, help='start_epoch')
    parser.add_argument('--seed', default=1, type=int, help='seed')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed) 


if __name__ == "__main__":
    args = parse_args('train')
    print(args)
