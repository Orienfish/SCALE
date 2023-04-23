import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--ckpt_dir_1', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--ood_eval', action='store_true',
                        help='Test on the OOD set')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'tinyimagenet',
                                 'stream51', 'core50', 'cub200', 'path'],
                        help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--training_data_type', type=str, default='sequential',
                        choices=['iid', 'class_iid', 'instance', 'class_instance'],
                        help='iid or sequential datastream')
    parser.add_argument('--blend_ratio', type=float, default=.0,
                        help="the ratio blend classes at the boundary")
    parser.add_argument('--n_concurrent_classes', type=int, default=1,
                        help="the number of concurrent classes showing at the same time")
    parser.add_argument('--train_samples_per_cls', type=int, nargs='+',
                        help="the number of training samples per class")
    parser.add_argument('--test_samples_per_cls', type=int, default=1000,
                        help="the number of testing samples per class")
    parser.add_argument('--knn_samples', type=int, default=0,
                        help='number of labeled samples loaded for kNN training')
    parser.add_argument('--kneighbor', type=int, default=50,
                        help="the number of neighbors in knn")

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='batch_size in validation')
    parser.add_argument('--steps_per_batch_stream', type=int, default=20,
                        help='number of steps for per batch of streaming data')
    parser.add_argument('--learning_rate_stream', type=float, default=0.01,
                        help='learning rate for streaming new data')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--temp_cont', type=float, default=0.07,
                        help='temperature for contrastive loss function')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')

    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    parser.add_argument('--k_scale', type=float, default=1.0,
                        help='to scale the number of classes during evaluation')
    parser.add_argument('--plot', default=False, action="store_true",
                        help="whether to plot during evaluation")

    # method
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'cnn'])
    parser.add_argument('--method', type=str, default='mixup',
                        choices=['mixup', 'pnn', 'si', 'der', 'finetune'],
                        help='choose method')
    parser.add_argument('--model_name', type=str, default='supcon',
                        choices=['simclr', 'simsiam', 'barlowtwins', 'vicreg',
                                 'byol'],
                        help='choose method')

    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    """if args.debug:
        if args.train: 
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval: 
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0"""


    # assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    #set_deterministic(args.seed)

    args.data_folder = '../datasets/'


    vars(args)['aug_kwargs'] = {
        'name':args.model_name,
        'image_size': args.datasetsetting.image_size
    }
    vars(args)['dataset_kwargs'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset': args.datasetsetting.name,
        'data_dir': args.data_dir,
        'download': args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }

    if len(args.train_samples_per_cls) == 1:
        im_ind = args.train_samples_per_cls[0]
    else:
        im_ind = 'im'

    args.model_dir = '{}_{}_{}_{}_{}_{}_{}_r{}_{}_{}_lrs_{}_bsz_{}_steps_{}_epoch_{}_trial_{}'.format(
        args.method, args.model_name, args.backbone, args.dataset,
        args.training_data_type, args.blend_ratio, args.n_concurrent_classes,
        im_ind, args.test_samples_per_cls, args.knn_samples,
        args.learning_rate_stream, args.batch_size,
        args.steps_per_batch_stream,
        args.epochs, args.trial)

    args.model_path = './save/{}_models'.format(args.dataset)
    args.tb_path = './save/{}_tensorboard'.format(args.dataset)
    args.tb_folder = os.path.join(args.tb_path, args.model_dir)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_dir)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args
