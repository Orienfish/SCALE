from __future__ import print_function
import torch
import os
import copy
import sys
import argparse
import time
import numpy as np
import random
from tqdm import tqdm

import tensorboard_logger as tb_logger

from util import AverageMeter

from memory import Memory
from losses.supcon import similarity_mask_new, similarity_mask_old  # for scale
from losses import get_loss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from eval_utils import knn_eval, knn_task_eval, cluster_eval, plot_mem_select, \
    plot_mem, save_model
from data_utils import set_loader
from set_utils import load_student_backbone, set_optimizer, set_constant_learning_rate

VAL_CNT = 10       # Number of validations to allow during training
SAVE_MODEL = False # Whether to save model for offline linear evaluation
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 20,
    'tinyimagenet': 10,  # temp setting
    'mnist': 10
}

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    #parser.add_argument('--plot_freq', type=int, default=5000,
    #                    help='plot frequency of steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='batch_size in validation')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--steps_per_batch_stream', type=int, default=20,
                        help='number of steps for per batch of streaming data')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')

    # optimization
    parser.add_argument('--learning_rate_stream', type=float, default=0.1,
                        help='learning rate for streaming new data')
    parser.add_argument('--learning_rate_pred', type=float, default=0.01,
                        help='learning rate for predictor g')
    # parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
    #                    help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1,
    #                    help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--distill_power', type=float, default=0.1)


    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'tinyimagenet', 'path'],
                        help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--training_data_type', type=str, default='iid',
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


    # method
    parser.add_argument('--criterion', type=str, default='supcon',
                        choices=['supcon', 'simclr', 'scale'],
                        help='major criterion')
    parser.add_argument('--lifelong_method', type=str, default='none',
                        choices=['none', 'scale', 'co2l', 'cassle'],
                        help='choose lifelong learning method')
    parser.add_argument('--simil', type=str, default='tSNE',
                        choices=['tSNE', 'kNN'],
                        help='choose similarity metric')

    # temperature
    parser.add_argument('--temp_cont', type=float, default=0.1,
                        help='temperature for contrastive loss function')
    parser.add_argument('--temp_tSNE', type=float, default=0.1,
                        help='temperature for tSNE similarity')
    parser.add_argument('--thres_ratio', type=float, default=0.3,
                        help='threshold ratio between mean and max')
    parser.add_argument('--current_temp', type=float, default=0.1,
                        help='temperature for loss function')
    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to the pretrained backbone to load')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')

    # Memory
    parser.add_argument('--mem_max_classes', type=int, default=1,
                        help='max number of classes in memory')
    parser.add_argument('--mem_size', type=int, default=128,
                        help='number of samples per class in memory')
    parser.add_argument('--mem_samples', type=int, default=0,
                        help='number of samples to add during each training step')
    parser.add_argument('--mem_update_type', type=str, default='reservoir',
                        choices=['rdn', 'mo_rdn', 'reservoir', 'simil'],
                        help='memory update policy')
    parser.add_argument('--mem_w_labels', default=False, action="store_true",
                        help="whether use labels during memory update")
    parser.add_argument('--mem_cluster_type', type=str, default='none',
                        choices=['none', 'kmeans', 'spectral', 'max_coverage',
                                 'psa', 'maximin', 'energy'],
                        help="which clustering method to use during unsupervised update")
    parser.add_argument('--mem_max_new_ratio', type=float, default=0.1,
                        help='max ratio of new samples at each memory update')

    # Evaluation
    parser.add_argument('--k_scale', type=float, default=1.0,
                        help='to scale the number of classes during evaluation')
    parser.add_argument('--plot', default=False, action="store_true",
                        help="whether to plot during evaluation")

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/{}_models/'.format(opt.dataset)
    opt.tb_path = './save/{}_tensorboard/'.format(opt.dataset)

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))

    if len(opt.train_samples_per_cls) == 1:
        im_ind = opt.train_samples_per_cls[0]
    else:
        im_ind = 'im'

    opt.model_name = '{}_{}_{}_{}_{}_{}_{}_s{}_{}_{}_lrs_{}_bsz_{}_mem_{}_{}_{}_{}_{}_{}_temp_{}_' \
                     'simil_{}_{}_{}_distill_{}_{}_{}_steps_{}_epoch_{}_trial_{}'.format(
        opt.lifelong_method, opt.criterion, opt.dataset, opt.model,
        opt.training_data_type, opt.blend_ratio, opt.n_concurrent_classes,
        im_ind, opt.test_samples_per_cls, opt.knn_samples,
        opt.learning_rate_stream, opt.batch_size, opt.mem_samples,
        opt.mem_size, opt.mem_max_classes, opt.mem_update_type, opt.mem_cluster_type, int(opt.mem_w_labels),
        opt.temp_cont, opt.simil, opt.temp_tSNE, opt.thres_ratio,
        opt.distill_power, opt.current_temp, opt.past_temp,
        opt.steps_per_batch_stream, opt.epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def train_step(images, labels, models, criterions, optimizer,
               meters, opt, mem, train_transform):
    """One gradient descent step"""

    model, past_model = models
    criterion, criterion_reg = criterions
    losses_stream, losses_contrast, losses_distill, \
        pos_pairs, forward_time, loss_time, pair_comp_time = meters

    # load memory samples
    mem_images, mem_labels = mem.get_mem_samples()

    # get augmented streaming and augmented samples and concatenate them
    if opt.mem_samples > 0 and mem_images is not None:
        sample_cnt = min(opt.mem_samples, mem_images.shape[0])
        select_ind = np.random.choice(mem_images.shape[0], sample_cnt, replace=False)

        # Augment memory samples
        aug_mem_images_0 = torch.stack([train_transform(ee.cpu() * 255)
                                        for ee in mem_images[select_ind]])
        aug_mem_images_1 = torch.stack([train_transform(ee.cpu() * 255)
                                        for ee in mem_images[select_ind]])
        # print('aug mem image', aug_mem_images[0][1:10])
        feed_images_0 = torch.cat([images[0], aug_mem_images_0], dim=0)
        feed_images_1 = torch.cat([images[1], aug_mem_images_1], dim=0)
        feed_labels = torch.cat([labels, mem_labels[select_ind]], dim=0)
    else:
        feed_images_0 = images[0]
        feed_images_1 = images[1]
        feed_labels = labels

    if torch.cuda.is_available():
        feed_images_0 = feed_images_0.cuda(non_blocking=True)
        feed_images_1 = feed_images_1.cuda(non_blocking=True)
        feed_labels = feed_labels.cuda(non_blocking=True)

    feed_images_all = torch.cat([feed_images_0, feed_images_1], dim=0)
    bsz = feed_images_0.shape[0]
    
    model.train()
    start = time.time()

    loss_distill = .0
    if opt.lifelong_method == 'none':

        if opt.criterion == 'supcon':
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1,
                             labels=feed_labels)

        elif opt.criterion in ['simclr', 'cka', 'barlowtwins', 'byol', 'vicreg', 'simsiam']:
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1)

        # print(loss_contrast)

    elif opt.lifelong_method == 'scale':

        f0_logits, loss_distill = criterion_reg(model, past_model,
                                                feed_images_0)
        losses_distill.update(loss_distill.item(), bsz)
        pair_comp_time.update(time.time() - start)

        #contrast_mask = similarity_mask_new(opt.batch_size, f0_logits, opt, pos_pairs)
        features_all = model(feed_images_all)
        contrast_mask = similarity_mask_old(features_all, bsz, opt, pos_pairs)
        loss_contrast = criterion(model, model, feed_images_0, feed_images_1,
                                  mask=contrast_mask)

    elif opt.lifelong_method == 'co2l':

        f0_logits, loss_distill = criterion_reg(model, past_model,
                                                feed_images_0)
        losses_distill.update(loss_distill.item(), bsz)

        loss_contrast = criterion(model, model, feed_images_0, feed_images_1)

    elif opt.lifelong_method == 'cassle':

        loss_contrast = criterion(model, model, feed_images_0, feed_images_1)

    else:
        raise ValueError('contrastive method not supported: {}'.format(opt.lifelong_method))


    losses_contrast.update(loss_contrast.item(), bsz)
    loss_time.update(time.time() - start)

    if train_step.distill_power <= 0.0 and loss_distill > 0.0:
        train_step.distill_power = losses_contrast.avg * opt.distill_power / losses_distill.avg

    loss = loss_contrast + train_step.distill_power * loss_distill

    losses_stream.update(loss.item(), bsz)

    # SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if opt.criterion == 'byol':
        criterion.update_moving_average()

train_step.distill_power = 0.0


def train(train_loader, test_loader, knntrain_loader,
          train_transform, model, criterions, optimizer, epoch,
          opt, mem, logger, task_list):
    """training of one epoch on single-pass of data"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    loss_time = AverageMeter()
    pair_comp_time = AverageMeter()
    mem_update_time = AverageMeter()
    losses_stream = AverageMeter()
    losses_contrast = AverageMeter()
    losses_distill = AverageMeter()
    pos_pairs_stream = AverageMeter()
    meters = [losses_stream, losses_contrast, losses_distill,
              pos_pairs_stream, forward_time, loss_time, pair_comp_time]

    batches_per_epoch = len(train_loader)
    steps_per_epoch_stream = opt.steps_per_batch_stream * batches_per_epoch
    cur_stream_step = (epoch - 1) * steps_per_epoch_stream

    # Set validation frequency
    val_freq = np.floor(len(train_loader) / VAL_CNT).astype('int')
    print('Validation frequency: {}'.format(val_freq))

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # test - plot t-SNE
        if idx % val_freq == 0:
            validate(test_loader, knntrain_loader, model, optimizer,
                     opt, mem, cur_stream_step, epoch, logger, task_list)

        # record a snapshot of the model as past model
        past_model = copy.deepcopy(model)
        past_model.eval()
        if opt.lifelong_method == 'cassle':
            criterion, _ = criterions
            criterion.freeze_backbone(model)

        models = [model, past_model]
        # compute loss
        for _ in range(opt.steps_per_batch_stream):
            
            # Trigger one gradient descent step
            train_step(images, labels, models, criterions,
                       optimizer, meters, opt, mem, train_transform)

            # tensorboard logger
            logger.log_value('learning_rate',
                             optimizer.param_groups[0]['lr'],
                             cur_stream_step)
            logger.log_value('loss',
                             losses_stream.avg,
                             cur_stream_step)
            logger.log_value('loss_contrast',
                             losses_contrast.avg,
                             cur_stream_step)
            logger.log_value('loss_distill',
                             losses_distill.avg * train_step.distill_power,
                             cur_stream_step)
            logger.log_value('pos_pair_stream', pos_pairs_stream.avg,
                             cur_stream_step)
            cur_stream_step += 1

            # print info
            if cur_stream_step % opt.print_freq == 0:
                print('Train stream: [{0}][{1}][{2}/{3}]\t'
                      'loss {4}\t'
                      'loss_contrast {5}\t'
                      'loss_distill {6}\t'
                      'pos pairs {pos_pair.val:.3f} ({pos_pair.avg:.3f})'.format(
                    epoch, cur_stream_step, idx + 1, len(train_loader),
                    losses_stream.avg,
                    losses_contrast.avg,
                    losses_distill.avg * train_step.distill_power,
                    pos_pair=pos_pairs_stream))
                print('Forward time {}\tloss time {}\t'
                      'pairwise comp time {}\tmemory update time {}'.format(
                    forward_time.avg, loss_time.avg,
                    pair_comp_time.avg, mem_update_time.avg
                ))
                sys.stdout.flush()

        # update memory
        mem_start = time.time()
        if opt.mem_w_labels:
            mem.update_w_labels(images[2], labels)
            mem_end = time.time()
        else:
            all_embeddings, all_true_labels, select_indexes = \
                mem.update_wo_labels(images[2], labels, model)
            mem_end = time.time()
            if opt.plot and idx % val_freq == 0:
                print('plot mem select')
                plot_mem_select(all_embeddings, all_true_labels, select_indexes, opt,
                                epoch, cur_stream_step)

        # measure elapsed time
        mem_update_time.update(mem_end - mem_start)
        batch_time.update(time.time() - end)
        end = time.time()


def validate(test_loader, knn_train_loader, model, optimizer,
             opt, mem, cur_step, epoch, logger, task_list):
    """validation, evaluate k-means clustering accuracy and plot t-SNE"""
    model.eval()
    test_labels, val_labels, knn_labels = [], [], []
    test_embeddings, val_embeddings, knn_embeddings = None, None, None

    # kNN training loader
    for idx, (images, labels) in enumerate(tqdm(knn_train_loader, desc="knn training")):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        embeddings = model(images).detach().cpu().numpy()
        if knn_embeddings is None:
            knn_embeddings = embeddings
        else:
            knn_embeddings = np.concatenate((knn_embeddings, embeddings), axis=0)
        knn_labels += labels.detach().tolist()
    knn_labels = np.array(knn_labels).astype(int)

    # Test loader
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc='test')):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

        # forward prediction
        embeddings = model(images).detach().cpu().numpy()
        if test_embeddings is None:
            test_embeddings = embeddings
        else:
            test_embeddings = np.concatenate((test_embeddings, embeddings), axis=0)
        test_labels += labels.detach().tolist()
    test_labels = np.array(test_labels).astype(int)

    # Unsupervised clustering
    cluster_eval(test_embeddings, test_labels, opt, mem, cur_step, epoch, logger)

    # kNN classification
    knn_eval(test_embeddings, test_labels, knn_embeddings, knn_labels,
             opt, mem, cur_step, epoch, logger)
    # knn_task_eval(test_embeddings, test_labels, knn_embeddings, knn_labels,
    #              opt, mem, cur_step, epoch, logger, task_list, model_name)

    # Memory plot
    if opt.plot and opt.mem_size > 0 and cur_step > 0:
        plot_mem(mem, model, opt, epoch, cur_step)

    # Save the current embeddings
    # np.save(os.path.join(opt.save_folder, 'test_embeddings.npy'), test_embeddings)
    # np.save(os.path.join(opt.save_folder, 'test_labels.npy'), test_labels)

    # Save the current model
    if SAVE_MODEL:
        save_file = os.path.join(opt.save_folder, '{}_{}.pth'.format(epoch, cur_step))
        save_model(model, optimizer, opt, opt.epochs, save_file)


def main():
    opt = parse_option()

    print("============================================")
    print(opt)
    print("============================================")

    # set seed for reproducing
    random.seed(opt.trial)
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)

    # build data loader
    train_loader, test_loader, knntrain_loader, train_transform = set_loader(opt)

    # build model
    model = load_student_backbone(opt.model, opt.ckpt)

    # build criterion
    criterion, criterion_reg = get_loss(opt)
    criterions = [criterion, criterion_reg]

    # build optimizer
    optimizer = set_optimizer(opt.learning_rate_stream,
                              opt.momentum,
                              opt.weight_decay,
                              model,
                              criterion=criterion)

    # init memory
    mem = Memory(opt)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    batches_per_epoch = len(train_loader)
    steps_per_epoch_stream = opt.steps_per_batch_stream * batches_per_epoch

    # set task list
    #all_labels = []
    #for idx, (images, labels) in enumerate(val_loader):
    #    all_labels += labels.detach().tolist()
    #all_labels = np.sort(np.unique(np.array(all_labels).astype(int)))
    task_list = np.reshape(np.arange(dataset_num_classes[opt.dataset]),
                           (-1, dataset_num_classes[opt.dataset])).tolist()
    print('task list', task_list)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        set_constant_learning_rate(opt.learning_rate_stream, optimizer)

        # train for one epoch
        time1 = time.time()
        train(train_loader, test_loader, knntrain_loader,
              train_transform, model, criterions, optimizer, epoch,
              opt, mem, logger, task_list)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Plot t-SNE
        cur_stream_step = epoch * steps_per_epoch_stream
        opt.plot = True
        print('Test student model')
        validate(test_loader, knntrain_loader, model, optimizer, opt,
                 mem, cur_stream_step, epoch, logger, task_list)
        opt.plot = False

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
