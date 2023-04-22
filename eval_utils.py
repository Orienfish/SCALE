#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import sys
import os
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

STU_NAME = 'student'
TEST_MODE = 'test'
VAL_MODE = 'val'

def plot_tsne(x, y_pred, y_true=None, title='', fig_name=''):
    """
    Plot the TSNE of x, assigned with true labels and pseudo labels respectively.
    Args:
        x: (batch_size, input_dim), raw data to be plotted
        y_pred: (batch_size), optional, pseudo labels for x
        y_true: (batch_size), ground-truth labels for x
        title: str, title for the plots
        fig_name: str, the file name to save the plot
    """
    tsne = TSNE(2, perplexity=50)
    x_emb = tsne.fit_transform(x)

    if y_true is not None: # Two subplots
        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_pred,
                        palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full", ax=ax1)
        ax1.set_title('Clusters with pseudo labels, {}'.format(title))
        ax2 = plt.subplot(122)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_true,
                        palette=sns.color_palette("hls", np.unique(y_true).size),
                        legend="full", ax=ax2)
        ax2.set_title('Clusters with true labels, {}'.format(title))
    else: # Only one plot for predicted labels
        fig = plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                        hue=y_pred, palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full")
        plt.title('Clusters with pseudo labels, {}'.format(title))

    if fig_name != '':
        plt.savefig(fig_name, bbox_inches='tight')

    plt.close(fig)


def plot_mem(mem, model, opt, epoch, cur_step):
    # plot t-SNE for memory embeddings
    mem_images, mem_labels = mem.get_mem_samples_w_true_labels()
    if torch.cuda.is_available():
        mem_images = mem_images.cuda(non_blocking=True)

    mem_embeddings = model(mem_images).detach().cpu().numpy()
    mem_labels = mem_labels.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=10, random_state=0).fit(mem_embeddings)
    mem_pred_labels = kmeans.labels_
    plot_tsne(mem_embeddings, mem_pred_labels, mem_labels,
              title='{}'.format(opt.criterion),
              fig_name=os.path.join(opt.save_folder, 'mem{}_{}.png'.format(epoch, cur_step)))


def plot_mem_select(all_embeddings, all_true_labels, select_indexes, opt,
                    epoch, cur_step):
    tsne = TSNE(2, perplexity=50)
    x_emb = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                    hue=all_true_labels,
                    palette=sns.color_palette("hls",
                                              np.unique(all_true_labels).size),
                    legend="full", alpha=0.2)
    sns.scatterplot(x=x_emb[select_indexes, 0], y=x_emb[select_indexes, 1],
                    hue=all_true_labels[select_indexes],
                    palette=sns.color_palette("hls",
                                              np.unique(all_true_labels).size),
                    legend=False)
    plt.title('Memory selection, {}'.format(opt.criterion))
    plt.savefig(os.path.join(opt.save_folder,
                             'mem_select_{}_{}.png'.format(epoch, cur_step)),
                bbox_inches='tight')
    plt.close(fig)

def eval_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert (y_pred.size == y_true.size), \
        "Incorrect label length in eval_acc! y_pred {}, y_true {}".format(
            y_pred.size, y_true.size)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    rInd, cInd = linear_assignment(w.max() - w)
    # print(w)
    acc = sum([w[rInd[i], cInd[i]] for i in range(rInd.size)]) * 1.0 / y_pred.size

    # compute confusion matrix and purity
    # confusion = np.zeros((D, D), dtype=np.int64)
    # for i in range(D):
    #    for j in range(D):
    #        confusion[i, rInd[i]] = w[rInd[i], cInd[j]]
    purity = np.sum(np.max(w, axis=0)) / np.sum(w)
    return acc, purity


def tsne_simil(x, metric='euclidean', sigma=1.0):
    dist_matrix = pairwise_distances(x, metric=metric)
    cur_sim = np.divide(- dist_matrix, 2 * sigma ** 2)
    # print(np.sum(cur_sim, axis=1, keepdims=True))

    # mask-out self-contrast cases
    # the diagonal elements of exp_logits should be zero
    logits_mask = np.ones((x.shape[0], x.shape[0]))
    np.fill_diagonal(logits_mask, 0)
    # print(logits_mask)
    exp_logits = np.exp(cur_sim) * logits_mask
    # print(exp_logits.shape)
    # print(np.sum(exp_logits, axis=1, keepdims=True))

    p = np.divide(exp_logits, np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    p = p + p.T
    p /= 2 * x.shape[0]
    return p


def cluster_eval(test_embeddings, test_labels, opt, mem, cur_step, epoch,
                 logger, model_name, mode):
    """Cluster and plot in evaluations"""
    num_classes = int(np.unique(test_labels).size * opt.k_scale)

    # perform k-means clustering
    st = time.time()
    test_pred_labels = KMeans(n_clusters=num_classes, init='k-means++', n_init=10,
                              max_iter=300, verbose=0).fit_predict(test_embeddings)
    kmeans_time = time.time() - st
    kmeans_acc, kmeans_purity = eval_acc(test_labels, test_pred_labels)

    print('Val: [{0}][{1}]\t kmeans: acc {acc} purity {purity} (time {time})'.format(
        epoch, cur_step, time=kmeans_time, acc=kmeans_acc, purity=kmeans_purity))
    sys.stdout.flush()

    if mode == TEST_MODE and model_name == STU_NAME and opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, test_pred_labels, test_labels,
                  title='{} kmeans {}'.format(opt.criterion, kmeans_acc),
                  fig_name=os.path.join(opt.save_folder, 'kmeans_{}_{}.png'.format(epoch, cur_step)))

    # perform agglomerative clustering
    metric = 'cosine'
    st = time.time()
    test_pred_labels = AgglomerativeClustering(
        n_clusters=10, affinity=metric, linkage='average').fit_predict(test_embeddings)
    exec_time = time.time() - st
    agg_acc, agg_purity = eval_acc(test_labels, test_pred_labels)

    print('Val: [{0}][{1}]\t agg {metric} {linkage}: acc {acc} purity {purity} (time {time})'.format(
        epoch, cur_step, metric=metric, linkage='average', time=exec_time, acc=agg_acc, purity=agg_purity))
    sys.stdout.flush()

    if mode == TEST_MODE and model_name == STU_NAME and opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, test_pred_labels, test_labels,
                  title='{method} agg {metric} {linkage} {acc}'.format(method=opt.criterion, metric=metric,
                                                                       linkage='average', acc=agg_acc),
                  fig_name=os.path.join(opt.save_folder,
                                        'agg_{}_{}_{}_{}.png'.format(metric, 'average', epoch, cur_step)))

    # perform spectral clustering
    st = time.time()
    similarity_matrix = tsne_simil(test_embeddings, metric=metric)
    test_pred_labels = SpectralClustering(n_clusters=num_classes, affinity='precomputed', n_init=10,
                                          assign_labels='discretize').fit_predict(similarity_matrix)
    spectral_time = time.time() - st
    spectral_acc, spectral_purity = eval_acc(test_labels, test_pred_labels)

    print('Val: [{0}][{1}]\t spectral {metric}: acc {acc} purity {purity} (time {time})'.format(
        epoch, cur_step, metric=metric, time=spectral_time, acc=spectral_acc, purity=spectral_purity))
    sys.stdout.flush()

    if mode == TEST_MODE and model_name == STU_NAME and opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, test_pred_labels, test_labels,
                  title='{} spectral {} {}'.format(opt.criterion, metric, spectral_acc),
                  fig_name=os.path.join(opt.save_folder,
                                        'spectral_{}_{}_{}.png'.format(metric, epoch, cur_step)))

    if mode == TEST_MODE:
        logger.log_value('kmeans acc {}'.format(model_name), kmeans_acc, cur_step)
        logger.log_value('kmeans purity {}'.format(model_name), kmeans_purity, cur_step)
        logger.log_value('agg {metric} {linkage} acc {model_name}'.format(
            metric=metric, linkage='average', model_name=model_name),
            agg_acc, cur_step)
        logger.log_value('agg {metric} {linkage} purity {model_name}'.format(
            metric=metric, linkage='average', model_name=model_name),
            agg_purity, cur_step)
        logger.log_value('spectral {metric} acc {model_name}'.format(
            metric=metric, model_name=model_name),
            spectral_acc, cur_step)
        logger.log_value('spectral {metric} purity {model_name}'.format(
            metric=metric, model_name=model_name),
            spectral_purity, cur_step)

    if mode == TEST_MODE and model_name == STU_NAME:
        with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('{epoch},{step},kmeans,{kmeans_acc},agg,{agg_acc},spectral,{spectral_acc},'.format(
                epoch=epoch, step=cur_step, kmeans_acc=kmeans_acc, agg_acc=agg_acc, spectral_acc=spectral_acc
            ))


def eval_knn(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels):
    # perform kNN classification
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=50)
    pred_labels = neigh.fit(knn_train_embeddings, knn_train_labels).predict(test_embeddings)
    knn_acc = np.sum(pred_labels == test_labels) / pred_labels.size
    return knn_acc


def knn_eval(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels,
             opt, mem, cur_step, epoch, logger, model_name, mode):
    """KNN classification and plot in evaluations"""
    # perform kNN classification
    from sklearn.neighbors import KNeighborsClassifier
    st = time.time()
    neigh = KNeighborsClassifier(n_neighbors=opt.kneighbor)
    pred_labels = neigh.fit(knn_train_embeddings, knn_train_labels).predict(test_embeddings)
    knn_time = time.time() - st
    knn_acc = np.sum(pred_labels == test_labels) / pred_labels.size

    print('Val: [{0}][{1}]\t knn: acc {acc} (time {time})'.format(
        epoch, cur_step, time=knn_time, acc=knn_acc))
    sys.stdout.flush()

    if mode == TEST_MODE and model_name == STU_NAME and opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, pred_labels, test_labels,
                  title='{} knn {}'.format(opt.criterion, knn_acc),
                  fig_name=os.path.join(opt.save_folder, 'knn_{}_{}.png'.format(epoch, cur_step)))

    logger.log_value('{}: knn acc {}'.format(mode, model_name), knn_acc, cur_step)

    if mode == TEST_MODE and model_name == STU_NAME:
        with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('{epoch},{step},knn,{knn_acc},\n'.format(epoch=epoch, step=cur_step, knn_acc=knn_acc))


def knn_task_eval(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels,
                  opt, mem, cur_step, epoch, logger, task_list, model_name):
    """KNN classification and plot in evaluations"""

    from sklearn.neighbors import KNeighborsClassifier
    st = time.time()
    knn_task_acc = []

    for task in task_list:
        # perform kNN classification
        knn_train_ind = np.isin(knn_train_labels, task)
        test_ind = np.isin(test_labels, task)
        neigh = KNeighborsClassifier(n_neighbors=opt.kneighbor)
        pred_labels = neigh.fit(knn_train_embeddings[knn_train_ind],
                                knn_train_labels[knn_train_ind]).predict(test_embeddings[test_ind])
        knn_acc = np.sum(pred_labels == test_labels[test_ind]) / pred_labels.size
        knn_task_acc.append(knn_acc)

    knn_time = time.time() - st
    knn_task_acc = np.mean(knn_task_acc)

    print('Val: [{0}][{1}]\t knn task: acc {acc} (time {time})'.format(
        epoch, cur_step, time=knn_time, acc=knn_task_acc))
    sys.stdout.flush()

    logger.log_value('knn task acc {}'.format(model_name), knn_task_acc, cur_step)

    if model_name == STU_NAME:
        with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('knn_task,{knn_acc},\n'.format(knn_acc=knn_task_acc))


def eval_forget(acc_mat):
    """
    Evaluate the forgetting measure based on accuracy matrix
    Args:
        acc_mat: numpy array with shape (phase#, class#)
                 acc_mat[i, j] is the accuracy on class j after phase i
    Return:
        a scalar forgetting measure
    """
    forget_pc = acc_mat - acc_mat[-1, :].reshape((1, -1)) # (phase#, class#)
    forget_pc = np.maximum(forget_pc, 0) # Make sure forgetting is positive
    forget_pc = np.max(forget_pc, axis=0) # (class#)
    return np.mean(forget_pc)


def eval_forward_transfer(self, acc_mat):
    """
    Evaluate the forward transfer based on accuracy matrix
    Args:
        acc_mat: numpy array with shape (phase#, class#)
                 acc_mat[i, j] is the accuracy on class j after phase i
    Return:
        a scalar forward transfer measure
    """
    transfer_pc = np.diagonal(acc_mat, offset=1) - 0.1 # set 10% as acc for random network
    return np.mean(transfer_pc)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state