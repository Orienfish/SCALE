import pdb
import csv
import progressbar
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from core.models.STAM_classRepo import *
from core.utils import *
from core.distance_metrics import *


from sklearn import metrics
from sklearn import mixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering
from kmodes.kmodes import KModes
from sklearn.metrics import pairwise_distances, jaccard_score

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
from eval_utils import *


class StamWrapper():

    def __init__(self, configs):
        
        # declare properties
        self.name = 'STAM'

        # extract scenario configs
        self.num_classes = configs['num_classes']
        self.class_labels = configs['class_labels']
        self.vis_train = configs['visualize_train']
        self.vis_cluster = configs['visualize_cluster']
        self.im_size = configs['im_size']
        self.num_c = configs['channels']
        self.seed = configs['seed']
        self.im_scale = configs['im_scale']
        self.scale_flag = configs['scale_flag']
        self.num_samples = configs['num_samples']
        self.num_phases = configs['num_phases']

        # extract stam configs
        self.num_layers = len(configs['layers'])
        self.rho = configs['rho']
        self.nd_fixed = configs['nd_fixed']
        print(self.nd_fixed)

        # directory paths
        self.results_directory = configs['results_directory']
        self.plot_directory = configs['plot_directory']
        
        # initialize variables
        self.images_seen = 0
        self.ltm_cent_counts = np.zeros((self.num_phases,3))

        # Informative centroid info storage
        self.informative_indices = [[] for i in range(self.num_layers)]
        self.informative_indices_2 = [[] for i in range(self.num_layers)]
        self.informative_class = [[] for i in range(self.num_layers)]

        self.expected_features = [configs['expected_features'] for l in range(self.num_layers)]
        self.num_images_init = configs['num_images_init']
        print(self.expected_features)

        # build stam hierarchy
        self.layers = []
        self.layers.append(Layer(self.im_size, self.num_c, *configs['layers'][0], 
                                 configs['WTA'], self.im_scale, self.scale_flag, 
                                 self.seed, configs['kernel'], self.expected_features[0], self.nd_fixed, self.num_images_init, self.plot_directory, self.vis_train))
        for l in range(1,self.num_layers):
            self.layers.append(Layer(self.im_size, self.num_c, *configs['layers'][l], 
                                     configs['WTA'], self.im_scale, self.scale_flag, 
                                     self.seed, configs['kernel'],  self.expected_features[l], self.nd_fixed, self.num_images_init, self.plot_directory, self.vis_train))

        # stam init
        self.init_layers()

        # classification parameters
        self.Fz = [[] for i in range(self.num_samples)]
        self.D = [[] for i in range(self.num_samples)]
        self.D_sum = [[] for i in range(self.num_samples)]
        self.cent_g = [[] for i in range(self.num_samples)]
        self.Nl_seen = [0 for i in range(self.num_samples)]

        # visualize task boundaries
        self.ndy = []

    # centroid init - note that current implementation does NOT use random 
    # init centroids but rather will change these centroids to sampled patch 
    # values in the learning alogrithm (see STAM_classRepo)
    def init_layers(self):
        
        # random seed
        np.random.seed(self.seed)
            
        # for all layers
        for l in range(self.num_layers):
        
            # number of centroids to initialize
            n_l = self.layers[l].num_cents
                                                        
            # random init
            self.layers[l].centroids = np.random.randn(n_l, 
                                                       self.layers[l].recField_size \
                                                       * self.layers[l].recField_size \
                                                       * self.layers[l].ch) * 0.1 

            # normalize sum to 1
            self.layers[l].centroids -= np.amin(self.layers[l].centroids, axis = 1)[:,None]
            self.layers[l].centroids /= np.sum(self.layers[l].centroids, axis = 1)[:,None]


    def train(self, x, y, n, experiment_params, sort=False):
        self.phase = n - 1
    
        # reset d samples - for visualization
        for l in range(self.num_layers):
            self.layers[l].d_sample = 100
            self.layers[l].delete_n = []
        self.ndy.append(self.images_seen+1)

        
        if sort:
            sortd = np.argsort(y)
            x = x[sortd]
            y = y[sortd]

        # make sure data is nonzero
        if len(x) > 0:
        
            # start progress bar
            bar = progressbar.ProgressBar(maxval=len(x), \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
                
            # for all data points
            for i in range(len(x)):            
                # reset d samples
                if i == len(x) - 100:
                    for l in range(self.num_layers):
                        self.layers[l].d_sample = 100

                # show image to hierarchy
                self.images_seen = self.images_seen + 1
                self.train_update(x[i], y[i])

                # update progress bar
                bar.update(i+1)


            # finish progress bar
            bar.finish()

        if True:
            self.save_visualizations(smart_dir(self.plot_directory + '/phase_{}/'.format(n-1)), n-1)

    # update step for stam model training
    def train_update(self, x, label):
    
        # for all layers
        x_i = x
        for l in range(self.num_layers):    
            x_i = self.layers[l].forward(x_i, label, update = True)       


    # initialize hierarchy classfiication parameters for each
    # evaluation data sample
    def setTask(self, num_samples, K):
        
        # classification parameters
        self.Fz = [[] for i in range(num_samples)]
        self.D = [[] for i in range(num_samples)]
        self.D_sum = [[] for i in range(num_samples)]
        self.cent_g = [[] for i in range(num_samples)]
        self.Nl_seen = [0 for i in range(num_samples)]

        # set rho
        self.rho_task = self.rho+(1/K)

    # get percent class informative centroids
    def get_ci(self, phase, index = 0, vis=False):

        # hold results here
        score = [0 for l in range(self.num_layers)]
        score_pc = [np.zeros((self.num_classes,)) for l in range(self.num_layers)]
        score_multi = [0 for l in range(self.num_layers)]

        # for each layer
        for l in range(self.num_layers):

            # for each centroid
            for j in range(len(self.cent_g[index][l])):

                # increase score if ci
                if max(self.cent_g[index][l][j]) > self.rho_task: #and np.sort(self.cent_g[index][l][j])[-2] <= 0.5 * max(self.cent_g[index][l][j]):
                    score[l] += 1
                
                if len(np.where(self.cent_g[index][l][j, :] > self.rho_task)) > 1:
                    score_multi[l] += 1

                for k in range(self.num_classes):
                    if self.cent_g[index][l][j,k] > self.rho_task:
                        score_pc[l][k] += 1
            
            # calculate percent ci at layer
            score[l] /= len(self.cent_g[index][l])
            score_pc[l] /= len(self.cent_g[index][l])
            score_multi[l] /= len(self.cent_g[index][l])

        
        return np.asarray(score), np.asarray(score_pc), np.asarray(score_multi)

    # given labeled data, associate class information with stam centroids
    def supervise(self, data, labels, phase, experiment_params, l_list = None, index = 0, image_ret=False, vis=True):
        
        # process inputs
        num_data = len(data)

        # get centroids for classification
        self.cents_ltm = []
        self.class_ltm = []

        for l in range(self.num_layers):
            if self.layers[l].num_ltm > 0:
                self.cents_ltm.append(self.layers[l].get_ltm_centroids())
                self.class_ltm.append(self.layers[l].get_ltm_classes())
            else:
                self.cents_ltm.append(self.layers[l].get_stm_centroids())
                self.class_ltm.append(self.layers[l].get_stm_classes())

        # this is repeat of self.setTask which is kept for scenario
        # where labeled data is NOT replayed
        if self.Nl_seen[index] == 0:
            self.D_sum[index] = [0 for l in range(len(l_list))]
            self.D[index] = [[] for l in range(len(l_list))]
            self.Fz[index] = [[] for l in range(len(l_list))]
            self.cent_g[index] = [[] for l in range(len(l_list))]    
        self.Nl_seen[index] += num_data

        # supervision per layer
        for l_index in range(len(l_list)):

            # get layer index from list of classification layers
            l = l_list[l_index]
        
            # get layer centroids
            centroids = self.cents_ltm[l]
            num_centroids = int(len(centroids))
            
            # get value of D for task
            # we use D to normalize distances wrt average centroid-patch distance
            for i in range(num_data):
            
                # get input to layer l
                x_i = data[i]
                for l_ in range(l):
                    x_i = self.layers[l_].forward(x_i, None, update = False)
            
                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                shape = patches.shape
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, _, _] = self.layers[l].scale(xp)
                
                # calculate and save distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind = np.argmin(d, axis = 1)
                self.D_sum[index][l_index] += np.sum(d[range(shape[0]),close_ind]) / shape[0]

            # final D calculation    
            self.D[index][l_index] = self.D_sum[index][l_index] / self.Nl_seen[index]
                       
            # this holds sum of exponential "score" for each centroid for each class
            sum_fz_pool = np.zeros((num_centroids, self.num_classes))


            # this code is relevant if we are not replaying labeled data
            ncents_past = len(self.Fz[index][l_index])
            if ncents_past > 0:
                sum_fz_pool[:ncents_past,:] = self.Fz[index][l_index]

            # for each image
            for i in range(num_data):
            
                # get input to layer l
                x_i = data[i]
                for l_ in range(l):
                    x_i = self.layers[l_].forward(x_i, None, update = False)
            
                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                shape = patches.shape
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, _, _] = self.layers[l].scale(xp)
                
                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)

                # get distance of *matched* centroid of each patch
                close_ind = np.argmin(d, axis = 1)
                dist = (d[range(shape[0]),close_ind])

                # get exponential distance and put into sparse array with same shape as 
                # summed exponential scores if we have two centroid matches in same 
                # image, only save best match
                td = np.zeros(d.shape)
                td[range(shape[0]),close_ind] = np.exp(-1*dist/self.D[index][l_index])
                fz = np.amax(td, axis = 0)
                
                # update sum of exponential "score" for each centroid for each class
                sum_fz_pool[:, int(labels[i])] += fz

            # save data scores and calculate g values as exponential "score" normalized 
            # accross classes (i.e. score of each centroid sums to 1)
            self.Fz[index][l_index] = sum_fz_pool    
            self.cent_g[index][l_index] = np.copy(sum_fz_pool)

            for j in range(num_centroids):
                self.cent_g[index][l_index][j,:] = self.cent_g[index][l_index][j,:] \
                    / (np.sum(self.cent_g[index][l_index][j,:]) + 1e-5)


    # call classification function
    def classify(self, data, phase, c_type, index, experiment_params):
        
        if c_type == 'hierarchy-vote':
            labels = self.topDownClassify(data, index, experiment_params, vis=True, phase=phase)
        
        return labels
    
    # stam primary classification function - hierarchical voting mechanism
    def topDownClassify(self, data, index, experiment_params, vis=False, phase=None):
    
        # process inputs and init return labels
        num_data = len(data)
        labels = -1 * np.ones((num_data,))

        # for each data
        for i in range(num_data):


            # get NN centroid for each patch
            close_ind = []
            close_distances = []
            for l in range(self.num_layers):

                # get ltm centroids at layer
                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))

                # get input to layer
                x_i = data[i]
                for l_ in range(l): x_i = self.layers[l_].forward(x_i, None, update = False)

                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, shifts, scales] = self.layers[l].scale(xp)

                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind.append(np.argmin(d, axis = 1))
                close_distances.append(np.min(d, axis = 1))
            
            
            # get highest layer containing at least one CIN centroid
            l = self.num_layers-1
            found_cin = False
            while l > 0 and not found_cin:
            
                # is there at least one CIN centroid?
                if np.amax(self.cent_g[index][l][close_ind[l]]) >= self.rho_task:
                    found_cin = True
                else:
                    l -= 1
            l_cin = l
                        
            # classification
            #
            # vote of each class for all layers
            wta_total = np.zeros((self.num_classes,)) + 1e-3

            # for all cin layers
            layer_range = range(l_cin+1)
            percent_inform = []
            for l in layer_range:
                # vote of each class in this layer
                wta = np.zeros((self.num_classes,))

                # get max g value for matched centroids
                votes_g = np.amax(self.cent_g[index][l][close_ind[l]], axis = 1)

                # nullify vote of non-cin centroids
                votes_g[votes_g < self.rho_task] = 0

                
                a = np.where(votes_g > self.rho_task)
                percent_inform.append(len(a[0])/ len(votes_g))      

                # calculate per class vote at this layer
                votes = np.argmax(self.cent_g[index][l][close_ind[l]], axis = 1)
                for k in range(self.num_classes):
                    wta[k] = np.sum(votes_g[votes == k])

                # add to cumalitive total and normalize
                wta /= len(close_ind[l])
                
                wta_total += wta

            # final step
            labels[i] = np.argmax(wta_total)
                 
        return labels

    def embed_centroid(self, X, layer, cents_ltm, experiment_params):

        normalization = experiment_params

        centroids = cents_ltm.reshape(len(cents_ltm), -1)

        X_ = np.zeros((X.shape[0], len(cents_ltm)))

        for i, x in enumerate(X):
            # ltm centroids were determined in 'supervise' function
            # get input to layer l
            x_layer = layer.forward(x, None, update=False)

            # extract patches
            patches = layer.extract_patches(x_layer)
            patches = patches.reshape(layer.num_RFs, -1)
            patches, _, _ = layer.scale(patches)

            # compute distance matrix
            d_mat = smart_dist(patches, centroids)

            # get indices of closest patch to each centroid and accumulate average 
            # closest-patch distances
            close_cent_dists = np.min(d_mat, axis=0)

            # Calculate normalization constant D_l
            D_l = np.sum(d_mat) / d_mat.shape[0] * d_mat.shape[1]

            if normalization:
                X_[i,:] = np.exp((-1 * close_cent_dists) / D_l)
            else:
                X_[i,:] = close_cent_dists


        return X_

    def embed_patch(self, X, layer, cents_ltm, experiment_params, cnt=False):

        method = experiment_params

        # patch size and num_features calculations
        p = layer.recField_size
        n_cols = len(layer.extract_patches(X[0]))


        if cnt:
            X_ = np.zeros((X.shape[0], n_cols), dtype=str)
        else:
            X_ = np.zeros((X.shape[0], n_cols), dtype=int)
        

        for i, x in enumerate(X):
            # get input to layer l
            x_layer = layer.forward(x, None, update=False)

            # extract patches
            patches = layer.extract_patches(x_layer)
            patches = patches.reshape(layer.num_RFs, -1)
            patches, _, _ = layer.scale(patches)

            d_mat = smart_dist(patches, cents_ltm)

            # get indices of closest patch to each centroid and accumulate average 
            # closest-patch distances
            close_patch_inds = np.argmin(d_mat, axis=1)
            
            if cnt:
                counts = np.bincount(close_patch_inds, minlength=max(close_patch_inds))
            
                temp_i = []
                for ind, count in enumerate(counts):
                    if count == 0:
                        continue
                    else:
                        for c in range(count):
                            temp_i.append("{}.{}".format(ind, c))

                X_[i] = np.array(temp_i)
            else:
                X_[i] = close_patch_inds
        
        print("Got Jaccard Embedding")
        return X_

    def jaccard(self, x, y):
        x = set(x)
        y = set(y)
        val = len(x.intersection(y)) / len(x.union(y))
        
        if val == None:
            return 0
        else:
            return val
    
    # cluster
    def cluster(self, X, Y, phase_num, task_num, dataset, num_classes,
                experiment_params, cluster_method='kmeans', 
                accuracy_method='acc', k_scale=1, eval_layers=[]):
        # returns total and per-class accuracy... (float, 1 by k numpy[float])
        print('Clustering Task Started...')

        embedding_mode, mode_name = experiment_params

        print("Embedding Mode: ", embedding_mode)
        print("Experiment Name: ", mode_name)
        print("Cluster Method: ", cluster_method)
        similarity_matrix = np.zeros(10)
        if embedding_mode == 0:
            X_1 = self.embed_centroid(X, self.layers[0], self.cents_ltm[0], True)
            X_2 = self.embed_centroid(X, self.layers[1], self.cents_ltm[1], True)
            X_3 = self.embed_centroid(X, self.layers[2], self.cents_ltm[2], True)
        
            embeddings = np.concatenate((X_1, np.concatenate((X_2, X_3), axis=1)), axis=1)

        elif embedding_mode == 1:
            X_3 = self.embed_patch(X, self.layers[2], self.cents_ltm[2], None, False)

            embeddings = X_3

            similarity_matrix = pairwise_distances(embeddings, embeddings, metric=self.jaccard)

        
        k = int(np.unique(Y).shape[0] * k_scale)
        accu_total = 0
        accu_perclass = np.zeros(num_classes, dtype=np.float64)

        # Clustering Predictions
        if cluster_method == 'kmeans':
            cluster_preds = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                                   max_iter=300, verbose=0).fit_predict(embeddings)
        elif cluster_method == 'spectral':
            try:
                cluster_preds = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=10,
                                               assign_labels='discretize').fit_predict(similarity_matrix)
            except Exception as e:
                cluster_preds = np.zeros(len(similarity_matrix))

        # Accuracy of Clustering
        if accuracy_method == 'purity':
            size = k
            print("Size ", size)
            cluster_counts = np.zeros((size, int(k/k_scale)))
            cluster_sizes = np.zeros(size)
            correct = np.zeros(size)
            total = np.zeros(size)
            cluster_indicies = [[] for i in range(num_classes)]

            for i in range(size):
                cluster_i = np.argwhere(cluster_preds == i).flatten()  # indexes of cluster i
                cluster_sizes[i] = len(cluster_i)
                cluster_counts[i,:] = np.bincount(Y[cluster_i], minlength=int(k/k_scale))

                # compute accuracy
                cluster_class = np.argmax(cluster_counts[i, :])
                correct[i] = cluster_counts[i, :].max()
                total[i] = cluster_counts[i, :].sum()
                cluster_indicies[cluster_class].append(i)

            for j in range(num_classes):
                if sum(total[cluster_indicies[j]]) > 0:
                    accu_perclass[j] = sum(correct[cluster_indicies[j]]) \
                        / sum(total[cluster_indicies[j]]) * 100
                else:
                    accu_perclass[j] = 0

            accu_total = sum(correct) / sum(total) * 100

        elif accuracy_method == 'acc':
            accu_total, accu_perclass = eval_acc(Y, cluster_preds)

        return accu_total, accu_perclass

    # newly added knn evaluator
    def knn(self, X, Y, X_KNN, Y_KNN, experiment_params):
        # returns total and per-class accuracy... (float, 1 by k numpy[float])
        print('KNN Task Started...')

        embedding_mode, mode_name = experiment_params

        print("Embedding Mode: ", embedding_mode)
        print("Experiment Name: ", mode_name)

        if embedding_mode == 0:
            X_1 = self.embed_centroid(X, self.layers[0], self.cents_ltm[0], True)
            X_2 = self.embed_centroid(X, self.layers[1], self.cents_ltm[1], True)
            X_3 = self.embed_centroid(X, self.layers[2], self.cents_ltm[2], True)

            X_embeddings = np.concatenate((X_1, np.concatenate((X_2, X_3), axis=1)), axis=1)

            X_KNN_1 = self.embed_centroid(X_KNN, self.layers[0], self.cents_ltm[0], True)
            X_KNN_2 = self.embed_centroid(X_KNN, self.layers[1], self.cents_ltm[1], True)
            X_KNN_3 = self.embed_centroid(X_KNN, self.layers[2], self.cents_ltm[2], True)

            X_KNN_embeddings = np.concatenate((X_KNN_1, np.concatenate((X_KNN_2, X_KNN_3), axis=1)), axis=1)

        elif embedding_mode == 1:
            X_3 = self.embed_patch(X, self.layers[2], self.cents_ltm[2], None, False)

            X_embeddings = X_3

            X_KNN_3 = self.embed_patch(X, self.layers[2], self.cents_ltm[2], None, False)

            X_KNN_embeddings = X_KNN_3

        # Call the KNN evaluator
        knn_acc = eval_knn(X_embeddings, Y, X_KNN_embeddings, Y_KNN)

        return knn_acc


    # save STAM visualizations    
    def save_visualizations(self, save_dir, phase):
        
        # Cent count
        plt.figure(figsize=(6,3)) 
        for l in range(self.num_layers):
            y = np.asarray(self.layers[l].ltm_history)
            x = np.arange(len(y))
            plt.plot(x, y, label = 'layer ' + str(l+1))
        plt.ylabel('LTM Count', fontsize=12)
        plt.xlabel('Unlabeled Images Seen', fontsize=12) 
        plt.title('LTM Centroid Count History', fontsize=14)
        plt.legend(loc='upper left', prop={'size': 8})
        plt.grid()
        plt.tight_layout()
        plt.savefig(smart_dir(save_dir+'cent_plots')+'ltm_count.png', format='png', dpi=200)
        plt.close()
        for l in range(self.num_layers):
            np.savetxt(smart_dir(save_dir+'ltm_csvs') + 'layer-' + str(l+1) + '_ci.csv', 
                       self.layers[l].ltm_history, delimiter=',')

        if not self.nd_fixed:
            
            # confidence interval
            plt.figure(figsize=(6,3))  
            p = np.asarray([0, 25, 50, 75, 90, 100])
            for l in range(self.num_layers):
                dd = np.asarray(self.layers[l].filo_d.getValues())            
                y = np.percentile(dd, p)
                x = np.arange(len(dd)) / len(dd)
                plt.plot(x, dd, label = 'layer ' + str(l+1))
                plt.plot(p/100., y, 'ro')
                plt.axhline(y=self.layers[l].cut_d, color='r', linestyle='--')
            plt.xticks(p/100., map(str, p))
            plt.ylabel('Distance', fontsize=12)
            plt.xlabel('Percentile', fontsize=12)
            plt.title('Distribution of Closest Matching Distance', fontsize=14)
            plt.legend(loc='lower right', prop={'size': 8})
            plt.grid()
            plt.tight_layout()
            plt.savefig(smart_dir(save_dir+'cent_plots')+'d-thresh.png', format='png', dpi=200)
            plt.grid()
            plt.close()

            # D threshold
            plt.figure(figsize=(6,3)) 
            for l in range(self.num_layers):
                y = np.asarray(self.layers[l].dthresh_history)
                x = np.arange(len(y))
                plt.plot(x, y, label = 'layer ' + str(l+1))
            plt.ylabel('ND Distance', fontsize=12)
            plt.xlabel('Unlabeled Images Seen', fontsize=12)  
            plt.gca().set_ylim(bottom=0)
            plt.title('Novelty Detection Threshold History', fontsize=14)
            plt.legend(loc='upper left', prop={'size': 8})
            plt.grid()
            plt.tight_layout()
            plt.savefig(smart_dir(save_dir+'cent_plots')+'d-thresh-history.png', 
                        format='png', dpi=200)
            plt.close()
            
        #self.save_reconstructions(save_dir)
        #self.detailed_classification_plots(save_dir)

    
    def detailed_classification_plots(self):
        index = 0
        labels = -1 * np.ones((len(self.sample_images),))
        
        for i in range(len(self.sample_images)):

            close_ind = []
            for l in range(self.num_layers):

                # get ltm centroids at layer
                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))

                # get input to layer
                x_i = self.sample_images[i]
                for l_ in range(l): x_i = self.layers[l_].forward(x_i, None, update = False)

                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, shifts, scales] = self.layers[l].scale(xp)

                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind.append(np.argmin(d, axis = 1))
            
            
            # get highest layer containing at least one CIN centroid
            l = self.num_layers-1
            found_cin = False
            while l > 0 and not found_cin:
            
                # is there at least one CIN centroid?
                if np.amax(self.cent_g[index][l][close_ind[l]]) >= self.rho_task:
                    found_cin = True
                else:
                    l -= 1
            l_cin = l
                        
            # classification
            #
            # vote of each class for all layers
            wta_total = np.zeros((self.num_classes,)) + 1e-3

            # for all cin layers
            layer_range = range(l_cin+1)
            percent_inform = []
            layer_wta = []
            for l in layer_range:
                # vote of each class in this layer
                wta = np.zeros((self.num_classes,))

                # get max g value for matched centroids
                votes_g = np.amax(self.cent_g[index][l][close_ind[l]], axis = 1)

                # nullify vote of non-cin centroids
                votes_g[votes_g < self.rho_task] = 0
                a = np.where(votes_g > self.rho_task)
                percent_inform.append(len(a[0])/ len(votes_g))      

                # calculate per class vote at this layer
                votes = np.argmax(self.cent_g[index][l][close_ind[l]], axis = 1)
                for k in range(self.num_classes):
                    wta[k] = np.sum(votes_g[votes == k])

                # add to cumalitive total
                wta /= len(close_ind[l])
                layer_wta.append(wta)
                wta_total += wta
                    
            # final step
            labels[i] = np.argmax(wta_total)
            
            # Visualizing Patches and Centroids
            for l in range(self.num_layers):
                nrows = ncols = int(np.sqrt(self.layers[l].num_RFs) / 2)
                rf_size = self.layers[l].recField_size
                
                plt.close()
                fig = plt.figure(figsize=(9,11))

                # First 3
                out_im, out_im_2 = self.layers[l].create_reconstruction(self.sample_images[i], self.sample_labels[i])
                ax1 = fig.add_axes([0.1, 0.75, 0.2, 0.2])

                ax2 = fig.add_axes([0.35, 0.75, 0.2, 0.2])

                ax3 = fig.add_axes([0.63, 0.83, 0.30, 0.12])

                ax1.imshow(out_im_2.squeeze())
                ax1.set_title('Patches')
                ax1.axis('off')
                
                ax2.imshow(out_im.squeeze())
                ax2.set_title('Matched Centroids')
                ax2.axis('off')
                
                ax3.bar(np.arange(self.num_classes), layer_wta[l])
                ax3.set_xticks(np.arange(self.num_classes))
                ax3.set_xticklabels(self.class_labels, rotation='vertical')
                ax2.tick_params(axis='y', which='major', labelsize=10)
                ax3.set_title('Layer {} Vote  (1/K + Gamma): {}'.format(l, self.rho_task))
                ax3.axis('on')
                
                patches = self.layers[l].extract_patches(x_i)
                xp_2 = patches.reshape(self.layers[l].num_RFs, -1)
                xp_2 = self.layers[l].scale(xp_2)[0].reshape(self.layers[l].num_RFs, -1)

                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))
                cp = centroids.reshape(num_centroids, -1)

                for p in range(4):
                    for j in range(5):
                        ax1 = fig.add_axes([0.08 + .17*j, 0.57 - .15*p, 0.05, 0.05])
                        ax2 = fig.add_axes([0.16 + .17*j, 0.57 - .15*p, 0.05, 0.05])
                        ax3 = fig.add_axes([0.08 + .17*j, 0.65 - .15*p, .13, .05])

                        ax1.set_title('Patch')
                        ax1.imshow(xp_2[int((p*ncols*2) + (2*j))].reshape(rf_size, rf_size, self.num_c).squeeze())
                        ax1.axis('off')

                        if np.max(self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]]) > self.rho_task:
                            ax2.set_title('Centroid', color='g')
                        else:
                            ax2.set_title('Centroid', color='r')

                        ax2.imshow(cp[close_ind[l][int((p*ncols*2) + (j*2))]].reshape(rf_size, rf_size, self.num_c).squeeze())
                        ax2.axis('off')
                        
                        vote = np.argmax(self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]])
                        ax3.set_title('Vote: {}'.format(self.class_labels[vote]))
                        ax3.bar(np.arange(self.num_classes), self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]])
                        ax3.axes.get_xaxis().set_ticks([])
                        ax3.tick_params(axis='y', which='major', labelsize=6)
                        #ax3.axis('off')
                    

                fig.suptitle('True Class: {}   Predicted Class: {}   Layer{}'.format(self.class_labels[int(self.sample_labels[i])], self.class_labels[int(labels[i])], l))
                plt.savefig(smart_dir(self.plot_directory + '/phase_{}/ex_{}/'.format(self.phase, i)) + 'layer_{}_vote.png'.format(l))    
                plt.close()

        return labels


    def save_reconstructions(self, save_dir):
        
        for i in range(len(self.sample_images)):
            for layer in range(self.num_layers):
                out_im, out_im_2 = self.layers[layer].create_reconstruction(self.sample_images[i], self.sample_labels[i])
                plt.figure(figsize=(6,3))
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(out_im.squeeze())
                ax2.imshow(out_im_2.squeeze())
                plt.title('Class : {}'.format(self.sample_labels[i]))
                plt.savefig(smart_dir(save_dir + '/reconstructions/layer_{}'.format(layer)) + 'image_{}'.format(i))
                plt.close()

    # scale image based on normalization
    def scaleImage(self, im_show):

        im_show = np.squeeze((im_show - self.im_scale[0]) / (self.im_scale[1] - self.im_scale[0]))
        im_show[im_show>1] = 1
        im_show[im_show<0] = 0
        return im_show


    def pick_sample_images(self, x_test, y_test, skip=20):

        self.sample_images = []
        self.sample_labels = []
        self.skip = skip

        k = np.unique(y_test)

        for i, im in enumerate(x_test):
            if i % self.skip == 0:
                self.sample_images.append(x_test[i])
                self.sample_labels.append(y_test[i])
                plt.imshow(im.reshape(self.im_size, self.im_size, self.num_c).squeeze())
                plt.title('Class: {}'.format(y_test[i]))
                plt.savefig(smart_dir(self.plot_directory + '/sample_imgs/') + 'sample_image_{}'.format(int(i / self.skip)))
                plt.close()


    def stm_eviction_plot(self):
        plt.close()
        
        for l in range(self.num_layers):
            plt.figure(figsize=(40,6))
            data = self.layers[l].eviction_tracker
            x = np.arange(len(data))
            plt.boxplot(data, positions=x, showfliers=False)


            plt.savefig(smart_dir(self.plot_directory) + '/eviction_layer_{}.png'.format(l))
            plt.close()
