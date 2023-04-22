import numpy as np
import math
import bisect
import cv2
import sys
import time
import code

from numpy.lib import stride_tricks
from sklearn.utils import resample
from collections import deque
from core.utils import *
from core.distance_metrics import *
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# filo sorted class - used to estimate
# distance distribution for novelty detection
class FILO:

    def __init__(self, capacity):

        self.cap = capacity
        self.values = []
        self.ages = []

    def insert(self, x):

        # insert new item into list
        pos = bisect.bisect(self.values, x)
        self.values.insert(pos, x)
        self.ages.insert(pos, -1)
        self.ages = [x+1 for x in self.ages]

        # if exceed capacity, remove oldest item
        if len(self.values) > self.cap:
            pos = np.argmax(self.ages)
            self.values.pop(pos)
            self.ages.pop(pos)

    def perc(self, percent):

        k = (len(self.values)-1) * percent
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return self.values[int(k)]
        d0 = self.values[int(f)] * (c-k)
        d1 = self.values[int(c)] * (k-f)
        return d0+d1

    def getValues(self):

        return np.asarray(self.values)

def perc(array, percent):

    k = (len(array)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return array[int(k)]
    d0 = array[int(f)] * (c-k)
    d1 = array[int(c)] * (k-f)
    return d0+d1


# The STAM Layer Class
class Layer:

    def __init__(self, img_size, ch, name, recField_size, stride, construct_stride, 
                 num_cents, alpha, ltm_alpha, beta, theta, wta, im_scale, scale_flag, 
                 seed, kernel, expected_features, nd_fixed, num_images_init, plot_dir, vis_train):
        
        # ViS
        self.vis = vis_train
        self.plot_directory = plot_dir

        # general layer parameters
        self.name = name # name of layer
        self.img_size = img_size # img size
        self.ch = ch # num channels
        self.recField_size = recField_size # size of rf i.e. patch
        self.stride = stride # stam stride for clustering
        self.construct_stride = construct_stride # stam stride for reconstructing image
        
        self.num_cents = num_cents # number of centroids in this layer at any time
        self.num_cents_init = num_cents # number of centroids in this layer for init purposes
        self.num_images_init = num_images_init
        self.expected_features = expected_features

        self.alpha = alpha # centroid learning rate (stm)
        self.ltm_alpha = ltm_alpha # centroid learning rate (ltm) - often zero
        self.beta = beta # percentile of distance distribution for novelty detection
        self.theta = theta # stm activations required for ltm
        self.WTA = wta # flag for winner-take-all reconstruction
        self.im_scale = im_scale # scale of images (based on normalization) for visualization purposes
        self.scale_flag = scale_flag # flag for per-patch normalization
        self.kernel = kernel # blurring kernel size for input image if using k-means

        # calculate number of receptive fields total at cluster level
        self.num_RFs = int(np.power(np.ceil((img_size - self.recField_size) / self.stride) + 1, 2))
        # calculate RFs per axis at cluster level
        self.num_RFs_axis = int(np.ceil((img_size - self.recField_size) / self.stride) + 1)
        # calculate number of receptive fields total at reconstruction level
        self.num_RFs_construct = int(np.power(np.ceil((img_size - self.recField_size) / self.construct_stride) + 1, 2)) 
        # calculate RFs per axis at reconstruction level
        self.num_patch_axis = int(np.ceil((img_size - self.recField_size) / 1) + 1)

        # init param
        self.num_init = 0 # init counter - reach self.num_cents_init to finish initialization
        self.num_ltm = 0 # num centroids in ltm

        # centroids
        self.centroids = np.zeros((self.num_cents, self.recField_size * self.recField_size * self.ch))
        self.centroid_classes = np.zeros(self.num_cents, int)
        
        # holds the most recent time centroid has been active
        self.centroid_recency = np.zeros((self.num_cents,),float)

        self.current_image = 0


        # holds the number of times centroid has been selected
        self.centroid_selections = 1 * np.ones((self.num_cents,),int)
        self.centroid_non_selections =  np.zeros((self.num_cents,),int)
        
        # holds the centroid age
        self.centroid_ages = 1 * np.ones((self.num_cents,),int) 
        
        # hold centroid frequency
        self.centroid_frequency = np.zeros((self.centroids.shape[0],),float)

        # hold centroid stm_count
        self.centroid_stm = np.zeros((self.centroids.shape[0],),float)
        self.ltm_match = -1 * np.ones((self.num_cents,),int)
        
        # holds centroid statistics
        self.nd_fixed = nd_fixed
        self.cut_d = -1 # novelty detection threshold
        if self.nd_fixed:
            self.phase_0_images = []
            self.num_sample_patches = 10
            self.init_patches = np.zeros((self.num_sample_patches * self.num_images_init, self.recField_size * self.recField_size * self.ch))
        else:
            self.filo_size = 1000 # size of filo queue to estimate nd threshold
            self.d_samples_per_image = 10
            self.filo_d = FILO(self.filo_size) # filo queue to estimate nd threshold

        # for visualization
        self.ltm_history = [] # history of ltm count
        self.stm_delete_history = [] # history of stm deletions
        self.stm_delete_queue = deque(maxlen=500) # queue to estimate smooth delete history
        self.stm_create_history = [] # history of stm novelty detections
        self.stm_create_queue = deque(maxlen=500) # queue to estimate smooth create history
        self.dthresh_history = [] # history of nd threshold
        self.novelty_total = 0 # total number of novelties
        self.delete_total = 0 # total number of deletions
        self.novelty_last_100 = 0

        # hold samples for visualization purposes
        self.d_sample = 0
        self.d_sample_hold = []
        self.d_sample_x = []
        self.delete_n = []
        self.im_seen = 0
        
        # get indexes of image patch locations
        self.RF_i_array = np.zeros((self.num_RFs_construct,2), int)   
        self.RF_j_array = np.zeros((self.num_RFs_construct,2), int)
        k = 0             
        for i in range(0, int(np.sqrt(self.num_RFs_construct))):
            for j in range(0, int(np.sqrt(self.num_RFs_construct))):
                if i < int(np.sqrt(self.num_RFs_construct)) - 1:
                    startI = i * construct_stride
                else:
                    startI = self.img_size - recField_size
                endI = startI + recField_size

                if j < int(np.sqrt(self.num_RFs_construct)) - 1:
                    startJ = j * construct_stride
                else:
                    startJ = self.img_size - recField_size
                endJ = startJ + recField_size
                
                self.RF_i_array[k,:] = [startI, endI]
                self.RF_j_array[k,:] = [startJ, endJ]
                k+=1               

    # get ALL patches from image    
    def extract_patches(self, im):
        shape = (self.num_patch_axis, self.num_patch_axis, self.recField_size, self.recField_size, self.ch)
        strides = (im.strides[0], im.strides[1], im.strides[0], im.strides[1], im.strides[2])
        patches = stride_tricks.as_strided(im, shape=shape, strides=strides)
        patches = patches[range(0,self.num_patch_axis,self.stride),:][:,range(0,self.num_patch_axis,self.stride),:,:,:]
        
        return patches.reshape(self.num_RFs, self.recField_size, self.recField_size, self.ch)

    # scale patches if flag set
    def scale(self, data):
        if self.scale_flag:
            shift = np.mean(data, axis = 1)[:,None]
            data -= shift
            scale = np.std(data, axis = 1)[:,None]
            scale[scale == 0] = 1
            data_out = data / scale
            return [data_out, shift, scale]
        else:
            return [data, np.zeros((data.shape[0], 1)), np.ones((data.shape[0], 1))]

    # foward pass
    def forward(self, img, label, update = True):

        # Blur if kernel > 1
        if self.kernel > 1:
            kernel = np.ones((self.kernel,self.kernel), np.float32) / (self.kernel)**2
            img_x = cv2.filter2D(img, -1, kernel)[:,:,None]
        else:
            img_x = img

        # update rule
        if update:

            if self.nd_fixed:

                if self.im_seen < self.num_images_init:        
                
                    # Collect init image
                    self.phase_0_images.append(img_x)

                    # images seen
                    self.im_seen += 1
                    
                    # after init, do bootstrap estimation of closest d!
                    if self.im_seen == self.num_images_init:

                        ind = 0
                        for imgg in self.phase_0_images:
                            xp = self.extract_patches(imgg).reshape(self.num_RFs, -1)
                            [xp, shifts, scales] = self.scale(xp)

                            xp = resample(xp, n_samples = 10)
                            
                            for pat in xp:
                                self.init_patches[ind] = pat
                                ind += 1
                        
                        print("Layer: ", self.name)
                        print("Ind ", ind)
                        kmeans = KMeans(n_clusters=self.expected_features).fit(self.init_patches)
                        
                        for c, cluster_cent in enumerate(kmeans.cluster_centers_):
                            self.centroids[c] = cluster_cent
                            if self.vis:
                                plt.imshow(cluster_cent.reshape((self.recField_size, self.recField_size, self.ch)).squeeze())
                                plt.savefig(smart_dir(self.plot_directory + '/init_centroids/{}/'.format(self.name)) + 'centroid_{}.png'.format(c))
                                plt.close()

                        d = smart_dist(self.init_patches, self.centroids[:self.expected_features])
                        
                        d_min = np.min(d, axis = 1)

                        plt.hist(d_min, bins='auto')
                        plt.title('Init Distance Distribution for Fixed Novelty Threshold\nLayer: {}'.format(self.name))
                        plt.xlabel('Distance')
                        plt.ylabel('Number of Patches')
                        plt.savefig(smart_dir(self.plot_directory + '/init_centroids/{}/'.format(self.name)) + 'distribution.png')
                        plt.close()

                        self.cut_d = perc(d_min, self.beta)
                        print("Novelty_Threshold: ", self.cut_d)

                    return img
                
             # Dynamic ND Threshold Init
            else:

                # if layer not init. then directly set patches to centroids
                if self.num_init < self.num_cents_init: 

                    # images seen
                    self.im_seen += 1

                    # get patches
                    xp = self.extract_patches(img_x).reshape(self.num_RFs, -1)

                    [xp, shifts, scales] = self.scale(xp)           

                    # sample 10
                    num_set = 10

                    # get samples
                    self.centroids[self.num_init:self.num_init+num_set] = resample(xp, n_samples = num_set)
                    self.centroid_classes[self.num_init:self.num_init+num_set] = int(label)
                    num_novelties = num_set
                    self.num_init += num_set

                    # after init, do bootstrap estimation of closest d!
                    if self.num_init == self.num_cents_init:
                        num_sample = min(self.num_cents_init, self.filo_size)
                        cent_samples = np.random.choice(self.num_cents, num_sample)
                        for j in cent_samples:
                            d_sample = np.amin(smart_dist(self.centroids[j][None,:], self.centroids[np.arange(self.num_cents)!=j]))
                            self.filo_d.insert(d_sample)
                        self.cut_d = self.filo_d.perc(self.beta)
                            
                    # update history
                    self.ltm_history.append(0)
                    self.stm_delete_history.append(0)
                    self.stm_create_history.append(0)

                    # cut d
                    self.dthresh_history.append(self.cut_d)
                    # return img
                    return img
            
            
            # get patches
            xp = self.extract_patches(img_x).reshape(self.num_RFs, -1)
            [xp, shifts, scales] = self.scale(xp)

            # find closest centroid for each patch
            d = smart_dist(xp, self.centroids)
            d_min = np.amin(d, axis = 1)
            
            self.close_ind = np.argmin(d, axis = 1)
            self.update_index = np.copy(self.close_ind)

            #####################
            # novelty detection #
            #####################
            
            # get indexes of centroids which pass novelty criterion
            novel_threshold = d_min - self.cut_d
            novel_index = np.where(novel_threshold > 0)[0]     
            num_novelties = len(novel_index)

            # this will determine which centroids chosen for novelty detection
            nd_centroid_recency = np.copy(self.centroid_recency)
            
            # do NOT consider chosen centroids in update pass!
            nd_centroid_recency[self.close_ind] = 0

            # do NOT consider chosen centroids in ltm!
            nd_centroid_recency[np.where(self.centroid_stm < -1)] = 0
            
            for ni in novel_index: 
                
                # get lowest frequently used centroid
                forget_ind = np.argmax(nd_centroid_recency)
                nd_centroid_recency[forget_ind] = 0

                # create new centroid
                self.centroids[forget_ind] = xp[ni]
                self.centroid_classes[forget_ind] = int(label)

                self.centroid_recency[forget_ind] = 0
                self.centroid_selections[forget_ind] = 1
                self.centroid_non_selections[forget_ind] = 0
                self.centroid_ages[forget_ind] = 0
                self.centroid_frequency[forget_ind] = 1
                self.centroid_stm[forget_ind] = 0
                self.update_index[ni] = forget_ind
                
                # init stats
                self.ltm_match[forget_ind] = -1                     
                            
            # only update a centroid at most once!
            patch_update = [True for i in range(len(xp))] 
            for i in range(len(xp)):
                
                # check if this centroid is selected multiple times
                ind = self.update_index[i]
                neighbors = np.where(self.update_index == ind)[0]

                # only update if closest selection!
                if len(neighbors) > 1:
                    ndist = d[:,ind]
                    if not np.argmin(ndist) == i:
                        patch_update[i] = False

            # centroid update of closest selection
            for i in range(len(xp)):
                if patch_update[i]:
                    ind = self.update_index[i]
                    if self.centroid_ages[ind] > 0 and self.centroid_stm[ind] >= 0: 
                        self.centroids[ind] = (1 - self.alpha) * self.centroids[ind] + self.alpha * xp[i] 
                    elif self.centroid_ages[ind] > 0 and self.centroid_stm[ind] < 0:
                        self.centroids[ind] = (1 - self.ltm_alpha) * self.centroids[ind] + self.ltm_alpha * xp[i]
                        if self.ltm_match[ind] >= 0:
                            self.centroids[self.ltm_match[ind]] = (1 - self.ltm_alpha) * self.centroids[self.ltm_match[ind]] + self.ltm_alpha * xp[i]

            if not self.nd_fixed:
                # centroid stats update - randomly select one distance to update with
                sampled_d = np.random.choice(d_min, self.d_samples_per_image)
                for sample in sampled_d:
                    self.filo_d.insert(sample)

                self.cut_d = self.filo_d.perc(self.beta)

                # for visualization purposes
                self.d_sample_hold.append(sampled_d)
                self.d_sample_x.append(self.im_seen)
                self.dthresh_history.append(self.cut_d)

            # update counting properties
            update_index, update_counts = np.unique(self.update_index, return_counts=True) 

            # ltm matches - update corresponding ltm centroid if stm centroid is chosen with ltm copy
            ltm_copy = np.where(self.ltm_match[update_index] >= 0)[0]
            for ind in ltm_copy:
                update_index = np.append(update_index, [self.ltm_match[update_index[ind]]], axis = 0)
                update_counts = np.append(update_counts, [update_counts[ind]], axis = 0)

            # update centroid properties
            self.centroid_non_selections += 1
            self.centroid_non_selections[update_index] -= 1
            self.centroid_selections[update_index] += 1
            self.centroid_stm[update_index[self.centroid_stm[update_index] >= 0]] += 1
            self.centroid_recency += 1
            self.centroid_recency[update_index] = 0
            self.centroid_ages += 1
            self.centroid_frequency = self.centroid_selections / (self.centroid_selections + self.centroid_non_selections)
                        
            # check save in ltm
            save_ind = np.where(self.centroid_stm >= self.theta)[0]
            num_save = len(save_ind)

            if len(save_ind) > 0:
                
                # ltm
                self.centroid_stm[save_ind] = -1
                for ind in save_ind:
                        
                    # create new centroid - mark it as ltm
                    self.centroids = np.append(self.centroids, np.copy(self.centroids[ind])[None,:], axis = 0)
                    self.centroid_classes = np.append(self.centroid_classes, np.array([self.centroid_classes[ind]]), axis=0)
                    self.centroid_recency = np.append(self.centroid_recency, [self.centroid_recency[ind]], axis = 0)
                    self.centroid_selections = np.append(self.centroid_selections, [self.centroid_selections[ind]], axis = 0)
                    self.centroid_non_selections = np.append(self.centroid_non_selections, [self.centroid_non_selections[ind]], axis = 0)
                    self.centroid_ages = np.append(self.centroid_ages, [self.centroid_ages[ind]], axis = 0)
                    self.centroid_frequency = np.append(self.centroid_frequency, [self.centroid_frequency[ind]], axis = 0)
                    self.centroid_stm = np.append(self.centroid_stm, [-2], axis = 0)
                    self.ltm_match = np.append(self.ltm_match, [-1], axis = 0)
                    
                    # update ltm match
                    self.ltm_match[ind] = len(self.centroids) - 1   

            # memory updates
            self.num_ltm = len(np.where(self.centroid_stm < -1)[0])
            self.num_cents = len(self.centroids)
            self.ltm_history.append(self.num_ltm)
            self.stm_delete_queue.append(num_novelties)
            self.stm_delete_history.append(np.mean(self.stm_delete_queue))
            self.stm_create_queue.append(num_save)
            self.stm_create_history.append(np.mean(self.stm_create_queue))
    
        if update:
            self.current_image += 1

        return img
        
    # return ltm centroids
    def get_ltm_centroids(self):
        val = self.centroids[self.centroid_stm < -1]
        return val

    # return ltm class labels
    def get_ltm_classes(self):
        return self.centroid_classes[self.centroid_stm < -1]

    # return stm centroids
    def get_stm_centroids(self):
        return self.centroids[self.centroid_stm >= -1]

    # return stm class labels
    def get_stm_classes(self):
        return self.centroid_classes[self.centroid_stm >= -1]

    # scale image based on normalization - for visualization purposes
    def scaleImage(self, im_show, l=None):

        if self.scale_flag:
            im_show = im_show - np.amin(im_show)
            if np.amax(im_show) > 0: im_show /= np.amax(im_show)

        else:
            im_show = (im_show - self.im_scale[0]) / (self.im_scale[1] - self.im_scale[0])
        
        im_show[im_show>1] = 1
        im_show[im_show<0] = 0
        return np.squeeze(im_show)

    def create_reconstruction(self, x, y):

        patches = self.extract_patches(x).reshape(self.num_RFs, -1)

        [patches, _, _] = self.scale(patches)

        d = smart_dist(patches, self.centroids)
        out_ind = np.argmin(d, axis = 1)

        num_patches = math.floor(self.img_size / self.recField_size)

        patch_skip = int(self.recField_size / self.stride)

        shape = num_patches * self.recField_size

        out_im = np.zeros((shape, shape, self.ch))
        out_im_2 = np.zeros((shape, shape, self.ch))

        i = 0
        j = 0
        row_skip = math.floor(len(out_ind) / num_patches)
        for row in range(num_patches):
            i = row_skip * j
            for col in range(num_patches):
                out_im[row * self.recField_size : (row+1) * self.recField_size, col * self.recField_size : (col + 1) * self.recField_size, :] = self.centroids[out_ind[i]].reshape(self.recField_size, self.recField_size, -1)
                out_im_2[row * self.recField_size : (row+1) * self.recField_size, col * self.recField_size : (col + 1) * self.recField_size, :] = patches[i].reshape(self.recField_size, self.recField_size, -1)
                i += patch_skip
            j += 1

        return out_im , out_im_2
