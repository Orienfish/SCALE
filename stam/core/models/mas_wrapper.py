import math
import time
import quadprog
import numpy as np
import seaborn as sn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics


from core.utils import *
from torch.utils import data as data_utils
from core.models.networks import NetworkInNetwork
import torchvision.models as models

class MASWrapper():

    def __init__(self, configs):

        # extract scenario configs
        self.num_classes = configs['num_classes']
        self.class_labels = configs['class_labels']
        self.vis_train = configs['visualize_train']
        self.im_size = configs['im_size']
        self.num_c = configs['channels']
        self.seed = configs['seed']
        self.im_scale = configs['im_scale']
        self.scale_flag = configs['scale_flag']
        self.dataset = configs['dataset']
        self.num_phases = configs['num_phases']
        
        # extract cnn configs
        self.epochs = configs['epochs']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']


        # directories
        self.results_directory = configs['results_directory']
        self.plot_directory = configs['plot_directory']


        # initialize variables
        self.images_seen = 0
        self.phase_image_start = []

        # classifier dic
        self.classifiers = {}
        self.feat_mean = {}
        self.feat_var = {}

        self.ops = {}

        # torch related config
        if torch.cuda.is_available():
            print("Using GPU")
            self.dev = torch.device("cuda:0")
        else:
            print("Using CPU")
            self.dev = torch.device("cpu")


        # Task related config
        self.loss_history = []

        # initalize the model
        print("Creating Model")
        opt = {'num_classes': 4, 
               'num_stages': 4,
               'num_inchannels': self.num_c,
               'use_avg_on_conv3': False,
              }
        self.model = NetworkInNetwork(opt)
        
        self.model.to(self.dev)

        print("LR: ", self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # MAS Related
        self.omega_data_size = None
        self.reg_coef = configs['reg_coef']

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.omegas = {}
        self.past_parameters = {}

        for n, p in self.model.named_parameters():
            self.omegas[n] = p.clone().fill_(0)
            self.past_parameters[n] = p.clone()

    
    def calculate_loss(self, outputs, labels, check_flag=False):

        loss = self.criterion(outputs, labels)

        if self.current_task >= 1:
            reg_loss = 0
            for n, p in self.model.named_parameters():
                reg_loss += (self.omegas[n] * (p.detach() - self.past_parameters[n]) ** 2).sum()
                
            if check_flag:
                print(loss, reg_loss)
            
            loss += self.reg_coef * reg_loss


        return loss

    def generate_rotations(self, data):
        data90 = np.rot90(data, axes=[1,2])
        data180 = np.rot90(data90, axes=[1,2])
        data270 = np.rot90(data180, axes=[1,2])

        return data, data90, data180, data270 
        
    def augment_data(self, data, phase):

        data0, data90, data180, data270 = self.generate_rotations(data)
        a, b, c, d = data.shape
        data = np.ndarray((4 * a, b, c, d))
        labels = np.zeros((len(data), 1), dtype=np.int32)

        for i in range(len(data0)):
            data[4*i] = data0[i]
            data[4*i + 1] = data90[i]
            data[4*i + 2] = data180[i]
            data[4*i + 3] = data270[i]
            
            labels[4*i, 0] = 0
            labels[4*i + 1, 0] = 1
            labels[4*i + 2, 0] = 2
            labels[4*i + 3, 0] = 3

        data = np.moveaxis(data, [1, 2, 3], [2, 3, 1])

        tensor_x_train = torch.Tensor(data[:-4*self.omega_data_size]) # transform to torch tensor
        tensor_y_train = torch.LongTensor(labels[:-4*self.omega_data_size])
        
        tensor_x_omega = torch.Tensor(data[-4*self.omega_data_size:]) # transform to torch tensor
        tensor_y_omega = torch.LongTensor(labels[-4*self.omega_data_size:])

        dataset_train = data_utils.TensorDataset(tensor_x_train, tensor_y_train) # create your datset
        dataloader_train = data_utils.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False) # create your dataloader
        
        dataset_omega = data_utils.TensorDataset(tensor_x_omega, tensor_y_omega) # create your datset
        dataloader_omega = data_utils.DataLoader(dataset_omega, batch_size=self.batch_size, shuffle=False) # create your dataloader
        

        return dataloader_train, dataloader_omega

    def train(self, x, y, phase, experiment_params, sort=False):
        self.current_task = phase - 1
        self.model.train()   
        self.omega_data_size = int(x.shape[0] * 0.1)

        print("Augmenting Data")
        train_data, omega_data = self.augment_data(x, phase)
        print(len(train_data), len(omega_data))

        print("Beginning Training")

        running_loss = 0


        for e in range(self.epochs):

            for i, data in enumerate(train_data):
                inputs, labels = data

                inputs = inputs.to(self.dev)
                labels = labels.to(self.dev)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                
                if i % 1000 == 0:
                    loss = self.calculate_loss(outputs, labels.squeeze(), check_flag=True)
                else:
                    loss = self.calculate_loss(outputs, labels.squeeze())
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Loss Statistics
                self.loss_history.append(loss)
                
                if i % 100 == 0:
                    print("{} / {} Examples Trained\tRunning Loss: {}".format(i, len(train_data), running_loss))
                    running_loss = 0

        
        # Omega Calculation and task update
        self.model.eval()
        self.optimizer.zero_grad()


        print("Calculating Omega")

        for n, p in self.model.named_parameters():
            self.omegas[n] = p.clone().fill_(0)

        for i, data in enumerate(omega_data):
            inputs, labels = data

            inputs = inputs.to(self.dev)
            labels = labels.to(self.dev)

            # forward + backward + optimize
            outputs = self.model(inputs)

            loss = torch.norm(outputs)

            loss.backward()

            for n, p in self.model.named_parameters():
                self.omegas[n] += (p.grad.abs() / (4 * self.omega_data_size))

            self.optimizer.zero_grad()
        
        for n, p in self.model.named_parameters():
            self.past_parameters[n] = p.clone()

        if self.vis_train:
            self.save_visualizations(phase)
        

    # build classifiers using learned embedding and labeled data
    def supervise(self, data, labels, phase, experiment_params, l_list = None, index = 0):
            
        # dictionary of classifiers
        l_class = {}
        
        # encode labeled examples
        print("Encoding Labeled Data")
        labeled_features = self.encode(data)
        a,b,c,d = labeled_features.shape
        labeled_features = labeled_features.reshape((a, b * c * d))
                     
        # create classifiers
        l_class['nn'] =  KNeighborsClassifier(1).fit(labeled_features, labels)
        l_class['5nn'] = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(labeled_features, labels)
        l_class['5nn-d'] = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(labeled_features, labels)

        # normalize features
        self.feat_mean[index] = np.mean(labeled_features, axis=0)
        self.feat_var[index] = (np.var(labeled_features, axis=0) + 0.01)**(1/2)
        labeled_features = (labeled_features - self.feat_mean[index]) / self.feat_var[index]

        # create classifiers - unit variance
        l_class['nn_N'] =  KNeighborsClassifier(1).fit(labeled_features, labels)
        l_class['5nn_N'] = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(labeled_features, labels)
        l_class['5nn-d_N'] = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(labeled_features, labels)

        
        # save classifiers
        self.index_norm = {'nn':False,'5nn':False,'5nn-d':False,'nn_N':True,'5nn_N':True,'5nn-d_N':True,'centroids_N':True}
        self.classifiers[index] = l_class

    # call to classification
    def classify(self, data, phase, c_type, index, experiment_params):
        
        # encode labeled examples
        labeled_features = self.encode(data)
        a,b,c,d = labeled_features.shape
        labeled_features = labeled_features.reshape((a, b * c * d))

        # make prediction
        if self.index_norm[c_type]:
            labeled_features = (labeled_features - self.feat_mean[index]) / self.feat_var[index]

        labels = self.classifiers[index][c_type].predict(labeled_features)                         

        return labels
       
    # called by main when new task is defined
    def setTask(self ,nsamples, K):
        pass

    # encode image into feature space
    def encode(self, data):
        self.model.eval()

        num_data = data.shape[0]

        data = np.moveaxis(data, [1, 2, 3], [2, 3, 1])
        data = torch.Tensor(data)

        data = data_utils.TensorDataset(data)
        dataloader = data_utils.DataLoader(data, batch_size=100, shuffle=False)
        
        for i, data in enumerate(dataloader):
            data_ = data[0].to(self.dev)
            encoded_feats = self.model(data_, out_feat_keys=['conv2'])
            if i == 0:
                out_feats = np.zeros((num_data, encoded_feats.shape[1], encoded_feats.shape[2], encoded_feats.shape[3]))
                out_feats[0:100] = np.asarray(encoded_feats.cpu().detach())
            else:
                out_feats[i*100:(i+1)*100] = np.asarray(encoded_feats.cpu().detach())

        return out_feats
        
    def cluster(self, X, Y, phase_num, task_num, dataset, num_classes,
                experiment_params, cluster_method='kmeans', 
                accuracy_method='purity', k_scale=2, eval_layers=[]):
        
        embeddings = self.encode(X)
        a,b,c,d = embeddings.shape
        embeddings = embeddings.reshape((a, b * c * d))

        ind = np.argsort(Y)
        embeddings = embeddings[ind, :]
        Y = Y[ind]
        X = X[ind]

        k = np.unique(Y).shape[0] * k_scale
        accu_total = 0
        accu_perclass = np.zeros(num_classes, dtype=np.float64)

        if cluster_method == 'kmeans':
            cluster_preds = KMeans(n_clusters=k, init='k-means++', n_init=10, 
            max_iter=300, verbose=0).fit_predict(embeddings)

        if accuracy_method == 'purity':
            size = k
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
        

        return accu_total, accu_perclass


    def pick_sample_images(self, x_test, y_test, skip=20):

        self.sample_images = []
        self.sample_labels = []

        k = np.unique(y_test)

        i = 0

        for i, im in enumerate(x_test):
            if i % skip == 0:
                self.sample_images.append(x_test[i])
                self.sample_labels.append(y_test[i])

                plt.imshow(im.reshape(self.im_size, self.im_size, self.num_c).squeeze())
                plt.title('Class: {}'.format(y_test[i]))
                plt.savefig(smart_dir(self.plot_directory + '/sample_imgs/') + 'sample_image_{}'.format(i))
                plt.close()

    # save STAM visualizations    
    def save_visualizations(self, phase):
        pass
        '''
        plt.figure(figsize=(6,3)) 
        y = np.asarray(self.loss_history)
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.vlines(self.new_task_detections, 0, 1, linestyles='dashed')
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Unlabeled Images Seen', fontsize=12) 
        plt.title('Loss Surface', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig(smart_dir(self.plot_directory + '/task_detection/{}/'.format(phase)) + 'loss_surface.png', format='png', dpi=200)
        plt.close()
        '''