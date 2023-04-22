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

# Utility Functions taken from https://github.com/facebookresearch/GradientEpisodicMemory

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


# Primary Network Class modified from https://github.com/facebookresearch/GradientEpisodicMemory
class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 ops):
        
        super(Net, self).__init__()

        self.margin = ops['memory_strength']
        self.dev = ops['dev']
        self.num_c = ops['num_channels']

        opt = {'num_classes': 4, 
               'num_stages': 4,
               'num_inchannels': ops['num_channels'],
               'use_avg_on_conv3': False,
              }
        

        self.model = NetworkInNetwork(opt)
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.im_size = ops['im_size']

        self.opt = optim.Adam(self.parameters(), ops['lr'])

        self.n_memories = ops['num_memories']

        # allocate episodic memory
        self.memory_data = np.zeros((n_tasks, self.n_memories, n_inputs))
        self.memory_labs = np.zeros((n_tasks, self.n_memories))

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        self.grads = self.grads.to(self.dev)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = n_outputs

    def forward(self, x, feature_extraction=False):
        
        if not feature_extraction:
            output = self.model(x, out_feat_keys=None)
        else:
            output = self.model(x, out_feat_keys=['conv2'])

        return output

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
      
        self.memory_data[t, int(self.mem_cnt % self.n_memories)] = x[0].reshape(-1)
        self.memory_labs[t, int(self.mem_cnt % self.n_memories)] = y[0]
        self.mem_cnt += 1


        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                mem_data = self.memory_data[past_task]
                mem_data = mem_data.reshape(mem_data.shape[0], self.num_c, self.im_size, self.im_size)
                tensor_data = torch.FloatTensor(mem_data).to(self.dev)
                tensor_labs = torch.LongTensor(self.memory_labs[past_task]).to(self.dev)

                ptloss = self.ce(self.forward(tensor_data), tensor_labs.squeeze())
                
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        x = torch.FloatTensor(x).to(self.dev)
        y = torch.LongTensor(y).to(self.dev)

        out = self.forward(x)

        loss = self.ce(out, y.squeeze())
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.LongTensor(self.observed_tasks[:-1]).to(self.dev)
            
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))

            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
        self.opt.step()

        return loss


class GEMWrapper():

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

        self.ops['lr'] = configs['lr']
        self.ops['memory_strength'] = 0
        self.ops['num_memories'] = configs['memory_size']
        self.ops['dev'] = self.dev
        self.ops['im_size'] = self.im_size
        self.ops['num_channels'] = self.num_c
        self.ops['batch_size'] = self.batch_size
        self.ops['image_size'] = self.im_size
        self.ops['dataset'] = self.dataset

    
        # initalize the model
        print("Creating Model")
        self.model = Net(n_inputs=self.im_size**2 * self.num_c, n_outputs=4, n_tasks=self.num_phases, ops=self.ops)
        self.model.to(self.dev)           
    

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

        tensor_x = torch.Tensor(data) # transform to torch tensor
        tensor_y = torch.LongTensor(labels)

        dataset = data_utils.TensorDataset(tensor_x, tensor_y) # create your datset
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # create your dataloader
        
        return dataloader

    def train(self, x, y, phase, experiment_params, sort=False):
        self.model.train()   
        phase = phase - 1

        print("Augmenting Data")
        augmented_data = self.augment_data(x, phase)

                
        print("Beginning Training")

        running_loss = 0

        for (i, (x, y)) in enumerate(augmented_data):

            if i % 100 == 0:
                print("{} / {} Examples Trained\tRunning Loss: {}".format(i, len(augmented_data), running_loss))
                running_loss = 0
        
            running_loss += self.model.observe(x, phase, y)
 

    # build classifiers using learned embedding and labeled data
    def supervise(self, data, labels, phase, experiment_params, l_list = None, index = 0):
            
        # dictionary of classifiers
        l_class = {}
        
        # encode labeled examples
        print("Encoding Labeled Data")
        labeled_features = self.encode(data, phase)
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
        labeled_features = self.encode(data, phase)
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
    def encode(self, data, phase):
        self.model.eval()

        num_data = data.shape[0]

        data = np.moveaxis(data, [1, 2, 3], [2, 3, 1])
        data = torch.Tensor(data)

        data = data_utils.TensorDataset(data)
        dataloader = data_utils.DataLoader(data, batch_size=100, shuffle=False)
        
        for i, data in enumerate(dataloader):
            data_ = data[0].to(self.dev)
            encoded_feats = self.model(data_, feature_extraction=True)
            if i == 0:
                out_feats = np.zeros((num_data, encoded_feats.shape[1], encoded_feats.shape[2], encoded_feats.shape[3]))
                out_feats[0:100] = np.asarray(encoded_feats.cpu().detach())
            else:
                out_feats[i*100:(i+1)*100] = np.asarray(encoded_feats.cpu().detach())

        return out_feats
        
    def cluster(self, X, Y, phase_num, task_num, dataset, num_classes,
                experiment_params, cluster_method='kmeans', 
                accuracy_method='purity', k_scale=2, eval_layers=[]):
        
        embeddings = self.encode(X, phase_num)
        a,b,c,d = embeddings.shape
        embeddings = embeddings.reshape((a, b * c * d))

        print(embeddings.min(), embeddings.max())

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

            print("TEST", accu_total)
        

        return accu_total, accu_perclass


    def generate_rotations(self, data):
        data90 = np.rot90(data, axes=[1,2])
        data180 = np.rot90(data90, axes=[1,2])
        data270 = np.rot90(data180, axes=[1,2])

        return data, data90, data180, data270   

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
