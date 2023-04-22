import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm, rcParams
import numpy as np, scipy.stats as st

import glob
import os
import csv
import shutil
import random

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

#fpath = os.path.join("/usr/share/fonts/truetype", "Consolas-Bold_11600.ttf")
#prop = fm.FontProperties(fname=fpath)

def smart_dir(dir_name, base_list = None):
    dir_name = dir_name + '/'
    if base_list is None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    else:
        dir_names = []
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for d in range(len(base_list)):
            dir_names.append(dir_name + base_list[d] + '/')
            if not os.path.exists(dir_names[d]):
                os.makedirs(dir_names[d])
        return dir_names


colors = ['b','r','g','y','m','b','r','g','y','m','b','r','g','y','m', 'b','r','g','y','m','b','r','g','y','m','b','r','g','y','m']
marks =  ['D','D','D','D','D','x','x','x','x','x','s','s','s','s','s', 'D','D','D','D','D','x','x','x','x','x','s','s','s','s','s']
styles = ['dashed','dashed','dashed','dashed','dashed','solid', 'solid', 'solid', 'solid', 'solid','dotted','dotted','dotted','dotted','dotted', 'dashed','dashed','dashed','dashed','dashed','solid', 'solid', 'solid', 'solid', 'solid','dotted','dotted','dotted','dotted','dotted']



def reshape_data(data, experiment):
    shape = list(data[experiment].shape)
    combined = shape[0] * shape[1]
    del shape[0]
    shape[0] = combined
    data[experiment] = data[experiment].reshape(tuple(shape))

    return data[experiment]


def process_data(dataset, model, experiment, task_ind, method_ind):
    
    loaded = pickle.load( open( "results/{}/{}/results.pkl".format(model, dataset.lower()), "rb" ))
    loaded = reshape_data(loaded, experiment)
    data = loaded


    if experiment == 'image_retreival_acc':
        mean = np.mean(data[:, :, task_ind], axis=0)
        std = np.std(data[:, :, task_ind], axis=0)

    elif experiment == 'classification_accuracy_pc':
        mean = np.mean(data[:, :, task_ind, method_ind, :], axis=0)
        std = np.std(data[:, :, task_ind, method_ind, :], axis=0)
        ret_info = []
        for i in range(mean.shape[1]):
            mean[:,i][mean[:,i] == -1] = np.nan
            ret_info.append((str(i), mean[:,i], std[:,i]))
        alll = ('all', np.mean(mean, axis=1), np.mean(std, axis=1))

        ret_info.append(alll)

        return ret_info
    elif experiment == 'class_informative':
        mean = np.mean(data[:, :, task_ind, :], axis=0) * 100
        std = np.std(data[:, :, task_ind, :], axis=0) * 10
        ret_info = []
        for i in range(mean.shape[1]):
            ret_info.append(('Layer {}'.format(i+1), mean[:,i], std[:,i]))

        return ret_info
    else:
        mean = np.mean(data[:, :, task_ind, method_ind], axis=0)
        std = np.std(data[:, :, task_ind, method_ind], axis=0)
    
    return (model, mean, std)



def comparison_line(models, title, x_label, y_label, dataset, fig_name, cluster_flag=False):

    num_phases, dataset, data_size, l_examples, classes_per_phase = dataset

    fig_name, schedule = fig_name

    title = "{}, {}, {} labeled examples p.c.\n{} phases, {} new classes per phase".format(dataset, schedule, l_examples, num_phases, classes_per_phase)
    if cluster_flag:
        title = "{}, {}\n{} phases, {} new classes per phase".format(dataset, schedule, num_phases, classes_per_phase)

    plt.figure(figsize=(6,3)) 

    x = np.arange(data_size, (num_phases+1)*data_size, data_size)

    mean = np.zeros((len(models), num_phases))
    std = np.zeros((len(models), num_phases))

    for i, model in enumerate(models):
        mean[i] = model[1]
        std[i] = model[2]
     
    indices = np.flip(np.argsort(mean[:, -1]))

    for ind in indices:
        label = models[ind][0].split("_")[0].upper()
        plt.errorbar(x, mean[ind], yerr=std[ind], fmt='o', label = label  , capthick=2, capsize=2, linestyle = styles[ind], color = colors[ind])
        plt.plot(x, mean[ind], linestyle = styles[ind], color = colors[ind])

    # axis and stuff
    plt.yticks(np.arange(10, 110, 10))
    plt.ylim(0,100)
    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.xlabel(x_label, fontweight='bold', fontsize=12)
    
    if dataset.lower() == 'emnist':
        plt.xticks(np.arange(0, (num_phases+1)*data_size, data_size*5))
    else:
        plt.xticks(np.arange(0, (num_phases+1)*data_size, data_size))

    plt.title(title, fontweight='bold', fontsize=14)
    
    plt.legend(loc='lower left', prop={'weight': 'bold', 'size': 10})
    if fig_name == 'per_class':
        plt.legend(loc='lower left', prop={'weight': 'bold', 'size': 8}, ncol=3)
    
    plt.tight_layout()

    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.savefig(smart_dir('plots/accuracies/{}/{}/'.format(schedule, fig_name)) + dataset + '_' + schedule.lower() + '_' + fig_name + '.png')
    plt.close()


def classification_plots(models, datasets, schedule):
    for d, dataset in enumerate(datasets):
        mdls = []
        for model in models:
            if 'stam' in model:
                ci = 0
            else:
                ci = 0
            info = process_data(dataset, model, 'classification_accuracy', 0, ci)
            mdls.append(info)

        comparison_line(mdls, 'Classification Accuracy', 'Unlabled Images Seen', 'Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('classification', schedule))

def clustering_plots(models, datasets, schedule):
    for d, dataset in enumerate(datasets):
        mdls = []
        for model in models:
            if 'stam' in model:
                ci = 0
            else:
                ci = 0
            info = process_data(dataset, model, 'clustering_acc', 0, ci)
            mdls.append(info)

        comparison_line(mdls, 'Clustering Accuracy', 'Unlabled Images Seen', 'Cluster Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('clustering', schedule), cluster_flag=True)

def retrieval_plots(models, datasets, schedule):

    for d, dataset in enumerate(datasets):
        mdls = []
        print(dataset)
        for model in models:
            info = process_data(dataset, model, 'image_retreival_acc', 0, 0)
            mdls.append(info)

        comparison_line(mdls, 'Retrieval Accuracy', 'Unlabled Images Seen', 'Retrieval Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('retrieval','Incremental'))

def ltm_growth_plots(datasets, schedule):

    for d, dataset in enumerate(datasets):
        loaded = pickle.load( open( "results/{}/{}/results.pkl".format('stam_{}'.format(schedule.lower()), dataset.lower()), "rb" ))
        ltm_growth = loaded['ltm_growth']
            
        title = "{}, {}, {} labeled examples p.c.\n{} phases, {} new classes per phase".format(dataset, schedule, l_examples[d], phases[d], classes_per_phase[d])
        xlabel = "Unlabled Images Seen"
        ylabel = "LTM Centroids"

        plt.figure(figsize=(6,3))

        plt.plot(ltm_growth[0], label='layer 1')
        plt.plot(ltm_growth[1], label='layer 2')
        plt.plot(ltm_growth[2], label='layer 3')
        
        plt.title(title, fontweight='semibold', fontsize=14)
        plt.xlabel(xlabel, fontweight='semibold', fontsize=12)
        plt.ylabel(ylabel, fontweight='semibold', fontsize=12)

        plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 12})
        plt.tight_layout()

        plt.grid(b=True, which='major', color='gray', linestyle='-')        
        plt.savefig(smart_dir('plots/ltm_growth/{}'.format(schedule)) + '{}_{}_ltm.png'.format(dataset, schedule.lower()))
        plt.close()

def pc_accuracy(models, datasets, schedule):

    for d, dataset in enumerate(datasets):
        if dataset.lower() == 'emnist':
            continue
        mdls = []
        for model in models:
            info = process_data(dataset, model, 'classification_accuracy_pc', 0, 0)
            for inf in info:
                mdls.append(inf)

        comparison_line(mdls, 'Classification Accuracy', 'Unlabled Images Seen', 'Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('per_class', schedule))


def cin_perc(models, datasets, schedule):
    print("cin")
    for d, dataset in enumerate(datasets):
        mdls = []
        for model in models:
            info = process_data(dataset, model, 'class_informative', 0, 0)
            for inf in info:
                mdls.append(inf)

        comparison_line(mdls, 'CIN (%)', 'Unlabled Images Seen', 'CIN (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('class_informative', schedule))

def ltm_ablation_plots(models, datasets, schedule, labels):
    
    for d, dataset in enumerate(datasets):
        mdls = []
        for m, model in enumerate(models):
            label, mean, std = process_data(dataset, model, 'classification_accuracy', 0, 0)
            mdls.append((labels[m], mean, std))

        comparison_line(mdls, 'Classification Accuracy', 'Unlabled Images Seen', 'Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('LTM Ablation', schedule), abl_flag=True)

def layer_ablation_plots(models, datasets, schedule, labels):
    
    for d, dataset in enumerate(datasets):
        mdls = []
        for m, model in enumerate(models):
            label, mean, std = process_data(dataset, model, 'classification_accuracy', 0, 0)
            mdls.append((labels[m], mean, std))

        comparison_line(mdls, 'Classification Accuracy', 'Unlabled Images Seen', 'Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('Layer Ablation', schedule), abl_flag=True)

def dynamic_ablation_plots(models, datasets, schedule, labels):
    
    for d, dataset in enumerate(datasets):
        mdls = []
        for m, model in enumerate(models):
            label, mean, std = process_data(dataset, model, 'classification_accuracy', 0, 0)
            mdls.append((labels[m], mean, std))

        comparison_line(mdls, 'Classification Accuracy', 'Unlabled Images Seen', 'Accuracy (%)', (phases[d], dataset, data_sizes[d], l_examples[d], classes_per_phase[d]), ('LTM Updates Ablation', schedule), abl_flag=True)



# Incremental
datasets = ['MNIST', 'CIFAR-10', 'SVHN', 'EMNIST']
data_sizes = [10000, 10000, 10000, 2000]
phases = [5, 5, 5, 23]
l_examples = [10, 100, 100, 10]
classes_per_phase = [2, 2, 2, 2]


classification_plots(['stam_incremental', 'gem_incremental', 'mas_incremental'], datasets, 'Incremental')
clustering_plots(['stam_incremental', 'gem_incremental', 'mas_incremental'], datasets, 'Incremental')
pc_accuracy(['stam_incremental'], datasets, 'Incremental')
cin_perc(['stam_incremental'], datasets, 'Incremental')
ltm_growth_plots(datasets, 'Incremental')


# Uniform
datasets = ['MNIST', 'SVHN', 'CIFAR-10']
data_sizes = [10000, 10000, 10000]
phases = [5, 5, 5]
l_examples = [10, 100, 100]
classes_per_phase = [2, 2, 2]

classification_plots(['stam_uniform'], datasets, 'Uniform')
clustering_plots(['stam_uniform'], datasets, 'Uniform')
pc_accuracy(['stam_uniform'], datasets, 'Uniform')
cin_perc(['stam_uniform'], datasets, 'Uniform')
ltm_growth_plots(datasets, 'Uniform')
