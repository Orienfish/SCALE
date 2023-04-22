import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import pickle
import signal
import time

from options import TrainParser

from core.models.stam_wrapper import StamWrapper
from core.models.gem_wrapper import GEMWrapper
from core.models.mas_wrapper import MASWrapper

from core.utils import *
from core.dataset import *
from core.config import *
from core.data_sampling import *
from core.distance_metrics import *
from core.checkpoints import *

#from plots import *

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from eval_utils import *

def run_train(results, trial, args, configs):
    
    # reset random seed
    np.random.seed(trial)
    configs['seed'] = trial
  
    print('**************************************************')
    print('TRIAL ', str(trial+1))
    print('**************************************************')
        
    # Trial Params
    num_classes = configs['num_classes']
    num_phases = configs['num_phases']

    # Datastream
    datastream = stream['datastream']
    tasks_set = stream['tasks_set']

    # Extra Experimental Params
    train_params = configs['train_params']

    if args.model_name == 'stam':
        configs = load_stam_configs(args) 
    if args.model_name == 'gem':     
        configs = load_gem_configs(args)
    if args.model_name == 'mas':
        configs = load_mas_configs(args)
    

    #if not args.experiment:
    x_train, y_train, x_test, y_test = sample_datastream(x_, x_eval, y_, y_eval, 
                                           datastream, num_classes, 
                                           num_phases, args)

    if args.vis:
        model.pick_sample_images(x_test[0][0], y_test[0][0])

    # for each incremental phase
    for phase in range(num_phases):

        class_tasks = tasks_set[phase]

        # train from unlabeled data
        print('Phase ' + str(phase+1) + ':')
        print('Learning from data stream...')
        print('Class Distribution {}'.format(np.bincount(y_train[phase], minlength=num_classes)))
        model.train(x_train[phase], y_train[phase], phase+1, train_params)

        save_checkpoint(None, model, trial, phase, args.log, args.dataset)


def run_test(log, trial, results, args, configs):
    
    # reset random seed
    np.random.seed(trial)
    configs['seed'] = trial

    print('**************************************************')
    print('TRIAL ', str(trial+1))
    print('**************************************************')
    
    # Trial Params
    num_classes = configs['num_classes']
    num_phases = configs['num_phases']

    # Accuracy Models
    classifiers = configs['classifiers']
    clustering_models = configs['clustering_models']

    # Datastream
    datastream = stream['datastream']
    stream_name = stream['stream_name']
    tasks_string = stream['tasks_string']
    tasks_set = stream['tasks_set']

    # Extra Experimental Params
    supervise_params = configs['supervise_params']
    classify_params = configs['classify_params']
    cluster_params = configs['cluster_params']

    _, _, x_test, y_test = sample_datastream(x_, x_eval, y_, y_eval, 
                                            datastream, num_classes, 
                                            num_phases, args)

    l_eval = configs['l_eval']

    # for each incremental phase
    for phase in range(args.start_phase-1, num_phases):
        class_tasks = tasks_set[phase]

        # train from unlabeled data
        print('Phase ' + str(phase+1) + ':')

        _, model = load_checkpoint(trial, phase, args.load_log, args.dataset, test=True)
   
        model.plot_directory = configs['plot_directory']
        model.vis_cluster = configs['visualize_cluster']
        model.vis_train = configs['visualize_train']

        
        # do not always have task to evaluate
        if len(class_tasks) > 0:
            
            # reset stored classification info
            model.setTask(args.ntp, len(np.unique(class_tasks[0])))

            # for num_tasks time (number tasks per phase)
            for sample in range(args.ntp):

                # Show eval sample number
                print('   Sample ' + str(sample+1) + '/' + str(args.ntp))

                # labeled data (supervision)
                x_supervise = x_test[sample][1]
                y_supervise = y_test[sample][1]

                # querry data (evaluation)
                x_query = x_test[sample][0]
                y_query = y_test[sample][0]

                # for all tasks
                for ti, task in enumerate(class_tasks):                    
                    # get tasks (i.e. classes for classification task)
                    t_string = tasks_string[ti]


                    # get task labeled data and eval data
                    x_query_t = x_query[np.isin(y_query, task)]
                    y_query_t = y_query[np.isin(y_query, task)].astype(int)
                    x_supervise_t = x_supervise[np.isin(y_supervise, task)]
                    y_supervise_t = y_supervise[np.isin(y_supervise, task)].astype(int)

                    print('Task: ' + str(task))
                    print('Total Labeled Examples Per Class ' + str(np.bincount(y_supervise_t, minlength=num_classes)))
                    print('Total Test Examples Per Class ' + str(np.bincount(y_query_t, minlength=num_classes)))
            
                    
                    # supervision
                    print('      Showing supervision...')
                    model.supervise(x_supervise_t, y_supervise_t, phase,
                                    supervise_params, l_list=l_eval, index=sample)

                    
                    # Percent of class informative centroids
                    if args.model_name == 'stam': 
                        ci_score, ci_score_pc, multi_ci = model.get_ci(phase, sample, args.vis)


                        results['class_informative'][trial, sample, phase, ti, :] = ci_score
                        results['class_informative_pc'][trial, sample, phase, ti, :, :] = ci_score_pc
                        results['class_informative_multi'][trial, sample, phase, ti, :] = multi_ci
                    
                    if args.vis and args.model_name == 'stam':
                        model.detailed_classification_plots()

                    # classification and evalution
                    print('      Eval Task: ' + t_string)
                    print('         Classifying...')

                    confusion_matrices = np.zeros((len(classifiers), num_classes, num_classes))

                    
                    for ci, classifier in enumerate(classifiers):

                        # Weakly supervised predictions
                        y_predict = model.classify(x_query_t, phase, classifier, 
                                                    sample, classify_params).astype(int)

                        # Calculate Confusion Matrix for Predictions
                        if args.vis and sample == 0:
                            for cf in range(len(y_predict)):
                                confusion_matrices[ci, int(y_query_t[cf]), int(y_predict[cf])] += 1

                            plt.imshow(confusion_matrices[0], cmap='hot', interpolation='nearest')
                            plt.ylabel("True Class")
                            plt.xlabel("Predicted Class")
                            plt.xticks(np.arange(num_classes))
                            plt.yticks(np.arange(num_classes))
                            for i in range(num_classes):
                                for j in range(num_classes):
                                    text = plt.text(j, i, confusion_matrices[0, i, j], ha="center", va="center", color="w")

                            plt.savefig(smart_dir(model.plot_directory + '/phase_{}/'.format(phase)) + 'confusion_matrix.png')
                            plt.close()

                        # all class results
                        score = 100 * np.mean(y_query_t == y_predict)
                        
                        print('            ' + classifier + ': ' \
                              + str(score) + '%')
                        if args.model_name == 'stam':
                            print('            ' + classifier + ': ' \
                                  + str(ci_score), ' % class informative')
                            print('            ' + classifier + ': ' \
                                  + str(multi_ci), ' % class informative - more than 1')

                        # per class results
                        score_pc = [100 * np.mean(y_query_t[np.where(y_query_t == k)] \
                            == y_predict[np.where(y_query_t == k)]) for k in task] 

                        results['classification_accuracy'][trial, sample, phase, ti, ci] = score
                        results['classification_accuracy_pc'][trial, sample, phase, ti, ci, :len(task)] = score_pc

                    
                    for ci, cluster_method in enumerate(clustering_models):

                        acc, pc_acc = model.cluster(x_query_t, y_query_t, phase,
                                                    sample, args.dataset, num_classes, 
                                                    cluster_params[ci],
                                                    cluster_method, eval_layers=l_eval)
                        
                        print('            ' + cluster_method + ': ' \
                              + str(acc) + '%')
                        print(ti, ci)

                        results['clustering_acc'][trial, sample, phase, ti, ci] = acc
                        results['clustering_acc_pc'][trial, sample, phase, ti, ci, :] = pc_acc
                    
    if args.model_name == 'stam':
        results['ltm_growth'] = [model.layers[0].ltm_history, model.layers[1].ltm_history, model.layers[2].ltm_history]
    
    return results

def run_trial(trial, results, args, configs):
    setting_name = '{}_{}_{}_{}'.format(args.dataset,
                                        args.training_data_type,
                                        str(args.blend_ratio),
                                        str(args.n_concurrent_classes)
                                        )
    
    # reset random seed
    np.random.seed(trial)
    configs['seed'] = trial
    
    # Trial Params
    num_classes = configs['num_classes']
    num_phases = configs['num_phases']

    # Accuracy Models
    classifiers = configs['classifiers']
    clustering_models = configs['clustering_models']

    # Datastream
    datastream = stream['datastream']
    stream_name = stream['stream_name']
    tasks_string = stream['tasks_string']
    tasks_set = stream['tasks_set']

    # Extra Experimental Params
    train_params = configs['train_params']
    supervise_params = configs['supervise_params']
    classify_params = configs['classify_params']
    cluster_params = configs['cluster_params']



    # print trial number
    print('**************************************************')
    print('TRIAL ', str(trial+1))
    print('**************************************************')

    if args.model_name == 'stam':
        model = StamWrapper(configs)   
    if args.model_name == 'gem':     
        model = GEMWrapper(configs)
    if args.model_name == 'mas':
        model = MASWrapper(configs)

    
    x_train, y_train, x_test, y_test = sample_datastream(x_, x_eval, y_, y_eval, 
                                            datastream, num_classes, 
                                            num_phases, args)

    l_eval = configs['l_eval']

    if args.vis:
        model.pick_sample_images(x_test[0][0], y_test[0][0])

    # for each incremental phase
    for phase in range(args.start_phase-1, num_phases):

        class_tasks = tasks_set[phase]

        # train from unlabeled data
        print('Phase ' + str(phase+1) + ':')
        print('Learning from data stream...')
        print('Class Distribution {}'.format(np.bincount(y_train[phase], minlength=num_classes)))
        print(model.layers[0].im_seen)

        model.train(x_train[phase], y_train[phase], phase+1, train_params)

        save_checkpoint(results, model, trial, phase, args.log, args.dataset)

        
        # do not always have task to evaluate
        if len(class_tasks) > 0:
            
            # reset stored classification info
            model.setTask(args.ntp, len(np.unique(class_tasks[0])))

            # for num_tasks time (number tasks per phase)
            for sample in range(args.ntp):

                # Show eval sample number
                print('   Sample ' + str(sample+1) + '/' + str(args.ntp))

                # labeled data (supervision)
                x_supervise = x_test[sample][1]
                y_supervise = y_test[sample][1]

                # querry data (evaluation)
                x_query = x_test[sample][0]
                y_query = y_test[sample][0]

                # for all tasks
                for ti, task in enumerate(class_tasks):                    
                    # get tasks (i.e. classes for classification task)
                    t_string = tasks_string[ti]


                    # get task labeled data and eval data
                    x_query_t = x_query # [np.isin(y_query, task)]
                    y_query_t = y_query # [np.isin(y_query, task)].astype(int)
                    x_supervise_t = x_supervise # [np.isin(y_supervise, task)]
                    y_supervise_t = y_supervise # [np.isin(y_supervise, task)].astype(int)

                    print('Task: ' + str(task))
                    print('Total Labeled Examples Per Class ' + str(np.bincount(y_supervise_t, minlength=num_classes)))
                    print('Total Test Examples Per Class ' + str(np.bincount(y_query_t, minlength=num_classes)))
            
                    
                    # supervision
                    print('      Showing supervision...')
                    model.supervise(x_supervise_t, y_supervise_t, phase,
                                    supervise_params, l_list=l_eval, index=sample)


                    # Percent of class informative centroids
                    if args.model_name == 'stam': 
                        ci_score, ci_score_pc, multi_ci = model.get_ci(phase, sample, args.vis)


                        results['class_informative'][trial, sample, phase, ti, :] = ci_score
                        results['class_informative_pc'][trial, sample, phase, ti, :, :] = ci_score_pc
                        results['class_informative_multi'][trial, sample, phase, ti, :] = multi_ci

                    if args.vis and args.model_name == 'stam':
                        model.detailed_classification_plots()

                    # classification and evalution
                    print('      Eval Task: ' + t_string)
                    print('         Classifying...')

                    confusion_matrices = np.zeros((len(classifiers), num_classes, num_classes))


                    for ci, classifier in enumerate(classifiers):

                        # Weakly supervised predictions
                        y_predict = model.classify(x_query_t, phase, classifier, 
                                                    sample, classify_params).astype(int)

                        # Calculate Confusion Matrix for Predictions
                        if args.vis and sample == 0:
                            for cf in range(len(y_predict)):
                                confusion_matrices[ci, int(y_query_t[cf]), int(y_predict[cf])] += 1

                            plt.imshow(confusion_matrices[0], cmap='hot', interpolation='nearest')
                            plt.ylabel("True Class")
                            plt.xlabel("Predicted Class")
                            plt.xticks(np.arange(num_classes))
                            plt.yticks(np.arange(num_classes))
                            for i in range(num_classes):
                                for j in range(num_classes):
                                    text = plt.text(j, i, confusion_matrices[0, i, j], ha="center", va="center", color="w")

                            plt.savefig(smart_dir(model.plot_directory + '/phase_{}/'.format(phase)) + 'confusion_matrix.png')
                            plt.close()

                        # all class results
                        score = 100 * np.mean(y_query_t == y_predict)
                        
                        print('            ' + classifier + ': ' \
                              + str(score) + '%')
                        if args.model_name == 'stam':
                            print('            ' + classifier + ': ' \
                                  + str(ci_score), ' % class informative')
                            print('            ' + classifier + ': ' \
                                  + str(multi_ci), ' % class informative - more than 1')

                        # per class results
                        score_pc = [100 * np.mean(y_query_t[np.where(y_query_t == k)] \
                            == y_predict[np.where(y_query_t == k)]) for k in task] 

                        results['classification_accuracy'][trial, sample, phase, ti, ci] = score
                        results['classification_accuracy_pc'][trial, sample, phase, ti, ci, :len(task)] = score_pc

                        if ti == len(class_tasks) - 1: # Report last task with all classes
                            if len(args.train_samples_per_cls) == 1:
                                im_ind = args.train_samples_per_cls[0]
                            else:
                                im_ind = 'im'
                            filename = 'accuracy_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                                args.dataset, args.training_data_type, classifier,
                                args.k_scale, args.blend_ratio, args.n_concurrent_classes, im_ind)
                            with open(filename, 'a+') as f:
                                f.write('trial{},phase{},'.format(trial, phase))
                                for sample in range(args.ntp):
                                    f.write(str(results['classification_accuracy'][trial, sample, phase, -1, 0]) + ',')
                                f.write('\n')
                    
                    for ci, cluster_method in enumerate(clustering_models):

                        acc, pc_acc = model.cluster(x_query_t, y_query_t, phase,
                                                    sample, args.dataset, num_classes, 
                                                    cluster_params[ci],
                                                    cluster_method, eval_layers=l_eval, accuracy_method='acc',
                                                    k_scale=args.k_scale)

                        print('            ' + cluster_method + ': ' \
                              + str(acc) + '%')

                        results['clustering_acc'][trial, sample, phase, ti, ci] = acc
                        results['clustering_acc_pc'][trial, sample, phase, ti, ci, :pc_acc.size] = pc_acc

                        purity, pc_purity = model.cluster(x_query_t, y_query_t, phase,
                                                    sample, args.dataset, num_classes,
                                                    cluster_params[ci],
                                                    cluster_method, eval_layers=l_eval, accuracy_method='purity',
                                                    k_scale=args.k_scale)
                        
                        print('            ' + cluster_method + ': ' \
                              + str(acc) + '%')

                        results['clustering_purity'][trial, sample, phase, ti, ci] = purity
                        results['clustering_purity_pc'][trial, sample, phase, ti, ci, :] = pc_purity

                        if ti == len(class_tasks) - 1: # Report last task with all classes
                            if len(args.train_samples_per_cls) == 1:
                                im_ind = args.train_samples_per_cls[0]
                            else:
                                im_ind = 'im'
                            filename = 'cluster_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                                args.dataset, args.training_data_type, cluster_method,
                                args.k_scale, args.blend_ratio, args.n_concurrent_classes, im_ind)
                            with open(filename, 'a+') as f:
                                f.write('trial{},phase{},'.format(trial, phase))
                                for sample in range(args.ntp):
                                    f.write(str(results['clustering_acc'][trial, sample, phase, -1, 0]) + '/' +
                                            str(results['clustering_purity'][trial, sample, phase, -1, 0]) + ',')
                                f.write('\n')

                    # Newly added knn evaluation
                    knn_acc = model.knn(x_query_t, y_query_t, x_supervise_t, y_supervise_t, cluster_params[0])

                    print('            knn : ' + str(knn_acc) + '%')

                    if ti == len(class_tasks) - 1:  # Report last task with all classes
                        if len(args.train_samples_per_cls) == 1:
                            im_ind = args.train_samples_per_cls[0]
                        else:
                            im_ind = 'im'
                        filename = 'knn_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                            args.dataset, args.training_data_type, cluster_method,
                            args.k_scale, args.blend_ratio, args.n_concurrent_classes, im_ind)
                        with open(filename, 'a+') as f:
                            f.write('trial{},phase{},'.format(trial, phase))
                            for sample in range(args.ntp):
                                f.write(str(knn_acc))
                            f.write('\n')

    # Log forgetting measure and forward transfer after finishing all phases
    #acc_mat = results['clustering_acc_pc'][trial, 0, :, -1, 0, :]
    #forget = eval_forget(acc_mat)
    #transfer = eval_forward_transfer(acc_mat) # Makes no sense
    #with open('cluster_{}_{}_{}.txt'.format(setting_name, cluster_method, args.k_scale), 'a+') as f:
    #    f.write('trial{},{},{}'.format(trial, forget, transfer))
                    
    if args.model_name == 'stam':
        results['ltm_growth'] = [model.layers[0].ltm_history, model.layers[1].ltm_history, model.layers[2].ltm_history]

    return results 


if __name__ == "__main__":
    trial = 0
    
    plt.rc('image', cmap='gray')

    parser = TrainParser()
    args = parser.parse()


    if args.model_name == 'stam':
        configs = load_stam_configs(args) 
    if args.model_name == 'gem':     
        configs = load_gem_configs(args)
    if args.model_name == 'mas':
        configs = load_mas_configs(args)


    # load dataset
    (x_, y_), (x_eval, y_eval), configs = load_dataset(configs, args)

    stream, configs = form_datastream(args.schedule_flag, configs)

    # Result storage init
    results = {}
    results['classification_accuracy'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_classifiers'],
                                        ))
    results['classification_accuracy_pc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_classifiers'],
                                        configs['num_classes']
                                        ))

    results['clustering_acc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_cluster_models'],
                                        ))
    results['clustering_acc_pc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_cluster_models'],
                                        configs['num_classes']
                                        ))
    results['clustering_purity'] = -1 * np.ones((configs['num_trials'],
                                              configs['num_samples'],
                                              configs['num_phases'],
                                              configs['num_tasks'],
                                              configs['num_cluster_models'],
                                              ))
    results['clustering_purity_pc'] = -1 * np.ones((configs['num_trials'],
                                                 configs['num_samples'],
                                                 configs['num_phases'],
                                                 configs['num_tasks'],
                                                 configs['num_cluster_models'],
                                                 configs['num_classes']
                                                 ))

    results['task_3_acc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        ))

    results['task_3_acc_pc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_classes']
                                        ))

    if args.model_name == 'stam':    
        results['class_informative'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_layers']
                                        ))
        results['class_informative_pc'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_layers'],
                                        configs['num_classes']
                                        ))
        results['class_informative_multi'] = -1 * np.ones((configs['num_trials'], 
                                        configs['num_samples'],
                                        configs['num_phases'],
                                        configs['num_tasks'],
                                        configs['num_layers']
                                        ))


    # save parameters
    parser.save_args(smart_dir('logs/' + args.log + '/' + args.dataset) + 'user_settings.yml')

    save_file = smart_dir('logs/' + args.log) + 'training_configs.yml'
    with open(save_file, 'w') as yaml_file:
        yaml.dump(configs, yaml_file)
    
    if args.train_only:
        print("Training Only")
        for trial in range(args.ntrials):
            args.seed = trial
            run_train(results, trial, args, configs)
    
    elif args.test_only:
        print("Testing Only")
        for trial in range(args.start_trial - 1, args.ntrials):
            args.seed = trial
            results = run_test(args.load_log, trial, results, args, configs)
    
    elif args.train_test:
        print("Full Trial")
        for trial in range(args.start_trial-1, args.ntrials):
            args.seed = trial
            results = run_trial(trial, results, args, configs)
    

    with open(configs['results_directory'] + 'results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

