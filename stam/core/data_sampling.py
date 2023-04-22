import numpy as np
import math
from sklearn.utils import shuffle


# sample K examples per class from dataset with no replacement
def sample_no_replace(x, y, K, seed):

    # x: training data M x D1 x D2
    # y: training labels M
    # k: int - number of examples from each class to sample

    # shuffle
    x,y = shuffle(x,y, random_state=seed)
    
    # sample training pairs
    ind = np.empty((0,))
    for l in set(y):
        ind = np.append(ind, np.where(y == l)[0][0:K], axis = 0)
    ind = ind.astype(int)
    x_sample = x[ind]
    y_sample = y[ind]
    x = np.delete(x, ind, 0)
    y = np.delete(y, ind, 0)

    # returning x,y has sampled examples removed
    return (x, y), (x_sample, y_sample)


def sample_datastream(x_, x_eval, y_, y_eval, datastream, num_classes, nphase, args):
    np.random.seed(args.seed)

    x_train = []
    y_train = [] 
  
    x_test = []
    y_test = []

    if args.schedule_flag < 3:

        for n in range(nphase):

            train_indices = []
            num_per_phase = args.N_p

            classes = np.argwhere(np.array(datastream[n]) > 0.0).flatten()
            
            if n == 0:
                num_per_phase += args.init_size

            class_examples_per_phase = np.random.choice(num_classes, num_per_phase, p = datastream[n])
            class_totals = np.bincount(class_examples_per_phase, minlength=num_classes)

            for i in classes:
                indices = np.argwhere(y_ == i).flatten()
                np.random.shuffle(indices)
                indices = indices[:class_totals[i]]
                train_indices += list(indices)

            
            if not args.sorted:
                np.random.shuffle(train_indices)
            

            x_train.append(x_[train_indices])
            y_train.append(y_[train_indices])

        
        example_inds = []
        test_classes = np.arange(num_classes)

        for task in range(args.ntp):
            l_ind = []
            t_ind = []

            for clas in test_classes:
                inds = np.argwhere(y_eval == clas).flatten()
                np.random.shuffle(inds)
                t_ind += list(inds[:args.N_e])

            for clas in test_classes:
                inds = np.argwhere(y_ == clas).flatten()
                np.random.shuffle(inds)
                l_ind += list(inds[-(args.N_l+1):-1])

            x_test.append([x_eval[t_ind], x_[l_ind]])
            y_test.append([y_eval[t_ind], y_[l_ind]])

    if args.schedule_flag == 3:
        
        num_sessions = int(args.N_p / (300 * 5))
        for n in range(nphase):
            train_indices = []
            classes = np.argwhere(np.array(datastream[n]) > 0.0).flatten()

            if n == 0:
                phase_data = x_[0][classes[0]][:int(args.init_size / len(classes[0]))]
                phase_lab = y_[0][classes[0]][:int(args.init_size / len(classes[0]))]
            else:
                phase_data = np.array([None])
                phase_lab = None

            for session in range(num_sessions):
                for clas in classes:
                    if phase_data.any() == None:
                        phase_data = x_[session][clas]
                        phase_lab = y_[session][clas]
                    else:
                        phase_data = np.concatenate((phase_data, x_[session][clas]))
                        phase_lab = np.concatenate((phase_lab, y_[session][clas]))

            x_train.append(phase_data)
            y_train.append(phase_lab)


        test_classes = np.arange(num_classes)

        for task in range(args.ntp):

            counter = 0
            l_data = np.ndarray((args.N_l * num_classes, 64, 64, 1))
            l_lab = np.ndarray(args.N_l * num_classes)

            while counter < (args.N_l * num_classes):
                for session in range(num_sessions):
                    if counter >= (args.N_l * num_classes):
                        break
                        
                    for clas in test_classes:
                        if counter >= (args.N_l * num_classes):
                            break
                        
                        ind = np.random.randint(0,290)
                        l_data[counter,:,:,:] = x_[session][clas][ind]
                        l_lab[counter] = y_[session][clas][ind]
                        counter += 1

                        if counter > (args.N_l * num_classes):
                            break

            
            counter = 0
            e_data = np.ndarray((args.N_e * num_classes, 64, 64, 1))
            e_lab = np.ndarray(args.N_e * num_classes)

            while counter < (args.N_e * num_classes):
                for session in range(num_sessions):
                    if counter >= (args.N_e * num_classes):
                        break
                        
                    for clas in test_classes:
                        if counter >= (args.N_e * num_classes):
                            break

                        ind = np.random.randint(0,290)
                        e_data[counter,:,:,:] = x_eval[session][clas][ind]
                        e_lab[counter] = y_eval[session][clas][ind]
                        counter += 1

            order = np.random.permutation(len(e_data))
            order_2 = np.random.permutation(len(l_data))
            x_test.append([e_data[order], l_data[order_2]])
            y_test.append([e_lab[order], l_lab[order_2]])

    
    return x_train, y_train, x_test, y_test

def form_datastream(schedule_flag, configs):

    num_classes = configs['num_classes']
    num_phases = configs['num_phases']
    transfer = configs['transfer']

    classes_per_phase = math.floor(num_classes/num_phases)
    
    # incremental upl
    if schedule_flag == 1:
        print("Incremental")
        # description
        stream_name = 'incremental'
        tasks_string = ['all-seen']

        if transfer:
            for n in range(num_phases):
                tasks_string.append('phase_{}'.format(n+1))

        # form datastream
        datastream = []
        for n in range(num_phases):
            data_dist = [0.0 for k in range(num_classes)]
            step = int(num_classes / num_phases)
            for c in range(classes_per_phase):
                data_dist[step*n+c] = 1.0/classes_per_phase
            datastream.append(data_dist)


        task_set = []
        for n in range(num_phases):
            tasks = []
            tasks.append([k for k in range((n+1) * classes_per_phase)])
            if transfer:
                for t in range(num_phases):
                    tasks.append([k for k in range(t * classes_per_phase, (t+1) * classes_per_phase)])
            task_set.append(tasks)

    # uniform upl
    if schedule_flag == 2:
        print("Uniform")
        # discription
        stream_name = 'uniform'
        tasks_string = ['all-seen']

        # form datastream
        datastream = []
        for d in range(num_phases):
            data_dist = [1.0/num_classes for k in range(num_classes)]
            datastream.append(data_dist)
        
        task_set = [[[k for k in range(num_classes)]] for n in range(num_phases)]

    if schedule_flag == 3:
        print("Temporal")

        stream_name = 'temporal'
        tasks_string = ['all-seen']

        if transfer:
            for n in range(num_phases):
                tasks_string.append('phase_{}'.format(n+1))

        

        objects_in_phase = [[] for n in range(num_phases)]
        for n in range(num_phases):
            objects_in_phase[n] += [(n*5) + i for i in range(5)]

        datastream = []
        for d in range(num_phases):
            datastream.append([0.0 for x in range(num_classes)])
            for obj in objects_in_phase[d]:
                datastream[d][obj] = 1.0

        task_set = []
        for n in range(num_phases):
            tasks = []
            
            incremental_task = []
            for i in range(n+1):
                incremental_task += objects_in_phase[i]
            tasks.append(incremental_task)

            if transfer:
                for t in range(num_phases):
                    tasks.append(objects_in_phase[t])
            
            task_set.append(tasks)

    return_dic = {}
    return_dic['datastream'] = datastream
    return_dic['stream_name'] = stream_name
    return_dic['tasks_string'] = tasks_string
    return_dic['tasks_set'] = task_set

    configs['num_tasks'] = len(tasks_string)

    return return_dic, configs
