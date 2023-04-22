import pickle
from core.utils import *


def save_checkpoint(results, model, trial, phase, log, dataset):

    results_d = smart_dir('checkpoints/results/' + log + '/' + dataset + '/' + 'trial_' + str(trial))
    models_d = smart_dir('checkpoints/models/' + log + '/' + dataset + '/' + 'trial_' + str(trial))

    if results != None:
        with open(results_d \
                + 'phase_{}.pkl'.format(phase), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(models_d \
              + 'phase_{}.pkl'.format(phase), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint(trial, phase, log, dataset, test=False):

    results_d = smart_dir('checkpoints/results/' + log + '/' + dataset + '/' + 'trial_' + str(trial))
    models_d = smart_dir('checkpoints/models/' + log + '/' + dataset + '/' + 'trial_' + str(trial))
    results = None

    if not test:
        with open(results_d \
                + 'phase_{}.pkl'.format(phase), 'rb') as handle:
            results = pickle.load(handle)

    with open(models_d \
              + 'phase_{}.pkl'.format(phase), 'rb') as handle:
        model = pickle.load(handle)

    return results, model
