import argparse
import json

class TrainParser:
    """
    Command line parser for the training
    """

    def __init__(self):

        parser = argparse.ArgumentParser()

        # learning schedule parameters
        parser.add_argument('--schedule_flag', type=int, default=None, help='class schedule selection')
        parser.add_argument('--shuffle_flag', type=int, default=None, help='class shuffle selection')
        parser.add_argument('--dataset', '-d', default=None, help='valid entries: mnist, emnist, cifar, stl10, svhn')
        parser.add_argument('--N_p', type=int, default=None, help='total unlabeled images per phase')
        parser.add_argument('--seed', type=int, default=1, help='random seed')

        # eval scenario paramters
        parser.add_argument('--N_l', type=int, default=100, help='number labeled examples per class')
        parser.add_argument('--ntrials', type=int, default=None, help='number of trials')
        parser.add_argument('--nts', type=int, default=10000, help='Number of Task Sample datapoints to sample eval and labeled from')
        parser.add_argument('--ntp', type=int, default=5, help='Number of Tasks to evaluate per Phase')
        parser.add_argument('--N_e', type=int, default=100, help='number of eval datapoints per class')

        # general experiment parameters
        parser.add_argument('--aug', default=False, action='store_true', help='data augmentation')
        parser.add_argument('--scale_flag', default=False, action='store_true', help='norm dataset')
        parser.add_argument('--norm_flag', type=int, default=0, help='0: no norm, 1: image standardization, 2: zca whitening, 3: sobel filtering, 4: patch standardization')
        parser.add_argument('--testing', default=False, action='store_true', help='use test data instead of validation data')

        # stam parameters
        parser.add_argument('--model_name', '-mn', default=None, help='valid entries: kmeans, stam, cnn')
        parser.add_argument('--model_flag', type=int, default=1, help='model architecture selection')
        parser.add_argument('--wta', default=False, action='store_true', help='stam reconstruct with winner takes all')
        parser.add_argument('--beta', type=float, default=0.95, help='stam beta')
        parser.add_argument('--delta', type=int, default=100, help='stam delta')
        parser.add_argument('--alpha', type=float, default=1e-1, help='stam alpha')
        parser.add_argument('--rho', type=float, default=0.15, help='for g threshold')
        parser.add_argument('--theta', type=int, default=30, help='stam ltm count')
        parser.add_argument('--kernel', type=int, default=0, help='image blurring kernel')
        parser.add_argument('--novelty_freeze', default=False, action='store_true', help='freeze init novelty detection statistics')
        parser.add_argument('--init_size', type=int, default=0, help='amount of data to show for init.')
        parser.add_argument('--exp_feat', type=int, default=200, help='expected features for init centroids')
        parser.add_argument('--nd_fixed', default=False, action='store_true', help='fixed or dynamic novelty detection')


        # dnn parameters
        parser.add_argument('--epochs', type=int, default=1, help='training data epochs')
        parser.add_argument('--feat_dim', type=int, default=32, help='latent dim of vae')
        parser.add_argument('--batch', type=int, default=4, help='batch size for data')
        parser.add_argument('--lr', type=float, default=1e-3)

        # MAS parameters        
        parser.add_argument('--reg_coef', type=float, default=10e-4, help='regularization coefficient parameter for loss')
        
        # file parameters
        parser.add_argument('--vis', default=False, action='store_true', help='save visualizations for training')
        parser.add_argument('--vis_cluster', default=False, action='store_true', help='save visualizations for clustering')
        parser.add_argument('--g_sweep', default=False, action='store_true', help='sweep over several g values')

        # other
        parser.add_argument('--color_format', default='rgb', choices=['rgb', 'gray', 'hsv', 'hv'], help='color format to be used')
        parser.add_argument('--layers_flag', type=int, default=None, help='layers used in classification')
        parser.add_argument('--kmeans_flag', default=False, action='store_true', help='if true do not reconstruct image in hierarchy')
        parser.add_argument('--k', type=int, default=2)
        parser.add_argument('--num_stages', type=int, default=4)
        parser.add_argument('--sorted', default=False, action='store_true', help='sort the incremental stream by class')
        parser.add_argument('--start_trial', type=int, default=1, help='trial to resume execution')
        parser.add_argument('--start_phase', type=int, default=1, help='phase to resume execution')
        parser.add_argument('--train_only', default=False, action='store_true', help='whether or not to train only')
        parser.add_argument('--test_only', default=False, action='store_true', help='whether or not to test only')
        parser.add_argument('--vis_only', default=False, action='store_true', help='whether or not to visualize only')
        parser.add_argument('--train_test', default=True, action='store_true', help='whether or not to train in addition to test')
        parser.add_argument('--log', default='None', help='directory to log results')
        parser.add_argument('--load_log', default='None', help='directory to load log results')
        parser.add_argument('--transfer', default=False, action='store_true', help='whether or not to test on transfer tasks')
        parser.add_argument('--ncomp', type=int, default=100, help='number of pca components')
        
        parser.add_argument('--experiment', default=False, action='store_true', help='whether or not to experiment on core50')

        # dataset
        #parser.add_argument('--dataset', type=str, default='mnist',
        #                    choices=['cifar10', 'mnist', 'omniglot', 'path'],
        #                    help='dataset')
        parser.add_argument('--training_data_type', type=str, default='sequential',
                            choices=['iid', 'sequential'],
                            help='iid or sequential datastream')
        parser.add_argument('--blend_ratio', type=float, default=.0,
                            help="the ratio blend classes at the boundary")
        parser.add_argument('--n_concurrent_classes', type=int, default=1,
                            help="the number of concurrent classes showing at the same time")
        parser.add_argument('--train_samples_per_cls', type=int, nargs='+',
                            help="the number of training samples per class")
        parser.add_argument('--test_samples_per_cls', type=int, default=500,
                            help="the number of testing samples per class")
        parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

        # Evaluation
        parser.add_argument('--k_scale', type=float, default=1.0,
                            help='to scale the number of classes during evaluation')


        self.parser = parser

    def parse(self):

        self.args = self.parser.parse_args()
        return self.args

    def save_args(self, file):
        json.dump(
            vars(self.args), open(file, "w")
        )
