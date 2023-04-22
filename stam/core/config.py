from core.utils import *

def load_stam_configs(args):
    configs = {}

    # reset random seed
    np.random.seed(args.seed)
    configs['seed'] = args.seed

    configs['results_directory'] = smart_dir('results/' + args.log + '/' + args.dataset)
    configs['plot_directory'] = smart_dir('plots/' + args.log + '/' + args.dataset)
    configs['visualize_train'] = args.vis
    configs['visualize_cluster'] = args.vis_cluster
    configs['dataset'] = args.dataset
    configs['color_format'] = args.color_format
    configs['num_images_init'] = args.init_size
    configs['expected_features'] = args.exp_feat
    configs['nd_fixed'] = args.nd_fixed

    # classifiers
    configs['classifiers'] = ['hierarchy-vote']
    configs['clustering_models'] = ['kmeans']
    #configs['clustering_models'] = ['spectral']

    # Experimental Params for altering each major function
    configs['train_params'] = ()
    configs['supervise_params'] = ()
    configs['classify_params'] = ()
    configs['cluster_params'] = [(0, 'cent-norm')]
    # configs['cluster_params'] = [(1, 'jaccard-rep-3')]

    ###############################################################################################
    # layers - [name, rf, stride, stride-reconstruct, num_cents, alpha-ltm, alpha-stm, beta,theta #
    ###############################################################################################

    # load mnist/emnist architecture
    if args.dataset == 'mnist' or args.dataset == 'emnist':
        configs['num_phases'] = 5
        if args.model_flag == 1:
            layers = [['L1', 8,  1, 2, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L2', 13, 1, 3, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L3', 20, 1, 4, args.delta, args.alpha, 0.0, args.beta, args.theta]]

    # load cifar/svhn architecture
    if args.dataset == 'svhn':  
        configs['num_phases'] = 5
        if args.model_flag == 1:
            layers = [['L1', 10, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L2', 14, 2, 3, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L3', 18, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta]]
    
    if args.dataset == 'cifar10':
        configs['num_phases'] = 5
        if args.model_flag == 1:
            layers = [['L1', 12, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L2', 18, 2, 3, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L3', 22, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta]]
    
    if args.dataset == 'cifar100' or args.dataset == 'imagenet':
        configs['num_phases'] = 20
        if args.model_flag == 1:
            layers = [['L1', 10, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L2', 14, 2, 3, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L3', 18, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta]]

    if args.dataset == 'core50' or args.dataset == 'l2m' or args.dataset == 'tinyimagenet':
        configs['num_phases'] = 10
        if args.model_flag == 1:
            layers = [['L1', 10, 2, 1, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L2', 18, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta],
                      ['L3', 26, 2, 2, args.delta, args.alpha, 0.0, args.beta, args.theta]]
                      
    # layers to perform classification               
    if args.layers_flag == 0:
        l_eval = []
        l_eval_name = []
    elif args.layers_flag == 1:
        l_eval = [l for l in range(len(layers))]
        l_eval_name = ['all']
    elif args.layers_flag == 2:
        l_eval = [0, 1]
        l_eval_name = ['two']
    elif args.layers_flag == 3:
        l_eval = [0]
        l_eval_name = ['one']

    # stam configs
    configs['layers'] = layers
    configs['num_layers'] = len(layers)
    configs['WTA'] = args.wta
    configs['kernel'] = args.kernel
    configs['rho'] = args.rho
    configs['l_eval'] = l_eval
    configs['l_eval_name'] = l_eval_name

    configs['num_trials'] = args.ntrials
    configs['num_samples'] = args.ntp
    configs['num_classifiers'] = len(configs['classifiers'])
    configs['num_cluster_models'] = len(configs['clustering_models'])
    print(configs['num_cluster_models'])

    return configs


def load_gem_configs(args):

    configs = {}

    # reset random seed
    np.random.seed(0)
    configs['seed'] = 0

    configs['results_directory'] = smart_dir('results/' + args.log + '/' + args.dataset)
    configs['plot_directory'] = smart_dir('plots/' + args.log + '/' + args.dataset)
    configs['visualize_train'] = args.vis
    configs['visualize_cluster'] = args.vis_cluster
    configs['dataset'] = args.dataset
    configs['color_format'] = args.color_format

    # vae/cae configs
    configs['epochs'] = args.epochs
    configs['delta'] = args.delta
    configs['batch_size'] = args.batch
    configs['model_name'] = args.model_name
    configs['num_stages'] = args.num_stages
    configs['lr'] = args.lr
    configs['l_eval'] = []
    configs['l_eval_name'] = []
    
    if args.dataset == 'mnist':
        configs['memory_size'] = int(285 / 5)
    elif args.dataset == 'emnist':
        configs['memory_size'] = int(285 / 23)
    elif args.dataset == 'svhn':
        configs['memory_size'] = int(1210 / 5)
    elif args.dataset == 'cifar-10':
        configs['memory_size'] = int(1515 / 5)
        
    # classifiers
    configs['classifiers'] = ['nn','5nn','5nn-d','nn_N','5nn_N','5nn-d_N']
    configs['clustering_models'] = ['kmeans']

    # Experimental Params for altering each major function
    configs['train_params'] = ()
    configs['supervise_params'] = ()
    configs['classify_params'] = ()
    configs['cluster_params'] = [(0, 'all')]#, (1, 'informative')]


    configs['num_trials'] = args.ntrials
    configs['num_samples'] = args.ntp
    configs['num_classifiers'] = len(configs['classifiers'])
    configs['num_cluster_models'] = len(configs['clustering_models'])

    return configs

def load_mas_configs(args):

    configs = {}

    # reset random seed
    np.random.seed(0)
    configs['seed'] = 0

    configs['results_directory'] = smart_dir('results/' + args.log + '/' + args.dataset)
    configs['plot_directory'] = smart_dir('plots/' + args.log + '/' + args.dataset)
    configs['visualize_train'] = args.vis
    configs['visualize_cluster'] = args.vis_cluster
    configs['dataset'] = args.dataset
    configs['color_format'] = args.color_format


    # vae/cae configs
    configs['epochs'] = args.epochs
    configs['delta'] = args.delta
    configs['batch_size'] = args.batch
    configs['model_name'] = args.model_name
    configs['num_stages'] = args.num_stages
    configs['lr'] = args.lr
    configs['reg_coef'] = args.reg_coef
    configs['l_eval'] = []
    configs['l_eval_name'] = []
    
    # classifiers
    configs['classifiers'] = ['nn','5nn','5nn-d','nn_N','5nn_N','5nn-d_N']
    configs['clustering_models'] = ['kmeans']

    # Experimental Params for altering each major function
    configs['train_params'] = ()
    configs['supervise_params'] = ()
    configs['classify_params'] = ()
    configs['cluster_params'] = [(0, 'all')]#, (1, 'informative')]


    configs['num_trials'] = args.ntrials
    configs['num_samples'] = args.ntp
    configs['num_classifiers'] = len(configs['classifiers'])
    configs['num_cluster_models'] = len(configs['clustering_models'])
    configs['num_images_init'] = args.init_size

    return configs