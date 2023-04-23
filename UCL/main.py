import torch
import random
import tensorboard_logger as tb_logger
from arguments import get_args
# from augmentations import get_aug
from models import get_model
from utils.loggers import *

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from eval_utils import cluster_eval, knn_eval, save_model, plot_mem
from data_utils import set_loader

VAL_CNT = 10  # Number of validations to allow during training
SAVE_MODEL = False # Whether to save model for offline linear evaluation

def validate(val_loader, knn_train_loader, model, optimizer_stream, opt, mem,
             cur_step, epoch, logger, task_list):
    """validation, evaluate k-means clustering accuracy and plot t-SNE"""
    model.eval()
    test_labels, knn_labels = [], []
    test_embeddings, knn_embeddings = None, None

    for idx, (images, labels) in enumerate(knn_train_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        embeddings = model(images).detach().cpu().numpy()
        if knn_embeddings is None:
            knn_embeddings = embeddings
        else:
            knn_embeddings = np.concatenate((knn_embeddings, embeddings), axis=0)
        knn_labels += labels.detach().tolist()
    knn_labels = np.array(knn_labels).astype(int)

    for idx, (images, labels) in enumerate(val_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

        # forward prediction
        embeddings = model(images).detach().cpu().numpy()
        if test_embeddings is None:
            test_embeddings = embeddings
        else:
            test_embeddings = np.concatenate((test_embeddings, embeddings), axis=0)
        test_labels += labels.detach().tolist()
    test_labels = np.array(test_labels).astype(int)

    # Unsupervised clustering
    cluster_eval(test_embeddings, test_labels, opt, mem, cur_step, epoch, logger)

    # kNN classification
    knn_eval(test_embeddings, test_labels, knn_embeddings, knn_labels,
             opt, mem, cur_step, epoch, logger)
    #knn_task_eval(test_embeddings, test_labels, knn_embeddings, knn_labels,
    #              opt, mem, cur_step, epoch, logger, task_list)

    # Memory plot
    if opt.plot and opt.mem_size_per_class > 0 and cur_step > 0:
        plot_mem(mem, model, opt, epoch, cur_step)

    # Save the current embeddings
    # np.save(os.path.join(opt.save_folder, 'test_embeddings.npy'), test_embeddings)
    # np.save(os.path.join(opt.save_folder, 'test_labels.npy'), test_labels)

    # Save the current model
    if SAVE_MODEL:
        save_file = os.path.join(opt.save_folder, '{}_{}.pth'.format(epoch, cur_step))
        save_model(model, optimizer_stream, opt, opt.epochs, save_file)


def main(device, args):
    #dataset = get_dataset(args)
    #dataset_copy = get_dataset(args)
    #train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    # define loader
    # build data loader
    train_loader, test_loader, knntrain_loader, train_transform = set_loader(args)

    # define model
    model = get_model(args, device, len(train_loader), train_transform)

    # logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    # accuracy = 0
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    #for t in range(dataset.N_TASKS):
      #train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)

    # Set task list
    all_labels = []
    for idx, (images, labels) in enumerate(test_loader):
        all_labels += labels.detach().tolist()
    all_labels = np.sort(np.unique(np.array(all_labels).astype(int)))
    task_list = np.reshape(all_labels, (-1, all_labels.size)).tolist()
    print('task list', task_list)

    # global_progress = tqdm(range(0, args.epochs), desc=f'Training')
    cur_step = 0
    for epoch in range(1, args.epochs + 1):
        # results, results_mask_classes = [], []

        # Set validation frequency
        val_freq = np.floor(len(train_loader) / VAL_CNT).astype('int')
        print('Validation frequency: {}'.format(val_freq))

        # local_progress = tqdm(train_loader, disable=args.hide_progress)
        for idx, (images, labels) in enumerate(train_loader):
            # Unsupervised validator
            if idx % val_freq == 0:
                validate(test_loader, knntrain_loader, model, model.opt,
                         args, None, cur_step, epoch, logger, task_list)

            model.train()
            images1, images2, notaug_images = images[0], images[1], images[2]
            for _ in range(args.steps_per_batch_stream):
                data_dict = model.observe(images1, labels, images2, notaug_images)
                cur_step += 1

                if cur_step % args.print_freq == 0:
                    print('[{0}][{1}][{2}/{3}] loss {loss} penalty {penalty} lr {lr}'.format(
                        epoch, cur_step, idx+1, len(train_loader),
                        loss=data_dict['loss'], penalty=data_dict['penalty'], lr=model.opt.param_groups[0]['lr']
                    ))
                logger.log_value('loss', data_dict['loss'], cur_step)
                    # logger.update_scalers(data_dict)

        # Validation at the end of epoch
        validate(test_loader, knntrain_loader, model, model.opt,
                 args, None, cur_step, epoch, logger, task_list)


        # global_progress.set_postfix(data_dict)

        #if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
        #    for i in range(len(dataset.test_loaders)):
        #      acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
        #      results.append(acc)
        #    mean_acc = np.mean(results)

        # epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
        # global_progress.set_postfix(epoch_dict)
        # logger.update_scalers(epoch_dict)

    # Supervised validator
    #if args.cl_default:
    #    accs = evaluate(model.net.module.backbone, dataset, device)
    #    results.append(accs[0])
    #    results_mask_classes.append(accs[1])
    #    mean_acc = np.mean(accs, axis=1)
    #    print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

    model_path = os.path.join(args.model_path, f"{args.method}_{args.name}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.net.state_dict()
    }, model_path)
    print(f"Task Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    #if hasattr(model, 'end_task'):
    #    model.end_task(dataset)

    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path

if __name__ == "__main__":
    args = get_args()

    print("============================================")
    print(args)
    print("============================================")

    # set seed for reproducing
    random.seed(args.trial)
    np.random.seed(args.trial)
    torch.manual_seed(args.trial)

    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


