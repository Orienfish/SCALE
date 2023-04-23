import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from dataset.tinyimagenet import TinyImagenet


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.'
    Code copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class ThreeCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, size):
        self.transform = transform
        self.notaug_transform = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.notaug_transform(x)]


class SeqSampler(Sampler):
    def __init__(self, dataset, blend_ratio, n_concurrent_classes,
                 train_samples_per_cls):
        """data_source is a Subset"""
        self.num_samples = len(dataset)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        self.train_samples_per_cls = train_samples_per_cls

        # Configure the correct train_subset and val_subset
        if torch.is_tensor(dataset.targets):
            self.labels = dataset.targets.detach().cpu().numpy()
        else:  # targets in cifar10 and cifar100 is a list
            self.labels = np.array(dataset.targets)
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

    def __iter__(self):
        """Sequential sampler"""
        # Configure concurrent classes
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print('cmin', cmin)
        print('cmax', cmax)

        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))

        # Configure sequential class-incremental input
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = filter_fn(self.labels)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)

            cls_idx = self.classes.index(c)
            if len(self.train_samples_per_cls) == 1:  # The same length for all classes
                sample_num = self.train_samples_per_cls[0]
            else:  # Imbalanced class
                assert len(self.train_samples_per_cls) == len(self.classes), \
                    'Length of classes {} does not match length of train ' \
                    'samples per class {}'.format(len(self.classes),
                                                  len(self.train_samples_per_cls))
                sample_num = self.train_samples_per_cls[cls_idx]

            sample_idx.append(filtered_ind.tolist()[:sample_num])
            print('Class [{}, {}): {} samples'.format(cmin[cls_idx], cmax[cls_idx],
                                                      sample_num))

        # Configure blending class
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                # Blend examples from the previous class if not the first
                if c > 0:
                    blendable_sample_num = \
                        int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    # Generate a gradual blend probability
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, \
                        'unmatched sample and probability count'

                    # Exchange with the samples from the end of the previous
                    # class if satisfying the probability, which decays
                    # gradually
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp

        final_idx = []
        for sample in sample_idx:
            final_idx += sample
        return iter(final_idx)

    def __len__(self):
        if len(self.train_samples_per_cls) == 1:
            return self.n_classes * self.train_samples_per_cls[0]
        else:
            return sum(self.train_samples_per_cls)


def set_loader(opt):
    # set seed for reproducing
    random.seed(opt.trial)
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tinyimagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif opt.dataset == 'path':
        mean = opt.mean
        std = opt.std
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=ThreeCropTransform(train_transform, opt.size),
                                         download=True,
                                         train=True)
        knn_train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             train=True,
                                             transform=val_transform)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=ThreeCropTransform(train_transform, opt.size),
                                          download=True,
                                          train=True)
        knn_train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                              train=True,
                                              transform=val_transform)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)

        # Convert sparse labels to coarse labels
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        knn_train_dataset.targets = sparse2coarse(knn_train_dataset.targets)
        val_dataset.targets = sparse2coarse(val_dataset.targets)

    elif opt.dataset == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size,
                                         scale=(0.08, 1.0),
                                         ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size,
                                         scale=(0.08, 1.0),
                                         ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                     transform=ThreeCropTransform(train_transform, opt.size),
                                     train=True,
                                     download=True)
        knn_train_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                         train=True,
                                         transform=val_transform)
        val_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                   train=False,
                                   transform=val_transform)

    elif opt.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.Resize(size=opt.size),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=opt.size),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=ThreeCropTransform(train_transform, opt.size),
                                       download=True,
                                       train=True)
        knn_train_dataset = datasets.MNIST(root=opt.data_folder,
                                           train=True,
                                           transform=val_transform)
        val_dataset = datasets.MNIST(root=opt.data_folder,
                                     train=False,
                                     transform=val_transform)

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=val_transform)
        knn_train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=val_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                           transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    # Configure the a smaller subset as validation dataset
    if torch.is_tensor(train_dataset.targets):
        labels = train_dataset.targets.detach().cpu().numpy()
    else:  # targets in cifar10 and cifar100 is a list
        labels = np.array(train_dataset.targets)
    num_labels = len(list(set(labels)))

    # Create training loader
    if opt.training_data_type == 'iid':
        train_subset_len = num_labels * opt.train_samples_per_cls[0]
        train_subset, _ = torch.utils.data.random_split(dataset=train_dataset,
                                                        lengths=[train_subset_len,
                                                                 len(train_dataset) - train_subset_len])
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
    else:  # sequential
        train_sampler = SeqSampler(train_dataset, opt.blend_ratio,
                                   opt.n_concurrent_classes,
                                   opt.train_samples_per_cls)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    # Create validation loader
    val_subset_len = num_labels * opt.test_samples_per_cls
    val_subset, _ = torch.utils.data.random_split(dataset=val_dataset,
                                                  lengths=[val_subset_len, len(val_dataset) - val_subset_len])
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=opt.val_batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    # Create kNN loader
    if opt.knn_samples > 0:
        knn_subset, _ = torch.utils.data.random_split(dataset=knn_train_dataset,
                                                      lengths=[opt.knn_samples, len(knn_train_dataset) - opt.knn_samples])
        knn_train_loader = torch.utils.data.DataLoader(knn_subset,
                                                      batch_size=opt.val_batch_size,
                                                      shuffle=False,
                                                      num_workers=0, pin_memory=True)
    else:
        knn_train_loader = None

    print('Training samples: ', len(train_loader) * opt.batch_size)
    print('Testing samples: ', len(val_loader) * opt.val_batch_size)
    print('kNN training samples: ', len(knn_train_loader) * opt.val_batch_size)

    return train_loader, val_loader, knn_train_loader, train_transform_runtime
