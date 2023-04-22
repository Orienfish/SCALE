import math
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from networks.resnet_big import ResNet, Bottleneck, BasicBlock, ConvEncoder

def create_model(model_type: str,
                 **kwargs):
    if model_type == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet101':
        ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_type == 'cnn':
        model = ConvEncoder()
    else:
        raise ValueError(model_type)

    return model


def load_student_backbone(model_type: str,
                          ckpt: str,
                          **kwargs):
    """
    Load student model of model_type and pretrained weights from ckpt.
    """
    # Set models
    model = create_model(model_type, **kwargs)

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    if ckpt is not None:
        state = torch.load(ckpt)
        state_dict = {}
        for k in list(state.keys()):
            if k.startswith("fc."):
                continue
            state_dict['module.' + k] = state[k]
            del state[k]
        model.load_state_dict(state_dict)

    return model

def set_constant_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(lr, momentum, weight_decay, model, **kwargs):
    parameters = [{
        'name': 'backbone',
        'params': [param for name, param in model.named_parameters()],
    }]

    # Include the model as heads as part of the parameters to optimize
    if 'criterion' in kwargs:
        parameters = [{
            'name': 'backbone',
            'params': [param for name, param in model.named_parameters()],
        }, {
            'name': 'heads',
            'params': [param for name, param in kwargs['criterion'].named_parameters()],
        }]

    optimizer = optim.SGD(parameters,
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer
