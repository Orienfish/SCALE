import os
import importlib
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
from .supcon import SupCon
from .vicreg import VICReg
from .byol import BYOL
#from torchvision.models import resnet50, resnet18
import torch
#from .backbones import resnet18, resnet50, cnn
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from set_utils import load_student_backbone

def get_backbone(backbone, method, dataset,
                 ckpt=None, castrate=True):
    # backbone_obj = eval(f"{backbone}()")
    backbone_obj = load_student_backbone(backbone,
                                         method,
                                         dataset,
                                         ckpt)

    if backbone == 'cnn':
        backbone_obj.output_dim = 64
    elif backbone == 'resnet18':
        backbone_obj.output_dim = 512
    elif backbone == 'resnet50':
        backbone_obj.output_dim = 2048
    if not castrate:
        backbone_obj.fc = torch.nn.Identity()

    return backbone_obj


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, len_train_loader, transform):
    loss = torch.nn.CrossEntropyLoss()
    if args.model_name == 'simsiam':
        backbone = SimSiam(get_backbone(args.backbone,
                                        args.method,
                                        args.datasetsetting.name,
                                        args.ckpt,
                                        args.cl_default)).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model_name == 'barlowtwins':
        backbone = BarlowTwins(get_backbone(args.backbone,
                                            args.method,
                                            args.datasetsetting.name,
                                            args.ckpt,
                                            args.cl_default
                                            ), device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model_name == 'simclr':
        if args.backbone == 'cnn':
            head = 'linear'
        else:
            head = 'mlp'
        backbone = SupCon(get_backbone(args.backbone,
                                       args.method,
                                       args.datasetsetting.name,
                                       args.ckpt,
                                       args.cl_default,
                                       ),
                          temperature=args.temp_cont, head=head).to(device)

        #if args.model.proj_layers is not None:
        #    backbone.projector.set_layers(args.model.proj_layers)
    elif args.model_name == 'vicreg':
        backbone = VICReg(get_backbone(args.backbone,
                                       args.method,
                                       args.datasetsetting.name,
                                       args.ckpt,
                                       args.cl_default)).to(device)
    elif args.model_name == 'byol':
        backbone = BYOL(get_backbone(args.backbone,
                                     args.method,
                                     args.datasetsetting.name,
                                     args.ckpt,
                                     args.cl_default), device).to(device)
    else:
        raise ValueError('Model name {} is not implemented!'.format(args.model_name))

    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)

    return names[args.method](backbone, loss, args, len_train_loader, transform)
