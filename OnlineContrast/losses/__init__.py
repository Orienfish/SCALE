import torch
from .supcon import SupConLoss, IRDLoss
from .simclr import SimCLRLoss

def get_loss(opt):
    if opt.criterion == 'simclr':
        criterion = SimCLRLoss(model=opt.model,
                               lifelong_method=opt.lifelong_method,
                               temperature=opt.temp_cont)
    elif opt.criterion == 'supcon':
        criterion = SupConLoss(stream_bsz=opt.batch_size,
                               model=opt.model,
                               temperature=opt.temp_cont)
    else:
        raise ValueError('loss method not supported: {}'.format(opt.criterion))

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    criterion_reg = IRDLoss(projector=criterion.projector,
                            current_temperature=opt.current_temp,
                            past_temperature=opt.past_temp)
    if torch.cuda.is_available():
        criterion_reg = criterion_reg.cuda()

    return criterion, criterion_reg
