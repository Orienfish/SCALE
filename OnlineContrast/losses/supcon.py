from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

model_dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'cnn': 128
}

class projection_MLP(nn.Module):
    def __init__(self, dim_in, head='mlp', feat_dim=128):
        super().__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.head(x)
        return feat

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,
                 stream_bsz,
                 model='resnet50',
                 temperature=0.07,
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.stream_bsz = stream_bsz
        self.temperature = temperature
        self.base_temperature = base_temperature

        dim_in = model_dim_dict[model]
        self.projector = projection_MLP(dim_in)

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        The arguments format is designed to align with other losses.
        In SimCLR, the two backbones should be the same
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if x_stu.is_cuda
                  else torch.device('cpu'))

        z_stu = F.normalize(self.projector(backbone_stu(x_stu)), dim=1)
        z_tch = F.normalize(self.projector(backbone_tch(x_tch)), dim=1)

        batch_size = x_stu.shape[0]

        all_features = torch.cat((z_stu, z_tch), dim=0)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        # print(mean_log_prob_pos.shape, mean_log_prob_pos.max().item(), mean_log_prob_pos.mean().item(), mean_log_prob_pos.min().item())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batch_size)
        stream_mask = torch.zeros_like(loss).float().to(device)
        stream_mask[:, :self.stream_bsz] = 1
        loss = (stream_mask * loss).sum() / stream_mask.sum()
        return loss


class IRDLoss(nn.Module):
    """Instance-wise Relation Distillation (IRD) Loss for Contrastive Continual Learning
        https://arxiv.org/pdf/2106.14413.pdf
    """
    def __init__(self, projector, current_temperature=0.2,
                 past_temperature=0.01):
        super(IRDLoss, self).__init__()
        self.projector = projector
        self.curr_temp = current_temperature
        self.past_temp = past_temperature

    def forward(self, backbone, past_backbone, x):
        """Compute loss for model.
        Args:
            backbone: current backbone
            past_backbone: past backbone
            x: raw input of shape [bsz * n_views, ...]
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if x.is_cuda
                  else torch.device('cpu'))

        cur_features = F.normalize(self.projector(backbone(x)), dim=1)
        past_features = F.normalize(self.projector(past_backbone(x)), dim=1)

        cur_features_sim = torch.div(torch.matmul(cur_features, cur_features.T),
                                     self.curr_temp)
        logits_mask = torch.scatter(
            torch.ones_like(cur_features_sim),
            1,
            torch.arange(cur_features_sim.size(0)).view(-1, 1).to(device),
            0
        )
        cur_logits_max, _ = torch.max(cur_features_sim * logits_mask, dim=1, keepdim=True)
        cur_features_sim = cur_features_sim - cur_logits_max.detach()
        row_size =cur_features_sim.size(0)
        cur_logits = torch.exp(cur_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            cur_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        # print('cur_logits', cur_logits * 1e4)

        past_features_sim = torch.div(torch.matmul(past_features, past_features.T), self.past_temp)
        past_logits_max, _ = torch.max(past_features_sim * logits_mask, dim=1, keepdim=True)
        past_features_sim = past_features_sim - past_logits_max.detach()
        past_logits = torch.exp(past_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            past_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        loss_distill = (- past_logits * torch.log(cur_logits)).sum(1).mean()
        #return loss_distill

        return cur_logits, loss_distill


def PairEnum(x, y):
    """ Enumerate all pairs of feature in x and y"""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x = x.repeat(x.size(0), 1)
    y = y.repeat(1, y.size(0)).view(-1, y.size(1))
    return x, y


def similarity_mask_new(batch_size, cur_digits, opt, pos_pairs):
    device = (torch.device('cuda')
              if cur_digits.is_cuda
              else torch.device('cpu'))

    bsz = cur_digits.size(0)
    contrast_digits = torch.eye(bsz).to(device)  # Set diagonal of similarity matrix to 1
    logits_mask = torch.scatter(
            torch.ones_like(contrast_digits),
            1,
            torch.arange(contrast_digits.size(0)).view(-1, 1).to(device),
            0
        )
    contrast_digits[logits_mask.bool()] = cur_digits.view(-1)
    contrast_digits = 0.5 * (contrast_digits + torch.transpose(contrast_digits, 0, 1))
    simil_max = contrast_digits[logits_mask.bool()].max()
    simil_mean = contrast_digits[logits_mask.bool()].mean()
    # simil_min = contrast_digits[logits_mask.bool()].min()
    #print('prob_simil_avg: dim {}\tmax {}\tavg {}\tmin {}'.format(
    #    contrast_digits.shape[0], simil_max, simil_mean, simil_min))

    contrast_mask = torch.zeros(cur_digits.size(0), cur_digits.size(0)).to(device)
    simil_thres = simil_mean + opt.thres_ratio * (simil_max - simil_mean)
    #print('simil thres: {} batch_size: {}'.format(simil_thres, batch_size))
    contrast_mask[contrast_digits > simil_thres] = 1
    #print(contrast_mask.sum().item())

    # mask out memory elements
    stream_mask = torch.zeros_like(contrast_mask).float().to(device)
    stream_mask[:batch_size, :batch_size] = 1
    #print(stream_mask)
    contrast_mask = contrast_mask * stream_mask
    #print(contrast_mask.shape, contrast_mask)

    pos_pairs.update(contrast_mask.sum().item() / cur_digits.size(0), 1)
    return contrast_mask


def similarity_mask_old(feat_all, bsz, opt, pos_pairs):
    """Calculate the pairwise similarity and the mask for contrastive learning
    Args:
        feat_all: all hidden features of shape [n_views * bsz, ...].
        bsz: int, batch size of input data (stacked streaming and memory samples)
        opt: arguments
        pos_pairs: averagemeter recording number of positive pairs
    Returns:
        contrast_mask: mask of shape [bsz, bsz]
    """
    #print(feat_all[0])
    #print(feat_all[1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feat_size = feat_all.size(0)
    n_views = int(feat_size / bsz)
    assert (n_views * bsz == feat_size), "Unmatch feature sizes and batch size!"

    # Compute the pairwise distance and similarity between each view
    # and add the similarity together for average
    simil_mat_avg = torch.zeros(bsz, bsz).to(device)
    mat_cnt = 0
    for i in range(n_views):
        for j in range(n_views):
            # feat_row and feat_col should be of size [bsz^2, bsz^2]
            #feat_row, feat_col = PairEnum(feat_all[i*bsz: (i+1)*bsz],
            #                              feat_all[j*bsz: (j+1)*bsz])
            #tmp_distance = -(((feat_row - feat_col) / temperature) ** 2.).sum(1)  # Euclidean distance
            # Note, all features are normalized
            if opt.simil == 'kNN':  # euclidean distance
                simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                                 feat_all[j*bsz: (j+1)*bsz].T)
            elif opt.simil == 'tSNE':  # tSNE similarity
                # compute euclidean distance pairs
                simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                                 feat_all[j*bsz: (j+1)*bsz].T)
                #print('\teuc dist', simil_mat * 1e4)
                tmp_distance = - torch.div(simil_mat, opt.temp_tSNE)
                tmp_distance = tmp_distance - 1000 * torch.eye(bsz).to(device)
                #print('\ttemp dist', tmp_distance * 1e4)
                simil_mat = 0.5 * torch.softmax(tmp_distance, 1) + 0.5 * torch.softmax(tmp_distance, 0)
                #print(torch.softmax(tmp_distance, 1))
                #print('simil_mat', simil_mat)
            else:
                raise ValueError(opt.simil)

            # Add the new probability to the average probability
            simil_mat_avg = (mat_cnt * simil_mat_avg + simil_mat) / (mat_cnt + 1)
            mat_cnt += 1
    #print('simil_mat_avg', simil_mat_avg * 1e4)
    logits_mask = torch.scatter(
        torch.ones_like(simil_mat_avg),
        1,
        torch.arange(simil_mat_avg.size(0)).view(-1, 1).to(device),
        0
    )
    simil_max = simil_mat_avg[logits_mask.bool()].max()
    simil_mean = simil_mat_avg[logits_mask.bool()].mean()
    simil_min = simil_mat_avg[logits_mask.bool()].min()
    #print('prob_simil_avg: dim {}\tmax {}\tavg {}\tmin {}'.format(
    #    simil_mat_avg.shape[0], simil_max, simil_mean, simil_min))
    # Set diagonal of similarity matrix to ones
    masks = torch.eye(bsz).to(device)
    simil_mat_avg = simil_mat_avg * (1 - masks) + masks

    # mask out memory elements
    stream_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    stream_mask[:opt.batch_size, :opt.batch_size] = 1
    simil_mat_avg = simil_mat_avg * stream_mask

    contrast_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    if opt.simil == 'tSNE':
        simil_thres = simil_mean + opt.thres_ratio * (simil_max - simil_mean)
        # print(simil_thres)
        contrast_mask[simil_mat_avg > simil_thres] = 1
    elif opt.simil == 'kNN':
        contrast_mask[:opt.batch_size, :opt.batch_size][
            simil_mat_avg[:opt.batch_size, :opt.batch_size] <
            torch.kthvalue(simil_mat_avg[:opt.batch_size, :opt.batch_size],
                           int(opt.simil_thres), 1, True)[0]] = 1
        contrast_mask[:opt.batch_size, :opt.batch_size][
            simil_mat_avg[:opt.batch_size, :opt.batch_size] <
            torch.kthvalue(simil_mat_avg[:opt.batch_size, :opt.batch_size],
                           int(opt.simil_thres), 0, True)[0]] = 1

    pos_pairs.update(contrast_mask.sum().item() / opt.batch_size, bsz)
    # print('Avg num of positive samples: {}'.format(pos_pairs.val))

    return contrast_mask


def get_similarity_matrix(features, temp):
    """
    Compute the similarity matrix of given features
    Args:
        features: (batch_size, feature_dim), raw features
        temp: a scalar temperature value for features
    Returns:
        simil: (batch_size, batch_size-1), a matrix for similarity between
            any feature pair, the diagonal is omitted
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    features_sim = torch.div(torch.matmul(features, features.T), temp)
    logits_mask = torch.scatter(
        torch.ones_like(features_sim),
        1,
        torch.arange(features_sim.size(0)).view(-1, 1).to(device),
        0
    )
    logits_max, _ = torch.max(features_sim * logits_mask, dim=1,
                              keepdim=True)
    features_sim = features_sim - logits_max.detach()
    row_size = features_sim.size(0)
    logits = torch.exp(
        features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
        features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

    return logits