from __future__ import print_function

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


model_dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'cnn': 128
}

class CaSSLe_Predictor(nn.Module):
    def __init__(self, distill_proj_hidden_dim,
                 dim_in=128):
        super().__init__()
        self.distill_predictor = nn.Sequential(
            nn.Linear(dim_in, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, dim_in),
        )

    def forward(self, x):
        return self.distill_predictor(x)


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


class SimCLRLoss(nn.Module):
    """Two-way SimCLR loss"""
    def __init__(self,
                 model='resnet50',
                 lifelong_method='none',
                 temperature=0.07,
                 base_temperature=0.07,
                 distill_lamb=1,
                 distill_proj_hidden_dim=2048,
                 distill_temperature=0.2
                 ):
        super(SimCLRLoss, self).__init__()
        self.device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distill_lamb = distill_lamb
        self.distill_temperature = distill_temperature

        dim_in = model_dim_dict[model]
        self.projector = projection_MLP(dim_in)

        self.cassle = (lifelong_method == 'cassle')
        if self.cassle:
            self.cassle_predictor = CaSSLe_Predictor(distill_proj_hidden_dim)
        self.frozen_backbone = None

    def freeze_backbone(self, backbone):
        self.frozen_backbone = copy.deepcopy(backbone)
        set_requires_grad(self.frozen_backbone, False)

    def loss(self, z_stu, z_tch, temperature=None):
        batch_size = z_stu.shape[0]

        all_features = torch.cat((z_stu, z_tch), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            temperature)  # (2*bsz, 2*bsz)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # did not protect mask.sum(1) to be zero, as it should not be
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()  # loss.view(2, batch_size).mean()

        return loss

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch):
        """Compute loss for model
        The arguments format is designed to align with other losses.
        In SimCLR, the two backbones should be the same
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
        Returns:
            A loss scalar.
        """
        z_stu = F.normalize(self.projector(backbone_stu(x_stu)), dim=1)
        z_tch = F.normalize(self.projector(backbone_tch(x_tch)), dim=1)

        loss = self.loss(z_stu, z_tch, temperature=self.temperature)

        # compute the cassle distillation loss if using cassle
        if self.cassle:
            assert self.frozen_backbone is not None, 'frozen encoder has not been created yet'
            p_stu = F.normalize(self.cassle_predictor(z_stu), dim=1)
            p_tch = F.normalize(self.cassle_predictor(z_tch), dim=1)
            p = torch.cat([p_stu, p_tch], dim=1)
            z_stu_frozen = F.normalize(self.projector(self.frozen_backbone(x_stu)), dim=1)
            z_tch_frozen = F.normalize(self.projector(self.frozen_backbone(x_tch)), dim=1)
            z = torch.cat([z_stu_frozen, z_tch_frozen], dim=1)
            loss_distill = (self.loss(p, z, temperature=self.distill_temperature) +
                            self.loss(z, p, temperature=self.distill_temperature)) / 2

            print('cassle loss: {} loss_distill: {}'.format(loss.item(), loss_distill.item()))

            return loss + self.distill_lamb * loss_distill

        return loss
