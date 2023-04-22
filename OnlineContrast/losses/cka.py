from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CKALoss(nn.Module):
    """Centered Kernel Alignment (CKA) Loss for distilling the knowledge from
       the teacher model and apply on the student model
        https://arxiv.org/abs/1203.0550
    """
    def __init__(self):
        super(CKALoss, self).__init__()

    def center_gram(sellf, gram, unbiased=False):
        """Center a symmetric Gram matrix.
        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.
        Args:
          gram: A num_examples x num_examples symmetric matrix tensor.
          unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.
        Returns:
          A symmetric matrix tensor with centered columns and rows.
        """
        if not torch.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.clone()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            gram.fill_diagonal_(0)
            means = torch.sum(gram, dim=0, dtype=torch.float64) / (n - 2)
            means = means - torch.sum(means) / (2 * (n - 1))
            means = means.detach()
            gram = gram - means[:, None]
            gram = gram - means[None, :]
            gram.fill_diagonal_(0)
        else:
            means = torch.mean(gram, dim=0, dtype=torch.float64)
            means = means - torch.mean(means) / 2
            means = means.detach()
            gram = gram - means[:, None]
            gram = gram - means[None, :]

        return gram

    def cka(self, gram_x, gram_y, debiased=False):
        """Compute CKA.
        Args:
          gram_x: A num_examples x num_examples Gram matrix.
          gram_y: A num_examples x num_examples Gram matrix.
          debiased: Use unbiased estimator of HSIC. CKA may still be biased.
        Returns:
          The value of CKA between X and Y.
        """
        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = torch.linalg.norm(gram_x)
        normalization_y = torch.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch, gram_stu=None):
        """Compute loss for model
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
            gram_stu: tensors of pre-computed student, if not None
        Returns:
            A loss scalar.
        """
        if gram_stu is None:
            z_stu = F.normalize(backbone_stu(x_stu), dim=1)
            gram_stu = torch.matmul(z_stu, z_stu.T)

        z_tch = F.normalize(backbone_tch(x_tch), dim=1)
        gram_tch = torch.matmul(z_tch, z_tch.T)

        loss = self.cka(gram_stu, gram_tch)

        # return - torch.log(loss), gram_stu, gram_tch
        return - loss, gram_stu, gram_tch