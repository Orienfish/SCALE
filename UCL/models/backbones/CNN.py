import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """CNN encoder"""
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 64)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_params(self):
        params = []
        for pp in list(self.parameters()):
          # print(pp[0])
          # if pp.grad is not None:
          params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self):
        grads = []
        for pp in list(self.parameters()):
            # if pp.grad is not None:
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

def cnn(**kwargs):
    return ConvEncoder()