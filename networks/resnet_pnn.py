import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.nn.parameter import Parameter
from typing import List
# https://raw.githubusercontent.com/huyvnphan/PyTorch_CIFAR10/master/cifar10_models/resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet18_pnn']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

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


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


###########################################################
# Definitions for PNN ResNet
###########################################################
class AlphaModule(nn.Module):
    def __init__(self, shape):
        super(AlphaModule, self).__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.alpha = Parameter(torch.rand(tuple([1] + list(shape))) * 0.1,
                               requires_grad=True)

    def forward(self, x):
        return x * self.alpha

    def parameters(self, recurse: bool = True):
        yield self.alpha


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.idx = 0
        for module in args:
            self.add_module(str(self.idx), module)
            self.idx += 1

    def append(self, module):
        self.add_module(str(self.idx), module)
        self.idx += 1

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.idx
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlockPnn(BasicBlock):
    """
    The basic block of ResNet. Modified for PNN.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ResNetPNN(ResNet):
    """
    ResNet network architecture modified for PNN.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, old_cols: List[nn.Module] = None,
                 x_shape: torch.Size = None):
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNetPNN, self).__init__(block, num_blocks, num_classes, nf)
        if old_cols is None:
            old_cols = []

        self.old_cols = old_cols
        self.x_shape = x_shape
        # self.classifier = self.linear
        if len(old_cols) == 0:
            return

        assert self.x_shape is not None
        self.in_planes = self.nf
        self.lateral_classifier = nn.Linear(nf * 8, num_classes)
        self.adaptor4 = nn.Sequential(
            AlphaModule((nf * 8 * len(old_cols), 1, 1)),
            nn.Conv2d(nf * 8 * len(old_cols), nf * 8, 1),
            nn.ReLU()
        )
        for i in range(5):
            setattr(self, 'old_layer' + str(i) + 's', ListModule())

        for i in range(1, 4):
            factor = 2 ** (i - 1)
            setattr(self, 'lateral_layer' + str(i + 1),
                    self._make_layer(block, nf * (2 ** i), num_blocks[i], stride=2)
                    )
            setattr(self, 'adaptor' + str(i),
                    nn.Sequential(
                        AlphaModule((nf * len(old_cols) * factor,
                                     self.x_shape[2] // factor, self.x_shape[3] // factor)),
                        nn.Conv2d(nf * len(old_cols) * factor, nf * factor, 1),
                        nn.ReLU(),
                        getattr(self, 'lateral_layer' + str(i + 1))
                    ))
        for old_col in old_cols:
            self.in_planes = self.nf
            self.old_layer0s.append(conv3x3(3, nf * 1))
            self.old_layer0s[-1].load_state_dict(old_col.conv1.state_dict())
            for i in range(1, 5):
                factor = (2 ** (i - 1))
                layer = getattr(self, 'old_layer' + str(i) + 's')
                layer.append(self._make_layer(block, nf * factor,
                                              num_blocks[i - 1], stride=(1 if i == 1 else 2)))
                old_layer = getattr(old_col, 'layer' + str(i))
                layer[-1].load_state_dict(old_layer.state_dict())

        # def _make_layer(self, block: BasicBlock, planes: int,
        #                num_blocks: int, stride: int = 1) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """

    #    strides = [stride] + [1] * (num_blocks - 1)
    #    layers = []
    #    for stride in strides:
    #        layers.append(nn.ReLU())
    #        layers.append(block(self.in_planes, planes, stride))
    #        self.in_planes = planes * block.expansion
    #    layers.append(nn.ReLU())
    #    return nn.Sequential(*(layers[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        if self.x_shape is None:
            self.x_shape = x.shape
        if len(self.old_cols) == 0:
            return super(ResNetPNN, self).forward(x)
        else:
            with torch.no_grad():
                out0_old = [relu(self.bn1(old(x))) for old in self.old_layer0s]
                out1_old = [old(out0_old[i]) for i, old in enumerate(self.old_layer1s)]
                out2_old = [old(out1_old[i]) for i, old in enumerate(self.old_layer2s)]
                out3_old = [old(out2_old[i]) for i, old in enumerate(self.old_layer3s)]
                out4_old = [old(out3_old[i]) for i, old in enumerate(self.old_layer4s)]

            out = relu(self.bn1(self.conv1(x)))
            out = F.relu(self.layer1(out))
            y = self.adaptor1(torch.cat(out1_old, 1))
            out = F.relu(self.layer2(out) + y)
            y = self.adaptor2(torch.cat(out2_old, 1))
            out = F.relu(self.layer3(out) + y)
            y = self.adaptor3(torch.cat(out3_old, 1))
            out = F.relu(self.layer4(out) + y)
            out = avg_pool2d(out, out.shape[2])
            out = out.view(out.size(0), -1)

            y = avg_pool2d(torch.cat(out4_old, 1), out4_old[0].shape[2])
            y = self.adaptor4(y)
            y = y.view(out.size(0), -1)
            y = self.lateral_classifier(y)
            out = self.linear(out) + y
        return out


def resnet18_pnn(nclasses: int, nf: int = 64,
                 old_cols: List[nn.Module] = None, x_shape: torch.Size = None):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    if old_cols is None:
        old_cols = []
    return ResNetPNN(BasicBlockPnn, [2, 2, 2, 2], nclasses, nf,
                     old_cols=old_cols, x_shape=x_shape)


