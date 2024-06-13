'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512, triplet=True, cosine=False):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64
        self.triplet = triplet
        self.cosine = cosine

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd
    
    def double_loss(self, z_p, z_n, y = 1, margin=0.1, l2=0):
        dist = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        dist_margin = F.relu(margin - dist)
        loss = F.relu(y * dist + (1 - y) * dist_margin)
        l_n = torch.mean(dist)
        l_d = torch.mean(dist_margin)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n))
        return loss, l_n, l_d
    
    def cosine_loss(self, z_p, z_n, y = None, margin=0.1, l2=0):
        y = torch.ones(z_p.size(0), device=z_p.device)
        l_n = nn.CosineEmbeddingLoss(margin=margin)(z_p, z_n, y)
        if self.triplet:
            l_d = nn.CosineEmbeddingLoss(margin=margin)(z_p, z_n, -y)
            loss = F.relu(l_n + l_d)
        else:
            loss = l_n
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n))
            
        if self.triplet:
            return loss, l_n, l_d
        return loss, l_n, torch.tensor([0]).cuda()
        
        

    def loss(self, patch, neighbor, distant=None, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n = (self.encode(patch), self.encode(neighbor))
        z_d = None
        if self.triplet:
            z_d = self.encode(distant)
        if self.cosine:
            return self.cosine_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        
        if self.triplet:
            return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        return self.double_loss(z_p, z_n, margin=margin, l2=l2)


def make_tilenet(in_channels=4, z_dim=512, triplet=True, cosine=False):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels, feature dimension, specifies also if triplets or only two images are used during training.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim, triplet=triplet, cosine=cosine)

