import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, inPlanes, outPlanes, stride=1):
        super(CBAM, self).__init__()
        self.conv1 = nn.Conv2d(inPlanes, outPlanes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outPlanes)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(outPlanes, outPlanes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outPlanes)
        self.relu2 = nn.LeakyReLU(inplace=True)
        if inPlanes != outPlanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inPlanes, outPlanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outPlanes)
            )
        else:
            self.downsample = nn.Sequential()

        self.ca = ChannelAttention(outPlanes)
        self.sa = SpatialAttention()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.LeakyReLU()

        self.extra = nn.Sequential()
        if out_channel != in_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        identity = self.extra(identity)
        out = out + identity
        out = self.relu(out)
        return out


class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.se = SELayer(outplanes, reduction)
        if outplanes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes)
            )
        else:
            self.downsample = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_class=2):
        super(ResNet18, self).__init__()
        self.blk1 = SEBasicBlock(3, 64, stride=2)
        self.blk2 = SEBasicBlock(64, 128, stride=2)
        self.blk3 = SEBasicBlock(128, 256, stride=2)
        self.out = nn.Linear(256 * 4 * 4, num_class)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


class Reconstructor(nn.Module):
    def __init__(self, latent_dim):
        super(Reconstructor, self).__init__()
        self.dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(256 * 4 * 4, self.dim)
        self.eps = nn.Linear(256 * 4 * 4, self.dim)

        self.linear = nn.Linear(self.dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 2, 2, 0),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        mu = self.mu(x)

        eps = self.eps(x)

        std = torch.exp(0.5 * eps)
        eps = torch.randn_like(std)
        h = mu + eps * std
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(eps, 2) -
            torch.log(1e-8 + torch.pow(eps, 2)) - 1
        ) / (batch_size * 32 * 32)

        x = self.linear(h)
        x = x.view(batch_size, 256, 4, 4)
        x = self.decoder(x)
        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


if __name__ == '__main__':
    #
    # C = ResNet18(2)
    # i = torch.randn((3, 3, 32, 32))
    # b = C(i)
    # print(b.shape)
    # logics = b
    # logics = torch.softmax(logics, dim=1)
    # predict = torch.argmax(logics, dim=1)
    # mask = torch.zeros_like(logics)
    # for id, i in enumerate(predict):
    #     mask[id][1] = 1
    # mask = mask.bool()
    # scores = torch.masked_select(logics, mask)
    # print(scores)
    a = torch.tensor([0.11, 0.23])
    print(torch.softmax(a, dim=0))
    print(torch.softmax(a, dim=0))
