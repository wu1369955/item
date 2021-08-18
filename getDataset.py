import torch
import visdom
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, Caltech101, Caltech256, CIFAR10, CIFAR100, ImageNet
import random
import numpy as np
from torchvision import transforms
from visdom import Visdom
from Tools import *


def getDataset(datasetName, root='.', train=True, download=False, normalClass=0, proportion=1, transform=None):
    inlier = normalClass
    dataset = None
    if datasetName == 'MNIST' or datasetName == 'FashionMNIST':
        dataset = MNIST(root=root, train=train, download=download, transform=transform) if datasetName == 'MNIST' else \
            FashionMNIST(root=root, train=train, download=download, transform=transform)
    if datasetName == 'CIFAR10' or datasetName == 'CIFAR100':
        dataset = CIFAR10(root=root, train=train, download=download, transform=transform) if datasetName == 'CIFAR10' \
            else CIFAR100(root=root, train=train, download=download, transform=transform)

    outlier = list(range(0, len(dataset.classes)))
    outlier.remove(inlier)
    outlier = random.sample(outlier, proportion)
    outlier.append(inlier)

    if train:
        labels = np.argwhere(np.isin(dataset.targets, [inlier])).flatten().tolist()
    else:
        labels = np.argwhere(np.isin(dataset.targets, [outlier])).flatten().tolist()

    dataset = Subset(dataset, labels)
    return dataset


def getDataset_(datasetName, root='.', train=True, download=False, normalClass=0, proportion=1, transform=None):
    inlier = normalClass
    dataset = None
    if datasetName == 'MNIST' or datasetName == 'FashionMNIST':
        dataset = MNIST(root=root, train=train, download=download, transform=transform) if datasetName == 'MNIST' else \
            FashionMNIST(root=root, train=train, download=download, transform=transform)
    if datasetName == 'CIFAR10' or datasetName == 'CIFAR100':
        dataset = CIFAR10(root=root, train=train, download=download, transform=transform) if datasetName == 'CIFAR10' \
            else CIFAR100(root=root, train=train, download=download, transform=transform)

    outlier = list(range(0, len(dataset.classes)))
    outlier.remove(inlier)
    outlier = random.sample(outlier, proportion)
    outlier.append(inlier)

    if train:
        labels = np.argwhere(np.isin(dataset.targets, [inlier])).flatten().tolist()
    else:
        labels = np.argwhere(np.isin(dataset.targets, [outlier])).flatten().tolist()

    dataset = Subset(dataset, labels)
    return dataset


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        lambda x: x.convert('RGB'),  # 扩展为RGB通道
        transforms.ToTensor(),  # 转换为tensor数据
    ])
    normalClass = 0
    trainDataset = getDataset('MNIST', root='.', train=True, download=True, normalClass=0, proportion=1,
                              transform=tf)
    testDataset = getDataset('MNIST', root='.', train=False, download=True, normalClass=0, proportion=4,
                             transform=tf)
    dataLoader = DataLoader(trainDataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    viz = visdom.Visdom()
    ave_img = torch.zeros((1, 3, 32, 32))
    img_count = 0
    for img, _ in dataLoader:
        viz.images(img, 32, win='images')
        img_count += 256
        img = torch.sum(img, dim=0)
        img = torch.unsqueeze(img, dim=0)
        ave_img = torch.add(img, ave_img)
        cur_img = torch.div(img, 256)
        viz.images(cur_img, 1, win='batch_ave_images', opts=dict(title='batch_ave_images'))
    ave_img /= img_count
    viz.images(ave_img, 1, win='ave_images', opts=dict(title='ave_images'))
    ave_img = torch.repeat_interleave(ave_img, 256, dim=0)
    ave_ssim = 0
    batch_count = 0
    for img, _ in dataLoader:
        img_count += 256
        ave_ssim += torch.sum((img - ave_img) ** 2)
        batch_count += 1
    print(f'average ssim : {ave_ssim / (batch_count * 256):.3f}')
