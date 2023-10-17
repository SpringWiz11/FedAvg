import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


def get_dataset(which: int, data_path: str = "./data"):
    if which == 1:
        tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST(data_path, train=True, download=True, transform=tr)
        testset = MNIST(data_path, train=False, download=True, transform=tr)
    elif which == 2:
        tr = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
        testset = CIFAR10(data_path, train=False, download=True, transform=tr)
    elif which == 3:
        tr = Compose([ToTensor(), Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        trainset = CIFAR100(data_path, train=True, download=True, transform=tr) 
        testset = CIFAR100(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, which: int, val_ratio: float = 0.1 ):

    trainset, testset = get_dataset(which)
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader