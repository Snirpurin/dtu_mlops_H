import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms




def mnist():
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.FashionMNIST('/home/hkh/git/dtu_mlops/data/corruptmnist/train_0.npz', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # Download and load the test data
    testset = datasets.FashionMNIST('/home/hkh/git/dtu_mlops/data/corruptmnist/test.npz', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader
