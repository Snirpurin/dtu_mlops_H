import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

def test_data():
    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    x_dim: 784
    train_dataset = MNIST('~/datasets',
                            transform=mnist_transform,
                            train=True,
                            download=True)
    test_dataset = MNIST('~/datasets',
                            transform=mnist_transform,
                            train=False,
                            download=True)
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=False)
    assert len(train_dataset) == 60000
    assert train_loader.batch_size == 16
    assert train_dataset.data.shape == torch.Size([60000, 28, 28])



    assert len(test_dataset) == 10000
    assert test_loader.batch_size == 16
    assert test_dataset.data.shape == torch.Size([10000, 28, 28]), "Checking shape"



    assert test_dataset.classes == [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - six",
            "7 - seven",
            "8 - eight",
            "9 - nine",
        ]



