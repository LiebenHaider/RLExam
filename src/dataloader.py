from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def get_dataloaders(data_path, batch_size=32):
    train_dataset = CIFAR10(root=data_path, download=True, train=True, transform=ToTensor())
    test_dataset = CIFAR10(root=data_path, download=True, train=False, transform=ToTensor())
    
    train_len = int(0.9 * len(train_dataset))
    val_len = len(train_dataset) - train_len

    train_data, val_data = random_split(train_dataset, [train_len, val_len])
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader