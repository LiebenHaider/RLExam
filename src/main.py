import os
import torch
from torchvision.datasets import CIFAR10

from resnet import ResNet18

def main():
    # Create root directory if not already exists
    DATAPATH = "RLExam/data/"
    os.makedirs(DATAPATH, exist_ok=True)
    dataset = CIFAR10(root=DATAPATH, download=True)
    
    # Set rng with seeds
    torch.manual_seed(0)
    
    # Create model
    resnet = ResNet18()
    
if __name__ == "__main__":
    main()