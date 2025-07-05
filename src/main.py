import os
import torch
import pandas as pd
# from torchvision.datasets import CIFAR10

from resnet import ResNet18
from train import train_loop

def main():
    # Create root directory if not already exists
    DATAPATH = "RLExam/data/"
    METRICS_PATH = "RLExam/data/final_metrics.csv"
    # os.makedirs(DATAPATH, exist_ok=True)
    # dataset = CIFAR10(root=DATAPATH, download=True)
    
    # Set rng with seeds
    torch.manual_seed(0)
    
    # Create test model
    test_data = torch.rand((3, 32, 32))
    resnet = ResNet18()
    
    try:
        with torch.no_grad():
            out = resnet(test_data.unsqueeze(0))
    except: print("Shape mismatch! Tensor must be B x C x H x W.")
    
    # Create three models to evaluate
    rl_resnet = ResNet18()      # model trained with rl
    norl_resnet = ResNet18()    # model trained without rl
    st_resnet = ResNet18()      # model trained without augmentation
    
    # train models
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    final_metrics = train_loop()
    
    # Save data
    final_metrics_df = pd.DataFrame(final_metrics)
    final_metrics_df.to_csv(METRICS_PATH)
    
if __name__ == "__main__":
    main()