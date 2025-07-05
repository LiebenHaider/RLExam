import os
import torch
import pandas as pd
from torchvision.datasets import CIFAR10

from dataloader import get_dataloaders
from resnet import ResNet18
from train import train_loop

def main():
    # Create root directory if not already exists
    DATAPATH = "src/data"
    METRICS_PATH = "src/data"
    METRICS_FILENAME = "final_metrics.csv"
    os.makedirs(DATAPATH, exist_ok=True)
    
    # Get data
    get_dataloaders(DATAPATH)
    
    # # Set rng with seeds
    # torch.manual_seed(0)
    
    # # Split data & create dataloaders
    # datalaoder_train = ...
    # datalaoder_val = ...
    
    # # Create test model
    # test_data = torch.rand((3, 32, 32))
    # resnet = ResNet18()
    
    # try:
    #     with torch.no_grad():
    #         out = resnet(test_data.unsqueeze(0))
    # except: print("Shape mismatch! Input layer must match B x C x H x W.")
    
    # # Create models to evaluate
    # rl_resnet = ResNet18()      # model trained with rl
    # norl_resnet = ResNet18()    # model trained without rl
    # st_resnet = ResNet18()      # model trained without augmentation
    
    # model_dict = {
    #     'rl': rl_resnet,
    #     'random': norl_resnet,
    #     'none': st_resnet
    # }
    
    # # Create agent
    # agent = ...
    
    # # train models
    # NUM_EPOCHS = 30
    # BATCH_SIZE = 32
    
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'
    # else: device = 'cpu'
    # print(f"Running jobs on {device}.")
    
    # final_metrics = train_loop(model_dict, dataloader_train=datalaoder_train, dataloader_val=datalaoder_val, device=device)
    
    # # Save data
    # final_metrics_df = pd.DataFrame(final_metrics)
    # os.makedirs(METRICS_PATH, exist_ok=True)
    # metrics_path = os.path.join(METRICS_PATH, METRICS_FILENAME)
    # final_metrics_df.to_csv(metrics_path)
    
if __name__ == "__main__":
    main()