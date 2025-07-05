import os
import torch
import pandas as pd
from torchvision.datasets import CIFAR10

from dataloader import get_dataloaders
from resnet import ResNet18
from train import train_loop, final_test

SEED = 42

def main():
    # Create root directory if not already exists
    DATAPATH = "src/data"
    METRICS_PATH = "src/data"
    METRICS_FILENAME = "final_metrics.csv"
    TESTMETRICS_FILENAME = "final_testmetrics.csv"
    os.makedirs(DATAPATH, exist_ok=True)
    
    # Get data, split data & create dataloaders
    trainloader, valloader, testloader = get_dataloaders(DATAPATH, SEED)
    
    # Set rng with seeds
    torch.manual_seed(SEED)
    
    # Create test model
    test_data = torch.rand((3, 32, 32))
    resnet = ResNet18()
    
    try:
        with torch.no_grad():
            out = resnet(test_data.unsqueeze(0))
    except: print("Shape mismatch!"
            "Input layer must match B x C x H x W.")
    
    # Create models to evaluate
    rl_resnet = ResNet18()      # model trained with rl
    rand_resnet = ResNet18()    # model trained without rl
    no_resnet = ResNet18()      # model trained without augmentation
    
    model_dict = {
        'rl': rl_resnet,
        'random': rand_resnet,
        'none': no_resnet
    }
    
    # # Create agent
    # agent = ...
    
    # # train models
    # NUM_EPOCHS = 1
    
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'
    else: device = 'cpu'
    print(f"Running jobs on {device}.")
    
    # ### TRAINING LOOP ###
    # final_metrics = train_loop(
    #     model_dict, 
    #     dataloader_train=trainloader, 
    #     dataloader_val=valloader, 
    #     device=device, 
    #     agent=agent,
    #     epochs=NUM_EPOCHS
    # )
    
    # Test final performance
    final_test_metrics = final_test(
        models=model_dict, 
        testloader=testloader,
        device=device
    )
    print(final_test_metrics)
    # Create folder and save data
    # final_metrics_df = pd.DataFrame(final_metrics)
    # final_test_metrics_df = pd.DataFrame(final_test_metrics)
    # os.makedirs(METRICS_PATH, exist_ok=True)
    # metrics_path = os.path.join(METRICS_PATH, METRICS_FILENAME)
    # testmetrics_path = os.path.join(METRICS_PATH, TESTMETRICS_FILENAME)
    # final_metrics_df.to_csv(metrics_path)
    # final_test_metrics_df.to_csv(testmetrics_path)
    
if __name__ == "__main__":
    main()