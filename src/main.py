import os
import torch
import pandas as pd
import numpy as np
import random

from dataloader import get_dataloaders
from resnet import ResNet18
from agent import PPOAgent
from train import train_loop, final_test

SEEDS = [42, 123, 456]
DATAPATH = "src/data"
METRICS_PATH = "src/data/metrics"

def main():
    
    # Run experiments over multiple seeds
    for seed in SEEDS:
        # Create root directory if not already exists
        METRICS_FILENAME = f"final_metrics_seednr_{seed}.csv"
        TESTMETRICS_FILENAME = f"final_testmetrics_seednr_{seed}.csv"
        AGENTPOLICIES_FILENAME = f"agent_policies_seednr_{seed}.csv"
        os.makedirs(DATAPATH, exist_ok=True)
        
        # Get data, split data & create dataloaders
        trainloader, valloader, testloader = get_dataloaders(DATAPATH, seed)

        # # Set rng with seeds
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        
        # # Create test model
        # test_data = torch.rand((3, 32, 32))
        # resnet = ResNet18()
        
        # try:
        #     with torch.no_grad():
        #         out = resnet(test_data.unsqueeze(0))
        # except: print("Shape mismatch!"
        #         "Input layer must match B x C x H x W.")
        
        # # train models
        # NUM_EPOCHS = 75
        # STATE_DIM = 9
        # ACTION_DIM = 18
        # HIDDEN_DIM = 128
        
        # if torch.cuda.is_available():
        #     device = 'cuda'
        #     torch.cuda.manual_seed(seed=seed)
        # elif torch.mps.is_available():
        #     device = 'mps'
        # else: device = 'cpu'
        # print(f"Running jobs on {device}.")
        
        # # Create models to run in parallel and evaluate
        # rl_resnet = ResNet18()      # model trained with rl
        # rand_resnet = ResNet18()    # model trained without rl
        # no_resnet = ResNet18()      # model trained without augmentation
        
        # # # Create agent
        # agent = PPOAgent(
        #     state_dim=STATE_DIM, 
        #     hidden_dim=HIDDEN_DIM
        # ).to(device)
        
        # model_dict = {
        #     'rl': rl_resnet.to(device),
        #     'random': rand_resnet.to(device),
        #     'none': no_resnet.to(device)
        # }
        
        # ### TRAINING LOOP ###
        # histories, best_scores, policy_trajectory = train_loop(
        #     model_dict, 
        #     dataloader_train=trainloader, 
        #     dataloader_val=valloader, 
        #     device=device, 
        #     agent=agent,
        #     epochs=NUM_EPOCHS
        # )
        # print("Training finished.")
        
        # # Test final performance
        # final_test_metrics = final_test(
        #     models=model_dict, 
        #     testloader=testloader,
        #     device=device
        # )
        
        # print("Testing finished. Saving data...")
        
        # # Create folder and save data
        # # agent_policies = {
        # #     'best_scores': best_scores,
        # #     'policy_trajectory': policy_trajectory
        # # }
        
        # agent_policies_df = pd.DataFrame.from_dict(policy_trajectory)
        # final_metrics_df = pd.DataFrame.from_dict(histories)
        # final_test_metrics_df = pd.DataFrame(final_test_metrics)
        # os.makedirs(METRICS_PATH, exist_ok=True)
        # metrics_path = os.path.join(METRICS_PATH, METRICS_FILENAME)
        # testmetrics_path = os.path.join(METRICS_PATH, TESTMETRICS_FILENAME)
        # agentmetrics_path = os.path.join(METRICS_PATH, AGENTPOLICIES_FILENAME)
        # final_metrics_df.to_csv(metrics_path)
        # final_test_metrics_df.to_csv(testmetrics_path)
        # agent_policies_df.to_csv(agentmetrics_path)
        
        # # Check if savings were successful
        # if (
        #     pd.read_csv(metrics_path).empty is False or 
        #     pd.read_csv(testmetrics_path).empty is False or 
        #     pd.read_csv(agentmetrics_path).empty is False
        # ):
            
        #     print(f"Saving was successful. Data saved to {METRICS_PATH}.")
    
if __name__ == "__main__":
    main()