import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from augment import AugmentationSpace

"""
Based on AutoAugment implementation
GitHub: https://github.com/DeepVoltaire/AutoAugment
"""

class Actor_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        # Actor implementation
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic implementation
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
    
    def get_action(self, state):
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

class PPOAgent:
    """
    Standard implementation of PPO tailored to DAC.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lr=3e-4, clip_epsilon=0.2, epochs=4):
        self.actor_critic = Actor_Critic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
    def update(self, states, actions, old_log_probs, rewards, advantages):
        """PPO update step"""
        for _ in range(self.epochs):
            # Get current policy predictions
            action_logits, values = self.actor_critic(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

class AutoAugmentPolicy:
    """Represents an augmentation policy with multiple sub-policies"""
    def __init__(self, num_sub_policies=3, ops_per_sub_policy=2):
        self.num_sub_policies = num_sub_policies
        self.ops_per_sub_policy = ops_per_sub_policy
        self.aug_space = AugmentationSpace()
        
    def decode_actions(self, actions):
        """Convert RL actions to augmentation policy"""
        # actions should be a tensor of shape [num_sub_policies * ops_per_sub_policy * 3]
        # Each operation needs [operation_idx, magnitude_idx, probability]

        policy = []
        # That ones kinda tricky
        for i in range(self.num_sub_policies):
            sub_policy = []
            for j in range(self.ops_per_sub_policy):
                base_idx = (i * self.ops_per_sub_policy + j) * 3
                op_idx = actions[base_idx] % self.aug_space.num_ops
                mag_idx = actions[base_idx + 1] % self.aug_space.num_magnitudes
                prob = actions[base_idx + 2] / 10.0  # Convert to probability
                
                # Short exmaple for understanding:
                # Lets say actor says [15, 23, 7, 3, 12, 8, ...]
                # op_idx = 15 % 6 = 3
                # mag_idx = 23 % 5 = 3
                # prob = 7 / 10.0 = 0.7
                # Sub-policy 1
                # {'operation': 7, 'magnitude': 3, 'probability': 0.7},  # Cutout at 70%
                
                sub_policy.append({
                    'operation': op_idx.item(),
                    'magnitude': mag_idx.item(),
                    'probability': prob.item()
                })
            policy.append(sub_policy)
        
        return policy
    
def collect_state_information(epoch, total_epochs, train_acc, val_acc, train_loss, val_loss, lr, recent_train_accs, recent_val_accs, device='cuda'):
    state_info_tensor = torch.tensor([
        epoch / total_epochs,
        train_acc,
        val_acc, 
        train_loss,
        val_loss,
        lr,
        train_acc - val_acc,
        np.mean(recent_train_accs[-5:]) - train_acc,  # Recent trend
        np.mean(recent_val_accs[-5:]) - val_acc,
        min(1.0, train_loss / 2.0) #. Normalized loss
    ], device=device)
    
    return state_info_tensor