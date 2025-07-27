import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from augment import AugmentationSpace

"""
Based on AutoAugment implementation
GitHub: https://github.com/DeepVoltaire/AutoAugment
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class Actor_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.num_sub_policies = 3 # Need that, mismatch before
        self.ops_per_sub_policy = 2 # Need that, mismatch before
        self.total_ops = self.num_sub_policies * self.ops_per_sub_policy  # 6 operations total
        
        """
        This is the new actor. Instead of pushing all into one vector, the actions are seperated now,
        with a shared sequential layer to extract the features from the fiven states.
        """
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate heads for each parameter type (Look into augmentation space)
        self.operation_heads = nn.Linear(hidden_dim, 6 * self.total_ops)    # 6 ops
        self.magnitude_heads = nn.Linear(hidden_dim, 5 * self.total_ops)    # 5 magnitudes
        self.probability_heads = nn.Linear(hidden_dim, 10 * self.total_ops) # 10 prob
        
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        
        # Get logits for each parameter type
        op_logits = self.operation_heads(features).view(-1, self.total_ops, 6) # operations
        mag_logits = self.magnitude_heads(features).view(-1, self.total_ops, 5) # magnitude
        prob_logits = self.probability_heads(features).view(-1, self.total_ops, 10) # probability decoding
        
        value = self.critic(features)
        
        return op_logits, mag_logits, prob_logits, value
    
    def get_action(self, state):
        if len(state.shape) == 1: # In case only one state, as in early training
            state = state.unsqueeze(0)
            
        op_logits, mag_logits, prob_logits, value = self.forward(state)
        
        # Sample from each distribution
        op_dists = Categorical(logits=op_logits)
        mag_dists = Categorical(logits=mag_logits)
        prob_dists = Categorical(logits=prob_logits)
        
        op_actions = op_dists.sample()
        mag_actions = mag_dists.sample()
        prob_actions = prob_dists.sample()
        
        # Get log probabilities
        op_log_probs = op_dists.log_prob(op_actions)
        mag_log_probs = mag_dists.log_prob(mag_actions)
        prob_log_probs = prob_dists.log_prob(prob_actions)
        
        # Combine actions and log probs
        actions = torch.stack([op_actions, mag_actions, prob_actions], dim=-1)
        log_probs = op_log_probs + mag_log_probs + prob_log_probs
        
        return actions.squeeze(0), log_probs.squeeze(0), value.squeeze(0)
    
    def compute_log_probs(self, state, actions):
        """Compute log probabilities for given actions"""
        op_logits, mag_logits, prob_logits, _ = self.forward(state)
        
        op_actions = actions[:, :, 0]
        mag_actions = actions[:, :, 1] 
        prob_actions = actions[:, :, 2]
        
        op_dists = Categorical(logits=op_logits)
        mag_dists = Categorical(logits=mag_logits)
        prob_dists = Categorical(logits=prob_logits)
        
        op_log_probs = op_dists.log_prob(op_actions)
        mag_log_probs = mag_dists.log_prob(mag_actions)
        prob_log_probs = prob_dists.log_prob(prob_actions)

        return op_log_probs + mag_log_probs + prob_log_probs

class AutoAugmentPolicy:
    def __init__(self, num_sub_policies=3, ops_per_sub_policy=2):
        self.num_sub_policies = num_sub_policies
        self.ops_per_sub_policy = ops_per_sub_policy
        
    def decode_actions(self, actions):
        """Convert RL actions to augmentation policy
        actions shape: [6, 3] where 6 = num_operations, 3 = [op_idx, mag_idx, prob_idx]
        """
        policy = []
        
        for i in range(self.num_sub_policies):
            sub_policy = []
            for j in range(self.ops_per_sub_policy):
                action_idx = i * self.ops_per_sub_policy + j
                
                op_idx = actions[action_idx, 0].item()
                mag_idx = actions[action_idx, 1].item()
                prob_idx = actions[action_idx, 2].item()
                
                sub_policy.append({
                    'operation': op_idx,
                    'magnitude': mag_idx,
                    'probability': prob_idx / 10.0  # Convert to 0-1 probability
                })
            policy.append(sub_policy)
        
        return policy

# Updated PPO Agent
class PPOAgent(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, lr=1e-3, clip_epsilon=0.2, epochs=4):
        super().__init__()
        self.actor_critic = Actor_Critic(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
    def update(self, states, actions, old_log_probs, rewards, advantages):
        """PPO update step"""
        states = torch.stack(states) if not isinstance(states, torch.Tensor) else states
        actions = torch.stack(actions) if not isinstance(actions, torch.Tensor) else actions
        old_log_probs = torch.stack(old_log_probs) if not isinstance(old_log_probs, torch.Tensor) else old_log_probs
        
        rewards = rewards.repeat_interleave(states.shape[0])
        
        for _ in range(self.epochs):
            # Get current policy predictions
            new_log_probs = self.actor_critic.compute_log_probs(states, actions)
            _, _, _, values = self.actor_critic(states)
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))
            print(f"Ratio mean/std: {ratio.mean():.6f}/{ratio.std():.6f}")
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
            
        return actor_loss.item(), critic_loss.item()
    
def collect_state_information(epoch, total_epochs, val_acc, train_loss, val_loss, lr, recent_train_loss, recent_val_accs, device='cuda'):
    if len(recent_train_loss) >= 5:
        train_trend = np.mean(recent_train_loss[-5:]) - train_loss
        val_trend = np.mean(recent_val_accs[-5:]) - val_acc
    else:
        train_trend = 0
        val_trend = 0
        
    state_info_tensor = torch.tensor([
        epoch / total_epochs,
        val_acc, 
        train_loss,
        val_loss,
        train_loss - val_loss,
        lr,
        train_trend,  # Recent trend
        val_trend,
        min(1.0, train_loss / 2.0) #. Normalized loss
    ], device=device, dtype=torch.float32)
    
    return state_info_tensor

def advantage_computation(rewards, values, device, gamma=0.99):
    """
    From our exercise implementation (simplified)
    """
    rewards = rewards[-len(values):] # Fix shape mismatch in ppo update
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Compute discounted returns
    returns = []
    discounted_sum = 0
    
    # Work backwards
    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    
    # Advantage = return - baseline (value estimate)
    advantages = returns - values
    
    # # Normalize advantages # Removed because std dev caused problems
    # if len(advantages) > 1:
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # else:
    #     advantages = advantages - advantages.mean()
    
    return advantages, returns, rewards