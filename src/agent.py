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
        self.action_dim = action_dim
        # Actor implementation
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, dtype=torch.float32)
        )
        
        # Critic implementation
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=torch.float32)
        )
    
    def forward(self, state):
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
    
    def get_action(self, state):
        if len(state.shape) >= 1: # Add batch dimension if single state
            state = state.unsqueeze(0)
        action_logits, value = self.forward(state)
        # Problem is action sampling, we need complete action space
        # but also logits for update
        # Solution -> discrete distribution
        actions = []
        log_probs = []
        
        action_logits = action_logits.reshape([state.shape[0] * self.action_dim, 1])

        for i in range(self.action_dim):
            dist = Categorical(logits=action_logits[i:i+1])
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)
            
        return (
            torch.stack(actions), 
            torch.stack(log_probs), 
            value
        )
    
    def compute_log_probs(self, action_logits, actions):
        """Compute log probabilities for given actions"""

        log_probs = []
        action_logits = action_logits.squeeze(1)
        for i in range(self.action_dim):
            dist = Categorical(logits=action_logits[:, i:i+1])
            log_prob = dist.log_prob(actions[:, i])
            log_probs.append(log_prob)
        
        return torch.stack(log_probs, dim=1)

class PPOAgent(nn.Module):
    """
    Standard implementation of PPO tailored to DAC.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lr=3e-4, clip_epsilon=0.2, epochs=4):
        super().__init__()
        self.actor_critic = Actor_Critic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
    def update(self, states, actions, old_log_probs, rewards, advantages):
        """PPO update step"""
        for _ in range(self.epochs):
            # Get current policy predictions
            action_logits, values = self.actor_critic(states)
            new_log_probs = self.actor_critic.compute_log_probs(action_logits, actions)
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            values = values[-1] # Use only final value
            critic_loss = nn.MSELoss()(values, rewards)
            
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
                # {'operation': 4, 'magnitude': 3, 'probability': 0.7},  # Cutout at 70%
                
                sub_policy.append({
                    'operation': op_idx.item(),
                    'magnitude': mag_idx.item(),
                    'probability': prob.item()
                })
            policy.append(sub_policy)
        
        return policy
    
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
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()
    
    return advantages, returns, rewards