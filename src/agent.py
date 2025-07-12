import torch
import torch.nn as nn
from torch.distributions import Categorical

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