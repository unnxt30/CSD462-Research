import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    """Standard Actor-Critic Model"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.policy_layer = nn.Linear(64, action_dim)  # Actor
        self.value_layer = nn.Linear(64, 1)  # Critic
    
    def forward(self, x):
        shared = self.shared_layers(x)
        logits = self.policy_layer(shared)
        value = self.value_layer(shared)
        return logits, value

class A3C:
    """Standard A3C Implementation"""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_action(self, state):
        """Select action using standard softmax"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][action]
    
    def update(self, states, actions, rewards, next_states, dones):
        """Train the model using advantage actor-critic update"""
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        
        targets = rewards + 0.99 * next_values * (1 - dones)
        advantages = targets - values
        
        loss_critic = advantages.pow(2).mean()
        
        logits, _ = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -torch.mean(log_probs[range(len(actions)), actions] * advantages.detach())
        
        loss = policy_loss + 0.5 * loss_critic
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training loop for standard A3C
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

a3c = A3C(state_dim, action_dim)
standard_rewards = []

for episode in range(100):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    
    while not done:
        action, _ = a3c.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        
        a3c.update([state], [action], [reward], [next_state], [done])
        state = next_state
        total_reward += reward
    
    standard_rewards.append(total_reward)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), standard_rewards, label="Standard A3C", color='r')
plt.plot(range(1, 101), episode_rewards, label="Weighted A3C", color='b')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Comparison: Standard A3C vs Weighted A3C")
plt.legend()
plt.show() 