import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd

def load_gocj_data(file_path="GoCJ Google Cloud Jobs Dataset/GoCJ_Dataset_1000.txt"):
    """Load job scheduling data from GoCJ dataset"""
    with open(file_path, 'r') as f:
        execution_times = [float(line.strip()) for line in f.readlines()]
    
    # Create a DataFrame with job information
    df = pd.DataFrame({
        'job_id': range(1, len(execution_times) + 1),
        'execution_time': execution_times,
        # Assign random priorities (1-5) for demonstration
        'priority': np.random.randint(1, 6, size=len(execution_times)),
        # Calculate resource usage based on execution time (normalized)
        'resource_usage': [t/max(execution_times) for t in execution_times]
    })
    return df

def load_google_cloud_data():
    """Load job scheduling data from a CSV file instead of BigQuery"""
    try:
        df = pd.read_csv("google_jobs.csv")
    except FileNotFoundError:
        print("google_jobs.csv not found, using GoCJ dataset instead...")
        df = load_gocj_data()
    return df

def weighted_reward(priority, latency, fairness):
    """Compute reward with priority-aware weighting"""
    # Normalize weights to sum to 1
    w_qos = 0.4      # QoS weight
    w_energy = 0.1   # Energy weight
    w_priority = 0.4 # Priority weight
    w_fairness = 0.1 # Fairness weight
    
    # Verify weights sum to 1
    total_weight = w_qos + w_energy + w_priority + w_fairness
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1, got {total_weight}"
    
    # Normalize inputs to [0,1] range
    norm_priority = priority / 5.0  # Assuming priority is 1-5
    norm_latency = 1 - min(latency, 1.0)  # Convert latency to [0,1] where 1 is best
    norm_fairness = min(fairness, 1.0)  # Cap fairness at 1
    
    # Calculate reward ensuring non-negative values
    reward = (w_qos * norm_latency) + (w_energy * 0.1) + (w_priority * norm_priority) - (w_fairness * norm_fairness)
    return max(0, reward)  # Ensure non-negative reward

def standard_reward(latency):
    """Compute reward with normalized QoS and energy weights that sum to 1"""
    # Normalize weights to sum to 1
    w_qos = 0.7    # QoS weight (70%)
    w_energy = 0.3 # Energy weight (30%)
    
    # Verify weights sum to 1
    total_weight = w_qos + w_energy
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1, got {total_weight}"
    
    # Normalize latency to [0,1] range where 1 is best
    norm_latency = 1 - min(latency, 1.0)
    
    # Calculate reward ensuring non-negative values
    reward = (w_qos * norm_latency) + (w_energy * 0.1)
    return max(0, reward)  # Ensure non-negative reward

class ActorCritic(nn.Module):
    """Actor-Critic Model with priority-weighted policy"""
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

class WeightedA3C:
    """A3C with Job Prioritization & Fairness Constraints"""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99  # Discount factor
        self.entropy_coef = 0.01  # Entropy coefficient for exploration
    
    def select_action(self, state, priority):
        """Select action using priority-aware softmax with entropy bonus"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        logits, value = self.model(state_tensor)
        
        # Normalize priority to [0,1] range
        norm_priority = priority / 5.0  # Assuming priority is 1-5
        
        # Apply softmax to get action probabilities with priority bias
        probs = torch.softmax(logits + norm_priority, dim=-1)
        
        # Sample action from the probability distribution
        action = torch.multinomial(probs, 1).item()
        
        # Calculate log probability for the selected action
        log_prob = torch.log(probs[0][action])
        
        return action, log_prob, value
    
    def update(self, states, actions, rewards, next_states, dones):
        """Train the model using advantage actor-critic update with entropy bonus"""
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get current value estimates
        _, current_values = self.model(states)
        
        # Get next state value estimates
        with torch.no_grad():
            _, next_values = self.model(next_states)
            next_values = next_values.squeeze()
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - current_values.squeeze()
        
        # Get action probabilities and values for current states
        logits, values = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate policy loss with entropy bonus
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        
        # Entropy bonus for exploration
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Calculate losses
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        
        # Total loss with entropy bonus
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

class StandardA3C:
    """Standard A3C Implementation with improved architecture"""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99  # Discount factor
        self.entropy_coef = 0.01  # Entropy coefficient for exploration
    
    def select_action(self, state):
        """Select action using standard softmax with entropy bonus"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.model(state_tensor)
        
        # Apply softmax to get action probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample action from the probability distribution
        action = torch.multinomial(probs, 1).item()
        
        # Calculate log probability for the selected action
        log_prob = torch.log(probs[0][action])
        
        return action, log_prob, value
    
    def update(self, states, actions, rewards, next_states, dones):
        """Train the model using advantage actor-critic update with entropy bonus"""
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get current value estimates
        _, current_values = self.model(states)
        
        # Get next state value estimates
        with torch.no_grad():
            _, next_values = self.model(next_states)
            next_values = next_values.squeeze()
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - current_values.squeeze()
        
        # Get action probabilities and values for current states
        logits, values = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate policy loss with entropy bonus
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        
        # Entropy bonus for exploration
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Calculate losses
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        
        # Total loss with entropy bonus
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

# Load real-world job scheduling data
job_data = load_google_cloud_data()

env = gym.make("CartPole-v1")  # Placeholder for a job scheduling environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize both A3C implementations
weighted_a3c = WeightedA3C(state_dim, action_dim)
standard_a3c = StandardA3C(state_dim, action_dim)

# Training loop for both implementations
weighted_rewards = []
standard_rewards = []
weighted_aggregate = []
standard_aggregate = []
weighted_metrics = []
standard_metrics = []

# Window size for moving average
window_size = 10

for episode in range(100):
    # Weighted A3C
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward_weighted = 0
    episode_metrics_weighted = []
    
    while not done:
        job = job_data.sample(1).iloc[0]
        priority = torch.tensor([job['priority']], dtype=torch.float32)
        action, log_prob, value = weighted_a3c.select_action(state, priority)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        fairness_penalty = np.var(priority.numpy())
        weighted_r = weighted_reward(priority.item(), reward, fairness_penalty)
        
        # Update model and collect metrics
        metrics = weighted_a3c.update([state], [action], [weighted_r], [next_state], [done])
        episode_metrics_weighted.append(metrics)
        
        state = next_state
        total_reward_weighted += weighted_r
    
    weighted_rewards.append(total_reward_weighted)
    weighted_metrics.append({
        'policy_loss': np.mean([m['policy_loss'] for m in episode_metrics_weighted]),
        'value_loss': np.mean([m['value_loss'] for m in episode_metrics_weighted]),
        'entropy': np.mean([m['entropy'] for m in episode_metrics_weighted]),
        'total_loss': np.mean([m['total_loss'] for m in episode_metrics_weighted])
    })
    
    # Standard A3C
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward_standard = 0
    episode_metrics = []
    
    while not done:
        action, log_prob, value = standard_a3c.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        
        # Use standard reward with normalized weights
        standard_r = standard_reward(reward)
        
        # Update model and collect metrics
        metrics = standard_a3c.update([state], [action], [standard_r], [next_state], [done])
        episode_metrics.append(metrics)
        
        state = next_state
        total_reward_standard += standard_r
    
    standard_rewards.append(total_reward_standard)
    standard_metrics.append({
        'policy_loss': np.mean([m['policy_loss'] for m in episode_metrics]),
        'value_loss': np.mean([m['value_loss'] for m in episode_metrics]),
        'entropy': np.mean([m['entropy'] for m in episode_metrics]),
        'total_loss': np.mean([m['total_loss'] for m in episode_metrics])
    })
    
    # Calculate aggregate rewards
    if episode >= window_size - 1:
        weighted_avg = np.mean(weighted_rewards[episode-window_size+1:episode+1])
        standard_avg = np.mean(standard_rewards[episode-window_size+1:episode+1])
        weighted_aggregate.append(weighted_avg)
        standard_aggregate.append(standard_avg)

# Plot comparison
plt.figure(figsize=(15, 8))

# Plot raw rewards
plt.subplot(2, 1, 1)
plt.plot(range(1, 101), standard_rewards, label="Standard A3C", color='r', alpha=0.3)
plt.plot(range(1, 101), weighted_rewards, label="Weighted A3C", color='b', alpha=0.3)
plt.xlabel("Episodes")
plt.ylabel("Raw Reward")
plt.title("Raw Rewards Comparison")
plt.legend()
plt.grid(True)

# Plot moving average rewards
plt.subplot(2, 1, 2)
plt.plot(range(window_size, 101), standard_aggregate, label="Standard A3C (MA)", color='r', linewidth=2)
plt.plot(range(window_size, 101), weighted_aggregate, label="Weighted A3C (MA)", color='b', linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Moving Average Reward (window=10)")
plt.title("Moving Average Rewards Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final performance metrics
print("\nFinal Performance Metrics:")
print("-" * 50)
print("Raw Rewards:")
print(f"Standard A3C - Final Episode Reward: {standard_rewards[-1]:.2f}")
print(f"Weighted A3C - Final Episode Reward: {weighted_rewards[-1]:.2f}")

print("\nMoving Average Rewards (window=10):")
print(f"Standard A3C - Final Moving Average: {standard_aggregate[-1]:.2f}")
print(f"Weighted A3C - Final Moving Average: {weighted_aggregate[-1]:.2f}")

print("\nAggregate Statistics:")
print(f"Standard A3C - Mean Reward: {np.mean(standard_rewards):.2f}")
print(f"Weighted A3C - Mean Reward: {np.mean(weighted_rewards):.2f}")
print(f"Standard A3C - Std Dev: {np.std(standard_rewards):.2f}")
print(f"Weighted A3C - Std Dev: {np.std(weighted_rewards):.2f}")
print(f"Standard A3C - Min Reward: {np.min(standard_rewards):.2f}")
print(f"Weighted A3C - Min Reward: {np.min(weighted_rewards):.2f}")
print(f"Standard A3C - Max Reward: {np.max(standard_rewards):.2f}")
print(f"Weighted A3C - Max Reward: {np.max(weighted_rewards):.2f}")

print("\nStandard A3C Training Metrics (Last Episode):")
print(f"Policy Loss: {standard_metrics[-1]['policy_loss']:.4f}")
print(f"Value Loss: {standard_metrics[-1]['value_loss']:.4f}")
print(f"Entropy: {standard_metrics[-1]['entropy']:.4f}")
print(f"Total Loss: {standard_metrics[-1]['total_loss']:.4f}")

print("\nWeighted A3C Training Metrics (Last Episode):")
print(f"Policy Loss: {weighted_metrics[-1]['policy_loss']:.4f}")
print(f"Value Loss: {weighted_metrics[-1]['value_loss']:.4f}")
print(f"Entropy: {weighted_metrics[-1]['entropy']:.4f}")
print(f"Total Loss: {weighted_metrics[-1]['total_loss']:.4f}")

print("\nImprovement Metrics:")
print(f"Raw Reward Improvement: {((weighted_rewards[-1] - standard_rewards[-1]) / standard_rewards[-1] * 100):.2f}%")
print(f"Moving Average Improvement: {((weighted_aggregate[-1] - standard_aggregate[-1]) / standard_aggregate[-1] * 100):.2f}%")
print(f"Mean Reward Improvement: {((np.mean(weighted_rewards) - np.mean(standard_rewards)) / np.mean(standard_rewards) * 100):.2f}%")
