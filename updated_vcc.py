import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import ast
from collections import deque
import seaborn as sns
from scipy import stats

def load_sample_data(file_path="sample.csv"):
    """Load job scheduling data from sample.csv dataset"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Extract relevant columns for job scheduling
        job_data = pd.DataFrame({
            'job_id': df.index,
            'priority': df['priority'],
            'duration': (df['end_time'] - df['start_time']) / 1e9,  # Convert to seconds
            'cpu_utilization': df['average_usage'].apply(
                lambda x: ast.literal_eval(x)['cpus'] if isinstance(x, str) else 0.0
            ) / df['resource_request'].apply(
                lambda x: ast.literal_eval(x)['cpus'] if isinstance(x, str) else 1.0
            ),
            'cycles_per_instruction': df['cycles_per_instruction'],
            'memory_accesses_per_instruction': df['memory_accesses_per_instruction']
        })
        return job_data
    except Exception as e:
        print(f"Error loading sample.csv: {e}")
        return None

def energy_consumption(cpu_utilization, duration, cycles_per_instruction, memory_accesses_per_instruction):
    """
    Calculate energy consumption based on the new formulation:
    Energy_Consumption = (Base_Power + (CPU_Utilization * Performance_Factor * (Max_Power - Base_Power))) * Duration
    """
    # Constants
    BASE_POWER = 100  # Watts
    MAX_POWER = 200   # Watts
    MEMORY_WEIGHT = 0.3  # Weight factor for memory impact
    
    # Calculate Performance Factor
    performance_factor = (cycles_per_instruction * MEMORY_WEIGHT + 
                        memory_accesses_per_instruction * (1 - MEMORY_WEIGHT))
    
    # Calculate Energy Consumption
    power = BASE_POWER + (cpu_utilization * performance_factor * (MAX_POWER - BASE_POWER))
    energy = power * duration
    
    return energy

def calculate_fairness_penalty(priorities, allocations, time_since_last_allocation):
    """
    Enhanced fairness calculation considering:
    1. Priority-based fairness
    2. Time-based fairness (starvation prevention)
    3. Resource allocation fairness
    """
    # Priority-based fairness
    priority_fairness = 1 - np.var(priorities) / np.mean(priorities)
    
    # Time-based fairness (prevent starvation)
    time_fairness = np.exp(-np.mean(time_since_last_allocation) / 100)  # Decay factor
    
    # Resource allocation fairness
    allocation_fairness = 1 - np.var(allocations) / np.mean(allocations)
    
    # Combined fairness score
    fairness_score = (0.4 * priority_fairness + 
                     0.3 * time_fairness + 
                     0.3 * allocation_fairness)
    
    return max(0, 1 - fairness_score)  # Convert to penalty

def adaptive_reward_weights(state, history, cpu_utilization):
    """
    Dynamically adjust reward component weights based on:
    1. Current system state
    2. Historical performance
    3. Resource utilization
    """
    # Base weights
    w_qos = 0.3
    w_energy = 0.3
    w_priority = 0.3
    w_fairness = 0.1
    
    # Adjust based on system load
    if cpu_utilization > 0.8:  # High load
        w_qos *= 1.2
        w_energy *= 0.8
    elif cpu_utilization < 0.2:  # Low load
        w_energy *= 1.2
        w_qos *= 0.8
    
    # Adjust based on historical performance
    if len(history) > 0:
        recent_fairness = np.mean([h['fairness'] for h in history[-5:]])
        if recent_fairness < 0.5:  # Poor fairness
            w_fairness *= 1.5
            w_priority *= 0.8
    
    # Normalize weights
    total = w_qos + w_energy + w_priority + w_fairness
    w_qos /= total
    w_energy /= total
    w_priority /= total
    w_fairness /= total
    
    return w_qos, w_energy, w_priority, w_fairness

def weighted_reward(priority, latency, fairness, cpu_utilization, duration, 
                   cycles_per_instruction, memory_accesses_per_instruction,
                   state, history):
    """Enhanced reward function with adaptive weights"""
    # Get adaptive weights
    w_qos, w_energy, w_priority, w_fairness = adaptive_reward_weights(state, history, cpu_utilization)
    
    # Calculate energy consumption
    energy = energy_consumption(
        cpu_utilization,
        duration,
        cycles_per_instruction,
        memory_accesses_per_instruction
    )
    
    # Normalize inputs
    norm_priority = priority / 5.0
    norm_latency = 1 - min(latency, 1.0)
    norm_fairness = min(fairness, 1.0)
    
    # Calculate normalized energy
    max_theoretical_energy = energy_consumption(
        1.0, duration, 
        max(cycles_per_instruction, 1), 
        max(memory_accesses_per_instruction, 1)
    )
    norm_energy = 1 - (energy / max_theoretical_energy)
    
    # Calculate reward components
    qos_reward = w_qos * norm_latency
    energy_reward = w_energy * norm_energy
    priority_reward = w_priority * norm_priority
    fairness_penalty = w_fairness * (1 - norm_fairness)
    
    # Combine rewards with non-linear transformation
    reward = (qos_reward ** 2 + energy_reward ** 2 + priority_reward ** 2) ** 0.5 - fairness_penalty
    
    return max(0, reward)

def standard_reward(latency, cpu_utilization):
    """Compute reward with normalized QoS and energy weights that sum to 1"""
    # Normalize weights to sum to 1
    w_qos = 0.5    # QoS weight (50%)
    w_energy = 0.5 # Energy weight (50%)
    
    # Verify weights sum to 1
    total_weight = w_qos + w_energy
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1, got {total_weight}"
    
    # Normalize latency to [0,1] range where 1 is best
    norm_latency = 1 - min(latency, 1.0)
    
    # Calculate energy efficiency based on CPU utilization
    # Use a default duration and instruction parameters
    duration = 1.0  # Default duration of 1 second
    cycles_per_instruction = 1.0  # Default cycles
    memory_accesses_per_instruction = 0.01  # Default memory accesses
    
    energy_cost = energy_consumption(
        cpu_utilization, 
        duration,
        cycles_per_instruction,
        memory_accesses_per_instruction
    )
    
    # Normalize energy cost (higher value means less energy used)
    max_energy = energy_consumption(
        1.0,  # Maximum CPU utilization
        duration,
        cycles_per_instruction,
        memory_accesses_per_instruction
    )
    
    norm_energy = 1 - (energy_cost / max_energy)
    
    # Calculate reward ensuring non-negative values
    reward = (w_qos * norm_latency) + (w_energy * norm_energy)
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
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.history = []
        self.time_since_allocation = {}
    
    def select_action(self, state, priority):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.model(state_tensor)
        
        # Enhanced priority weighting
        norm_priority = priority / 5.0
        time_factor = np.exp(-self.time_since_allocation.get(priority.item(), 0) / 100)
        priority_bias = norm_priority * time_factor
        
        # Apply softmax with enhanced priority bias
        probs = torch.softmax(logits + priority_bias, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0][action])
        
        # Update time since allocation
        self.time_since_allocation[priority.item()] = 0
        for p in self.time_since_allocation:
            self.time_since_allocation[p] += 1
        
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

class PerformanceMetrics:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.energy_consumptions = deque(maxlen=window_size)
        self.fairness_scores = deque(maxlen=window_size)
        self.priority_fulfillment = deque(maxlen=window_size)
        
    def update(self, reward, latency, energy, fairness, priority_fulfillment):
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.energy_consumptions.append(energy)
        self.fairness_scores.append(fairness)
        self.priority_fulfillment.append(priority_fulfillment)
        
    def get_metrics(self):
        return {
            'mean_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'mean_latency': np.mean(self.latencies),
            'mean_energy': np.mean(self.energy_consumptions),
            'mean_fairness': np.mean(self.fairness_scores),
            'mean_priority_fulfillment': np.mean(self.priority_fulfillment),
            'reward_95_ci': stats.t.interval(0.95, len(self.rewards)-1, 
                                          loc=np.mean(self.rewards), 
                                          scale=stats.sem(self.rewards))
        }

def plot_comparison(standard_metrics, weighted_metrics, save_path=None):
    """Enhanced visualization of performance comparison"""
    plt.figure(figsize=(20, 15))
    
    # 1. Raw Rewards Comparison
    plt.subplot(3, 2, 1)
    plt.plot(standard_metrics.rewards, label="Standard A3C", alpha=0.6)
    plt.plot(weighted_metrics.rewards, label="Weighted A3C", alpha=0.6)
    plt.title("Raw Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # 2. Fairness Comparison
    plt.subplot(3, 2, 2)
    plt.plot(standard_metrics.fairness_scores, label="Standard A3C", alpha=0.6)
    plt.plot(weighted_metrics.fairness_scores, label="Weighted A3C", alpha=0.6)
    plt.title("Fairness Scores Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Fairness Score")
    plt.legend()
    plt.grid(True)
    
    # 3. Priority Fulfillment
    plt.subplot(3, 2, 3)
    plt.plot(standard_metrics.priority_fulfillment, label="Standard A3C", alpha=0.6)
    plt.plot(weighted_metrics.priority_fulfillment, label="Weighted A3C", alpha=0.6)
    plt.title("Priority Fulfillment Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Priority Fulfillment")
    plt.legend()
    plt.grid(True)
    
    # 4. Energy Consumption
    plt.subplot(3, 2, 4)
    plt.plot(standard_metrics.energy_consumptions, label="Standard A3C", alpha=0.6)
    plt.plot(weighted_metrics.energy_consumptions, label="Weighted A3C", alpha=0.6)
    plt.title("Energy Consumption Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    
    # 5. Latency Comparison
    plt.subplot(3, 2, 5)
    plt.plot(standard_metrics.latencies, label="Standard A3C", alpha=0.6)
    plt.plot(weighted_metrics.latencies, label="Weighted A3C", alpha=0.6)
    plt.title("Latency Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Latency")
    plt.legend()
    plt.grid(True)
    
    # 6. Statistical Comparison
    plt.subplot(3, 2, 6)
    metrics = ['Reward', 'Fairness', 'Priority', 'Energy', 'Latency']
    standard_values = [
        np.mean(standard_metrics.rewards),
        np.mean(standard_metrics.fairness_scores),
        np.mean(standard_metrics.priority_fulfillment),
        np.mean(standard_metrics.energy_consumptions),
        np.mean(standard_metrics.latencies)
    ]
    weighted_values = [
        np.mean(weighted_metrics.rewards),
        np.mean(weighted_metrics.fairness_scores),
        np.mean(weighted_metrics.priority_fulfillment),
        np.mean(weighted_metrics.energy_consumptions),
        np.mean(weighted_metrics.latencies)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, standard_values, width, label='Standard A3C')
    plt.bar(x + width/2, weighted_values, width, label='Weighted A3C')
    plt.title("Mean Performance Comparison")
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Load sample.csv job scheduling data
job_data = load_sample_data("sample.csv")

env = gym.make("CartPole-v1", render_mode=None)  # Updated environment creation
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
weighted_metrics = PerformanceMetrics()
standard_metrics = PerformanceMetrics()

# Window size for moving average
window_size = 10

for episode in range(100):
    # Weighted A3C
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward_weighted = 0
    episode_metrics_weighted = []
    # print(job_data)
    while not done:
        job = job_data.sample(1).iloc[0]
        priority = torch.tensor([job['priority']], dtype=torch.float32)
        cpu_util = job['cpu_utilization']
        cycles = job['cycles_per_instruction']
        memory_accesses = job['memory_accesses_per_instruction']
        duration = job['duration']        
        action, log_prob, value = weighted_a3c.select_action(state, priority)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        fairness_penalty = calculate_fairness_penalty([job['priority']], [cpu_util], [0])
        
        # Use the energy consumption model in the reward calculation
        weighted_r = weighted_reward(
            priority.item(),
            reward,
            fairness_penalty,
            cpu_util,
            duration,
            cycles,
            memory_accesses,
            state,
            weighted_a3c.history
        )
        
        # Update model and collect metrics
        metrics = weighted_a3c.update([state], [action], [weighted_r], [next_state], [done])
        episode_metrics_weighted.append(metrics)
        
        state = next_state
        total_reward_weighted += weighted_r
    
    weighted_rewards.append(total_reward_weighted)
    weighted_metrics.update(weighted_r, reward, energy_consumption(cpu_util, duration, cycles, memory_accesses), fairness_penalty, priority.item())
    
    # Standard A3C
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward_standard = 0
    episode_metrics = []
    
    while not done:
        # Sample a job to get CPU utilization
        job = job_data.sample(1).iloc[0]
        cpu_util = job['cpu_utilization']  # CPU utilization from the job data
        
        action, log_prob, value = standard_a3c.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        
        # Use standard reward with normalized weights and energy consumption
        standard_r = standard_reward(reward, cpu_util)
        
        # Update model and collect metrics
        metrics = standard_a3c.update([state], [action], [standard_r], [next_state], [done])
        episode_metrics.append(metrics)
        
        state = next_state
        total_reward_standard += standard_r
    
    standard_rewards.append(total_reward_standard)
    standard_metrics.update(standard_r, reward, energy_consumption(cpu_util, duration, cycles, memory_accesses), calculate_fairness_penalty([job['priority']], [cpu_util], [0]), job['priority'])
    
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

print("\nImprovement Metrics:")
print(f"Raw Reward Improvement: {((weighted_rewards[-1] - standard_rewards[-1]) / standard_rewards[-1] * 100):.2f}%")
print(f"Moving Average Improvement: {((weighted_aggregate[-1] - standard_aggregate[-1]) / standard_aggregate[-1] * 100):.2f}%")
print(f"Mean Reward Improvement: {((np.mean(weighted_rewards) - np.mean(standard_rewards)) / np.mean(standard_rewards) * 100):.2f}%")

# Plot comparison
plot_comparison(standard_metrics, weighted_metrics)


