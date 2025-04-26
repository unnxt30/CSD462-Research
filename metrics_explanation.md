# Performance Metrics Explanation

This document provides a detailed explanation of the metrics used in the A3C (Asynchronous Advantage Actor-Critic) model evaluations.

## Core Metrics Overview

| Metric | Description | Optimization Goal |
|--------|-------------|------------------|
| Reward | Combined performance indicator | Maximize |
| Fairness | Resource distribution equality | Maximize |
| Priority Fulfillment | Job priority satisfaction | Maximize |
| Energy Consumption | System power utilization | Minimize |
| Latency | Response time performance | Minimize |

## Detailed Metrics Explanation

### 1. Reward

**Definition**: A composite metric that combines QoS (Quality of Service), energy efficiency, priority handling, and fairness into a single value.

**Calculation**:
- **Standard A3C**:
  ```python
  reward = (w_qos * norm_latency) + (w_energy * norm_energy)
  ```
  Where `w_qos` = 0.5 and `w_energy` = 0.5

- **Weighted A3C** (Enhanced):
  ```python
  # Get adaptive weights based on system state
  w_qos, w_energy, w_priority, w_fairness = adaptive_reward_weights(state, history, cpu_utilization)
  
  # Component rewards
  qos_reward = w_qos * norm_latency
  energy_reward = w_energy * norm_energy
  priority_reward = w_priority * norm_priority
  fairness_penalty = w_fairness * (1 - norm_fairness)
  
  # Non-linear combination
  reward = (qos_reward ** 2 + energy_reward ** 2 + priority_reward ** 2) ** 0.5 - fairness_penalty
  ```

**Significance**: Higher values indicate better overall system performance, balancing multiple competing objectives.

### 2. Fairness

**Definition**: A measure of equitable resource distribution across jobs with different priorities.

**Calculation**:
```python
def calculate_fairness_penalty(priorities, allocations, time_since_last_allocation):
    # Priority-based fairness
    priority_fairness = 1 - np.var(priorities) / np.mean(priorities)
    
    # Time-based fairness (prevent starvation)
    time_fairness = np.exp(-np.mean(time_since_last_allocation) / 100)
    
    # Resource allocation fairness
    allocation_fairness = 1 - np.var(allocations) / np.mean(allocations)
    
    # Combined fairness score
    fairness_score = (0.4 * priority_fairness + 
                     0.3 * time_fairness + 
                     0.3 * allocation_fairness)
    
    return max(0, 1 - fairness_score)  # Convert to penalty
```

**Components**:
1. **Priority-based Fairness**: Measures variance in priority handling
2. **Time-based Fairness**: Prevents job starvation
3. **Resource Allocation Fairness**: Ensures equitable distribution of resources

**Significance**: Lower values indicate more equitable resource distribution.

### 3. Priority Fulfillment

**Definition**: The degree to which the system satisfies job priority requirements.

**Calculation**: 
```python
# Normalized priority fulfillment
norm_priority = priority / 5.0  # Assuming priority ranges from 0-5
```

**Significance**: Higher values indicate better handling of high-priority jobs.

### 4. Energy Consumption

**Definition**: The amount of power consumed by the system during job processing.

**Calculation**:
```python
def energy_consumption(cpu_utilization, duration, cycles_per_instruction, memory_accesses_per_instruction):
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
```

**Components**:
1. **Base Power**: Constant power draw (idle power)
2. **CPU Utilization**: Percentage of CPU resources used
3. **Performance Factor**: Impact of CPU cycles and memory accesses
4. **Duration**: Execution time of the job

**Significance**: Lower values indicate more energy-efficient operation.

### 5. Latency

**Definition**: The response time for job execution, normalized to a 0-1 scale.

**Calculation**:
```python
# Normalize latency where lower is better
norm_latency = 1 - min(latency, 1.0)
```

**Significance**: Higher normalized values (lower raw latency) indicate better QoS.

## Visualization Metrics

### 1. Raw Rewards

**Definition**: The actual reward values obtained in each episode.

**Significance**: Shows the unfiltered performance of the policy, including natural variance.

### 2. Moving Average Rewards

**Definition**: A smoothed version of rewards using a rolling window (default: 10 episodes).

**Calculation**:
```python
weighted_avg = np.mean(weighted_rewards[episode-window_size+1:episode+1])
standard_avg = np.mean(standard_rewards[episode-window_size+1:episode+1])
```

**Significance**: Reveals trend patterns by reducing noise in the raw data.

### 3. Improvement Metrics

**Definition**: Percentage improvement of Weighted A3C over Standard A3C.

**Calculations**:
```python
# Raw reward improvement
((weighted_rewards[-1] - standard_rewards[-1]) / standard_rewards[-1] * 100)

# Moving average improvement
((weighted_aggregate[-1] - standard_aggregate[-1]) / standard_aggregate[-1] * 100)

# Mean reward improvement
((np.mean(weighted_rewards) - np.mean(standard_rewards)) / np.mean(standard_rewards) * 100)
```

**Significance**: Quantifies the relative performance gain of the enhanced approach.

## Adaptive Weighting System

The Weighted A3C implements an adaptive weighting system that dynamically adjusts component weights based on:

### 1. System Load

```python
# Adjust based on system load
if cpu_utilization > 0.8:  # High load
    w_qos *= 1.2
    w_energy *= 0.8
elif cpu_utilization < 0.2:  # Low load
    w_energy *= 1.2
    w_qos *= 0.8
```

### 2. Historical Performance

```python
# Adjust based on historical performance
if len(history) > 0:
    recent_fairness = np.mean([h['fairness'] for h in history[-5:]])
    if recent_fairness < 0.5:  # Poor fairness
        w_fairness *= 1.5
        w_priority *= 0.8
```

### 3. Weight Normalization

```python
# Normalize weights
total = w_qos + w_energy + w_priority + w_fairness
w_qos /= total
w_energy /= total
w_priority /= total
w_fairness /= total
```

This adaptive approach allows the system to respond dynamically to changing conditions, prioritizing different aspects of performance based on the current system state. 