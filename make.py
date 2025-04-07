import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic job scheduling data
num_jobs = 1000
job_data = {
    "job_id": np.arange(1001, 1001 + num_jobs),  # Unique Job IDs
    "priority": np.random.randint(1, 6, num_jobs),  # Priority levels (1 to 5)
    "execution_time": np.round(np.random.uniform(0.5, 5.0, num_jobs), 2),  # Execution time (0.5s to 5s)
    "resource_usage": np.round(np.random.uniform(0.5, 2.0, num_jobs), 2),  # Resource usage (0.5GB to 2GB)
}

# Convert to DataFrame
df = pd.DataFrame(job_data)

# Save to CSV file
df.to_csv("google_jobs_large.csv", index=False)

print("Dataset google_jobs_large.csv created successfully!")
