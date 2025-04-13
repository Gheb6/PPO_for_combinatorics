import json
import matplotlib.pyplot as plt
import numpy as np

# Open the JSON file and load the data
with open('test1/ppo_graph_gym_results/rewards_history.json', 'r') as file:
    data = json.load(file)

# Extract the best rewards data
best_rewards = data['best_rewards']

# Create the x-axis (iterations)
iterations = np.arange(1, len(best_rewards) + 1)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(iterations, best_rewards, marker='o', linestyle='-', color='b')
plt.title('Best Rewards Over Time')
plt.xlabel('Iteration')
plt.ylabel('Best Reward')
plt.grid(True)

# Adjust y-axis limits for better visualization
plt.ylim(min(best_rewards) - 1, max(best_rewards) + 1)

# Save the figure
plt.savefig('best_rewards_plot.pdf')

# Show the plot
plt.show()