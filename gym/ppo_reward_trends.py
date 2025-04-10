import json
import matplotlib.pyplot as plt
import numpy as np

# Open JSON file and load data
with open('test8/ppo_graph_gym_results/rewards_history.json', 'r') as file:
    data = json.load(file)

# Extract episode data
episode_rewards = data['episode_rewards'] 

# Calculate mean for each episode
episode_means = [np.mean(episode) for episode in episode_rewards]

# Create episode array (x-axis)
episodes = np.arange(1, len(episode_means) + 1)

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(episodes, episode_means, marker='o', linestyle='-', color='b')
plt.title('Average Score Trend per Episode')
plt.xlabel('Episode Number')
plt.ylabel('Average Score')
plt.grid(True)

# Adjust y-axis limits for better visualization
plt.ylim(min(episode_means) - 10, max(episode_means) + 10)

plt.savefig('score_trend8.pdf')

# Display plot
plt.show()
