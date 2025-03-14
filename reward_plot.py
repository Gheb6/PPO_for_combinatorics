import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


# List of directories and corresponding file names
directories = ["test1/output.csv", "test2/output.csv", "test3/output.csv", "test4/output.csv"]
colors = ['b', 'g', 'r', 'c']  # Colors for each plot

plt.figure(figsize=(10, 6))

for idx, filename in enumerate(directories):
    # Read the CSV file
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()
    
    # Extract the eighth column (mean_best_reward)
    if "mean_best_reward" in data.columns:
        mean_best_reward = data["mean_best_reward"]
        x = range(len(mean_best_reward))
        
        # Plot the data
        plt.plot(x, mean_best_reward, linestyle="solid", color=colors[idx], label=f"{filename}")
    else:
        print(f"Warning: {filename} does not contain 'mean_best_reward' column, skipping.")

# Customize the plot
plt.title("Comparison of Mean Best Reward")
plt.xlabel("Iterations")
plt.ylabel("Mean Best Reward")
plt.legend()
plt.axhline(y=0, color='k', linestyle='--', linewidth=1, label='y=0')  # Add horizontal line at y=0
plt.grid(True)

# Save the plot to a file
#plt.show()
plt.savefig('output_comparison.pdf')
