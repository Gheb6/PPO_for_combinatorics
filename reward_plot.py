import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
 
 
filename = "results/output.csv"  # Replace with the name of your file

# Read the CSV file
data = pd.read_csv(filename)
data.columns = data.columns.str.strip()
# Extract the eighth column (mean_best_reward)
mean_best_reward = data["mean_best_reward"]
 
# Generate the x-axis (row index)
x = range(len(mean_best_reward))
 
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, mean_best_reward, linestyle="solid", color="b", label="Mean Best Reward")
 
# Customize the plot
plt.title("Trend of Mean Best Reward")
plt.xlabel("Iterations")
plt.ylabel("Mean Best Reward")
plt.legend()
plt.grid(True)

# Show the plot
#plt.show()
plt.savefig('output.pdf')
