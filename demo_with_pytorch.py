# Code to accompany the paper "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner
# Code for conjecture 2.1, without the use of numba
#
# Please keep in mind that I am far from being an expert in reinforcement learning.
# If you know what you are doing, you might be better off writing your own code.
#
# This code works on tensorflow version 1.14.0 and python version 3.6.3
# It mysteriously breaks on other versions of python.
# For later versions of tensorflow there seems to be a massive overhead in the predict function for some reason, and/or it produces mysterious errors.
# Debugging these was way above my skill level.
# If the code doesn't work, make sure you are using these versions of tf and python.
#
# I used keras version 2.3.1, not sure if this is important, but I recommend this just to be safe.




import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
from statistics import mean
import pickle
import time
import math
import matplotlib
matplotlib.use('Agg')  # Set Agg as backend
import matplotlib.pyplot as plt
import io
import os
import json
import sys
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models #used for summary


N = 19   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm
MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.0001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
                          #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.

observation_space = 2*MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                                                  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                                                  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                                                  #Is there a better way to format the input to make it easier for the neural network to understand things?



len_game = MYN
state_dim = (observation_space,)

INF = 1000000

# Global counters
number_of_elite_graphs = 0
number_of_super_graphs = 0
number_of_elite_actions = 0


# Global variables to terminate the program 1000 iterations after the counterexample
found_counterexample = False
iterations_after_counterexample = 0
MAX_ITERATIONS_AFTER_COUNTEREXAMPLE = 1000
max_score = -INF

##NumPy array that tracks statistics
statistics = np.empty((0, 15), dtype=float)
new_tuple = ()

# Variable to accumulate actions for each block
actions_block = []
iteration_block = 500  # Number of iterations to calculate the heatmap

CHANGE_ORDER = False

#Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
#I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
#It is important that the loss is binary cross-entropy if alphabet size is 2.

# Device configuration - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NeuralNet(nn.Module):
    """
    A simple feedforward neural network for binary classification.

    Architecture:
    - Input layer: observation_space neurons
    - Hidden layers: three fully connected layers with ReLU activation
    - Output layer: 1 neuron with Sigmoid activation (for binary classification)
    """

    def __init__(self, observation_space, first_neurons, second_neurons, third_neurons):
        """
        Initializes the neural network layers.

        Parameters:
        - observation_space (int): Number of input features.
        - first_neurons (int): Number of neurons in the first hidden layer.
        - second_neurons (int): Number of neurons in the second hidden layer.
        - third_neurons (int): Number of neurons in the third hidden layer.
        """
        super(NeuralNet, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(observation_space, first_neurons)  # First hidden layer
        self.fc2 = nn.Linear(first_neurons, second_neurons)     # Second hidden layer
        self.fc3 = nn.Linear(second_neurons, third_neurons)     # Third hidden layer
        self.fc4 = nn.Linear(third_neurons, 1)                  # Output layer

        # Activation functions
        self.relu = nn.ReLU()       # ReLU activation for hidden layers

        # Apply Xavier uniform initialization to weights
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (batch_size, observation_space).

        Returns:
        - torch.Tensor: Output tensor with shape (batch_size, 1), representing probabilities.
        """
        x = self.relu(self.fc1(x))   # Apply ReLU after first layer
        x = self.relu(self.fc2(x))   # Apply ReLU after second layer
        x = self.relu(self.fc3(x))   # Apply ReLU after third layer
        x = self.fc4(x)              # Output raw logits, no sigmoid
        return x

def predict(model, states, step):
    """
    Performs a forward pass on the model to get predictions.

    Parameters:
    - model (NeuralNet): The trained PyTorch model.
    - states (numpy array or torch.Tensor): Input data of shape (batch_size, observation_space, time_steps).
    - step (int): Time step to extract from states.

    Returns:
    - torch.Tensor: Model predictions.
    """
    # Convert states[:,:,step-1] to a PyTorch tensor and move to GPU
    input_tensor = torch.tensor(states[:, :, step-1], dtype=torch.float32).to(device)

    # Set model to evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid to convert logits to probabilities

    return prob  # Return predictions


def train_model(model, elite_states, elite_actions, epochs=1, batch_size=32, learning_rate=LEARNING_RATE):
    """
    Train the PyTorch model using elite graphs (elite_states) and their corresponding good actions (elite_actions).

    Parameters:
    - model: the PyTorch neural network
    - elite_states: tensor representing elite graphs
    - elite_actions: tensor of the elite actions taken by the agent
    - epochs: number of training iterations
    - batch_size: size of mini-batches
    - learning_rate: step size for optimizer updates
    """

    # Convert elite_states and elite_actions to PyTorch tensors and move to GPU
    elite_states = torch.tensor(elite_states, dtype=torch.float32).to(device)
    elite_actions = torch.tensor(elite_actions, dtype=torch.float32).to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss(reduction="sum")  # Combines sigmoid activation with binary cross-entropy loss
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.0, weight_decay=0, nesterov=False)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for better convergence
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # L2 regularization

    # Get dataset size
    dataset_size = len(elite_states)

    # Training loop
    model.train()
    for epoch in range(epochs):

        # Create a random permutation of indices (0 to dataset_size-1) and move it to the specified device
        # This tensor will be used to shuffle both states and actions in the same order
        indices = torch.randperm(dataset_size).to(device)

        # Use the random indices to shuffle the states tensor
        # For example, if indices=[3,0,4,1,2], shuffled_states will contain elite_states in that order
        shuffled_states = elite_states[indices]
        shuffled_actions = elite_actions[indices]

        total_loss = 0
        batches = 0

        for i in range(0, len(elite_states), batch_size):
            # Mini-batch
            batch_states = shuffled_states[i:i+batch_size]
            batch_actions = shuffled_actions[i:i+batch_size]

            # Forward pass
            predictions = model(batch_states).squeeze()  # Ensure proper shape

            # Compute loss

            # Add this before the loss calculation
            #predictions = predictions.view(-1)  # Reshape to match batch_actions shape
                # Or alternatively:
            # batch_actions = batch_actions.view(predictions.size())
            loss = criterion(predictions, batch_actions)
            total_loss += loss.item()
            batches += 1

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

    print("Training completed!")

# Model creation and move to GPU
model = NeuralNet(observation_space, FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS).to(device)

# Change this to check the initialized weights
if False:
    # Print weights of each layer
    print("L1 weights:", model.fc1.weight)
    print("L2 weights:", model.fc2.weight)
    print("L3 weights:", model.fc3.weight)
    print("L4 weights:", model.fc4.weight)

print(model)

def calcScore(state):
        """
        Calculates the reward for a given word.
        This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
        :param state: the first MYN letters of this param are the word that the neural network has constructed.


        :returns: the reward (a real number). Higher is better, the network will try to maximize this.
        """

        #Example reward function, for Conjecture 2.1
        #Given a graph, it minimizes lambda_1 + mu.
        #Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
        #There is a lot of run-to-run variance.
        #Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.
        
        global found_counterexample, iterations_after_counterexample, max_score, CHANGE_ORDER
        
        #Construct the graph
        G= nx.Graph()
        G.add_nodes_from(list(range(N)))
        count = 0
        
        if CHANGE_ORDER:
                for k in range(1, N):
                        for j in range(0, k):
                                if state[count] == 1:
                                        G.add_edge(j, k)
                                count += 1
        else:
                for k in range(N):
                        for j in range(k+1,N):
                                if state[count] == 1:
                                        G.add_edge(k,j)
                                count += 1

        #G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
        if not (nx.is_connected(G)):
                return -INF

        #Calculate the eigenvalues of G
        evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
                evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)

        #Calculate the matching number of G
        maxMatch = nx.max_weight_matching(G)
        mu = len(maxMatch)

        #Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
        #We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
        myScore = math.sqrt(N-1) + 1 - lambda1 - mu
        if myScore > max_score:
                max_score = myScore
        if myScore > 0:
                if not found_counterexample:
                        found_counterexample = True
                        print(state)
                        nx.draw_kamada_kawai(G)
                        #Save the graph
                        plt.savefig('output.pdf')
                
                        with open('positive score graph.txt', 'w') as f:
                                f.write("Graph state: \n")
                                f.write(str(state) + "\n")
                                f.write("Score: " + str(myScore) + "\n")
                                end_time = time.time()
                                execution_time = end_time - start_time
                        with open('Total_execution_time.txt', 'w') as f:
                                print("Total execution time: " + str(execution_time))
                                f.write("Total execution time: " + str(execution_time))
                return myScore

        return myScore



####No need to change anything below here.




def generate_session(agent, n_sessions, verbose = 1):
        """
        Play n_session games using agent neural network.
        Terminate when games finish

        Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
        """
        global new_tuple
        states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
        actions = np.zeros([n_sessions, len_game], dtype = int)
        state_next = np.zeros([n_sessions,observation_space], dtype = int)
        prob = np.zeros(n_sessions)
        states[:,MYN,0] = 1
        step = 0
        total_score = np.zeros([n_sessions])
        recordsess_time = 0
        play_time = 0
        scorecalc_time = 0
        pred_time = 0
        while (True):
                step += 1
                tic = time.time()
                prob = predict(model, states, step)
                pred_time += time.time()-tic

                for i in range(n_sessions):

                        if np.random.rand() < prob[i]:
                                action = 1
                        else:
                                action = 0
                        actions[i][step-1] = action
                        tic = time.time()
                        state_next[i] = states[i,:,step-1]
                        play_time += time.time()-tic
                        if (action > 0):
                                state_next[i][step-1] = action
                        state_next[i][MYN + step-1] = 0
                        if (step < MYN):
                                state_next[i][MYN + step] = 1
                        terminal = step == MYN
                        tic = time.time()
                        if terminal:
                                total_score[i] = calcScore(state_next[i])
                        scorecalc_time += time.time()-tic
                        tic = time.time()
                        if not terminal:
                                states[i,:,step] = state_next[i]
                        recordsess_time += time.time()-tic


                if terminal:
                        break
        #new_tuple = new_tuple + (pred_time, play_time, scorecalc_time, recordsess_time,)
        return states, actions, total_score



def     select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
        """
        Select states and actions from games that have rewards >= percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i]

        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

        This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
        If this function is the bottleneck, it can easily be sped up using numba
        """
        global number_of_elite_actions
        global number_of_elite_graphs
        number_of_elite_actions = 0
        number_of_elite_graphs = 0

        counter = n_sessions * (100.0 - percentile) / 100.0
        reward_threshold = np.percentile(rewards_batch,percentile)

        elite_states = []
        elite_actions = []
        elite_rewards = []
        for i in range(len(states_batch)):
                if rewards_batch[i] >= reward_threshold-0.0000001:
                        if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                                number_of_elite_graphs += 1
                                for item in states_batch[i]:
                                        elite_states.append(item.tolist())
                                for item in actions_batch[i]:
                                        number_of_elite_actions += 1
                                        elite_actions.append(item)
                        counter -= 1
        elite_states = np.array(elite_states, dtype = int)
        elite_actions = np.array(elite_actions, dtype = int)
        return elite_states, elite_actions

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
        """
        Select all the sessions that will survive to the next generation
        Similar to select_elites function
        If this function is the bottleneck, it can easily be sped up using numba
        """
        global number_of_super_graphs
        number_of_super_graphs = 0

        counter = n_sessions * (100.0 - percentile) / 100.0
        reward_threshold = np.percentile(rewards_batch,percentile)

        super_states = []
        super_actions = []
        super_rewards = []
        for i in range(len(states_batch)):
                if rewards_batch[i] >= reward_threshold-0.0000001:
                        if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                                number_of_super_graphs += 1
                                super_states.append(states_batch[i])
                                super_actions.append(actions_batch[i])
                                super_rewards.append(rewards_batch[i])
                                counter -= 1
        super_states = np.array(super_states, dtype = int)
        super_actions = np.array(super_actions, dtype = int)
        super_rewards = np.array(super_rewards)
        return super_states, super_actions, super_rewards


def print_elite_graph(iteration, elite_action, base_path):
    
    elite_graph_dir = os.path.join(base_path, "elite_graph")
    os.makedirs(elite_graph_dir, exist_ok=True)
    
    # Change folder every iteration (modifiable based on desired logic)
    folder_number = iteration // 1
    directory = os.path.join(elite_graph_dir, f"Elite_graphs_{folder_number}")

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    file_index = 1
    current_chars = 0
    buffer = ""

    for item in elite_action:
        item_str = str(item)  # We do not include the space here for character counting
        buffer += item_str + " "  # Add space only for writing purposes
        current_chars += len(item_str)  # Count only actual characters
        # If we have reached 171 characters, write to file and reset
        if current_chars >= 171:
            file_path = os.path.join(directory, f"file_{file_index}.txt")
            with open(file_path, "w") as f:
                f.write(buffer.strip())  # Remove extra spaces before writing
            file_index += 1
            current_chars = 0
            buffer = ""

    # Write any remaining characters
    if buffer.strip():  # Check if there are characters to write
        file_path = os.path.join(directory, f"file_{file_index}.txt")
        with open(file_path, "w") as f:
            f.write(buffer.strip())

super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0



myRand = random.randint(0,1000) #used in the filename


# Create output directory for results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for i in range(1000000): #1000000 generations should be plenty
        #generate new sessions
        #performance can be improved with joblib
        start_time = time.time()
        tic = time.time()
        sessions = generate_session(model, n_sessions, 0) #change 0 to 1 to print out how much time each step in generate_session takes
        sessgen_time = time.time()-tic
        tic = time.time()

        if found_counterexample:
                iterations_after_counterexample += 1
                if iterations_after_counterexample >= MAX_ITERATIONS_AFTER_COUNTEREXAMPLE:
                        print("Termination after 1000 additional iterations")
                        sys.exit()

        states_batch = np.array(sessions[0], dtype = int)
        actions_batch = np.array(sessions[1], dtype = int)
        rewards_batch = np.array(sessions[2])
        states_batch = np.transpose(states_batch,axes=[0,2,1])

        states_batch = np.append(states_batch,super_states,axis=0)

        if i>0:
                actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)
        rewards_batch = np.append(rewards_batch,super_rewards)

        randomcomp_time = time.time()-tic
        tic = time.time()

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
        select1_time = time.time()-tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
        select2_time = time.time()-tic

        tic = time.time()
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
        select3_time = time.time()-tic

        tic = time.time()
        # Train the model with elite states and actions
        train_model(model, elite_states, elite_actions)
        fit_time = time.time()-tic

        tic = time.time()

        super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
        super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
        super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]

        rewards_batch.sort()
        mean_all_reward = np.mean(rewards_batch[-100:])
        mean_best_reward = np.mean(super_rewards)

        score_time = time.time()-tic
  
        # Directory path to save the files - modify as needed for your system
        base_path = os.path.join(output_dir, "dictionaries")
        os.makedirs(base_path, exist_ok=True)

        occurrences = {}

        for action in super_actions:
                # Ensure that action is a full row of the array and convert it to a string
                key = " ".join(map(str, action))  # Convert each bit to a string and separate them with spaces
                if key in occurrences:
                        occurrences[key] += 1
                else:
                        occurrences[key] = 1
        

        # Generate a unique filename for each iteration
        file_name = f"dictionary_{i + 1}.json"
        file_path = os.path.join(base_path, file_name)

        # Write the dictionary in JSON format
        with open(file_path, "w") as file:
                json.dump(occurrences, file, indent=4)

        #TERMINATION CONDITION
        # Check if the dictionary has only one key 
        if len(occurrences) == 1 and not found_counterexample:
                print("The dictionary has a single key. Program terminated.")
                sys.exit()


        # Add the actions of the current block
        actions_block.extend(super_actions)
        # Calculate and save the heatmap every 500 iterations
        if (i + 1) % iteration_block == 0:
                # Convert the actions into a NumPy array
                actions_array = np.array(actions_block)

                mean_matrix = np.zeros((N, N))

                # Calculate the mean of each bit (axis 0)
                bit_means = actions_array.mean(axis=0)
                
                # Populate the matrix symmetrically
                bit_index = 0  # Track position in bit_means array
                
                if CHANGE_ORDER:
                        for k in range(1, N):
                                for j in range(0, k):
                                        mean_value = bit_means[bit_index]
                                        mean_matrix[j, k] = mean_value
                                        mean_matrix[k, j] = mean_value
                                        bit_index += 1
                                        
                else:
                        for k in range(N):
                                for j in range(k + 1, N):
                                        mean_value = bit_means[bit_index]
                                        mean_matrix[k, j] = mean_value
                                        mean_matrix[j, k] = mean_value
                                        bit_index += 1
                
                # Set the diagonal to 0 (should already be 0, but just to be explicit)
                np.fill_diagonal(mean_matrix, 0)
                # Create heatmap directory
                heatmap_dir = os.path.join(output_dir, "heatmap")
                os.makedirs(heatmap_dir, exist_ok=True)
                mean_matrix_file_path = os.path.join(heatmap_dir, f"mean_matrix_{i + 1}.csv")
                with open(mean_matrix_file_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([""] + [f"Node_{j}" for j in range(N)])  # Column header
                        for k in range(N):
                                writer.writerow([f"Node_{k}"] + mean_matrix[k].tolist())  # Row with label and values


                # Generate and save the heatmap
                plt.figure(figsize=(10, 2))
                plt.imshow(
                [bit_means], aspect="auto", cmap="viridis", interpolation="nearest"
                )
                plt.colorbar(label="Average Value")
                plt.title(f"Heatmap of Average Bits - Iterations {i + 1 - iteration_block + 1} to {i + 1}")
                plt.xlabel("Bit Index")
                plt.yticks([])  # Remove the Y-axis since it's a single row

                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{i + 1}.pdf")
                plt.savefig(heatmap_path)
                plt.close()

                print(f"Heatmap saved: {heatmap_path}")

                #Reset the data for the next block
                actions_block = []

        # Change this to print every elite graph:
        if(True):
                print_elite_graph(i, elite_actions, output_dir)
                
         # Clean up GPU memory (if necessary)
        if torch.cuda.is_available() and i % 10 == 0:
            torch.cuda.empty_cache()

        new_tuple = new_tuple + (sessgen_time, randomcomp_time, select1_time, select2_time, select3_time, fit_time, score_time,)
        number_of_parameters = sum(p.numel() for p in model.parameters())
        new_tuple = new_tuple + (mean_best_reward, number_of_elite_graphs, number_of_elite_actions, number_of_super_graphs, number_of_parameters, max_score, max(occurrences.values()), len(occurrences))
        statistics = np.append(statistics, [new_tuple], axis=0)
        header = "sessgen_time, randomcomp_time, select1_time, select2_time, select3_time, fit_time, score_time, mean_best_reward, number_of_elite_graphs, number_of_elite_actions, number_of_super_graphs, parameters, max_score, maximum frequency, number of different graphs"
        np.savetxt(os.path.join(output_dir, 'output.csv'), statistics, delimiter=',', fmt='%.2f', header=header, comments='')
        new_tuple = ()

        print(f"Iteration {i+1} completed. CSV file successfully written!")

print('maximum iterations exceeded\n')
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time: " + str(execution_time))

with open(os.path.join(output_dir, 'Total_execution_time.txt'), 'w') as f:
    f.write("Total execution time: " + str(execution_time))
