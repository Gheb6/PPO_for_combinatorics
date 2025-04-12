import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import math

class GraphGymEnv(gym.Env):
    """
    A minimal Gymnasium environment for graph generation problems based on the paper
    "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner.
    
    This environment allows an agent to generate a graph by deciding whether to add each
    potential edge one by one. The goal is to minimize lambda_1 + mu (the largest eigenvalue
    plus the matching number) relative to sqrt(N-1) + 1.
    """

    def __init__(self, n_vertices=19, penalize_components=10, penalize_disconnected=-10):
        super(GraphGymEnv, self).__init__()
        
        # Number of vertices in the graph
        self.N = n_vertices
        
        # Number of potential edges (N choose 2)
        self.num_edges = int(self.N * (self.N - 1) / 2)
        
        # Current step in the episode
        self.current_step = 0
        
        # The constructed graph
        self.G = None
        
        # The current state representation
        self.state = None
        
        # Action space: binary choice for each edge (add it or not)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: partial graph + position indicator
        # First part represents the partial graph (which edges have been decided)
        # Second part is one-hot encoding of the current position
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 * self.num_edges,), 
            dtype=np.int32
        )

        # For disconnected graphs, reward is a linear function of the
        # number of components: these are the coefficients
        self.penalize_components = penalize_components
        self.penalize_disconnected = penalize_disconnected

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        # Set seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Initialize an empty graph
        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(self.N)))
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize state: all zeros except for the indicator of the first position
        self.state = np.zeros(2 * self.num_edges, dtype=np.int32)
        self.state[self.num_edges] = 1  # Mark first position
        
        return self.state, {}  # Return initial state and empty info dict per Gymnasium API

    def step(self, action):
        """
        Take a step in the environment by deciding whether to add an edge.
        
        Args:
            action: 0 (don't add edge) or 1 (add edge)
            
        Returns:
            observation: The new state
            reward: The reward for the action
            terminated: Whether the episode is complete
            truncated: Whether the episode was truncated (unused here)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Update the state with the action
        self.state[self.current_step] = action
        
        # Add the edge to the graph if action is 1
        if action == 1:
            # Convert current_step to graph edge indices
            i, j = self._step_to_edge(self.current_step)
            self.G.add_edge(i, j)
        
        # Move to next position
        self.current_step += 1
        
        # Update the one-hot encoding of current position
        self.state[self.num_edges + self.current_step - 1] = 0
        if self.current_step < self.num_edges:
            self.state[self.num_edges + self.current_step] = 1
        
        # Check if episode is done
        terminated = self.current_step == self.num_edges
        truncated = False  # We don't truncate episodes
        
        # Calculate reward
        reward = 0
        if terminated:
            reward = self.calcScore()
        
        # Prepare info dict
        info = {}
        if terminated:
            # Add additional information when episode is complete
            evals = np.linalg.eigvalsh(nx.adjacency_matrix(self.G).todense())
            lambda1 = max(abs(evals))
            max_match = nx.max_weight_matching(self.G)
            mu = len(max_match)
            info = {
                'lambda1': lambda1,
                'mu': mu,
                'is_connected': nx.is_connected(self.G),
                'score': reward
            }
        
        return self.state, reward, terminated, truncated, info

    def calcScore(self):
        """Calculate the score for the complete graph."""
        # Check if graph is connected (required by the conjecture)
        if not nx.is_connected(self.G):
            # Instead of a huge negative score, calculate how close we are to connectivity
            components = list(nx.connected_components(self.G))
            # Penalize based on number of components (fewer is better)
            return -(self.penalize_components * len(components) + self.penalize_disconnected)
        
        # Calculate eigenvalues
        evals = np.linalg.eigvalsh(nx.adjacency_matrix(self.G).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
                evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)
        
        # Calculate matching number
        max_Match = nx.max_weight_matching(self.G)
        mu = len(max_Match)
        
        # Calculate score: we want to maximize score = sqrt(N-1) + 1 - lambda1 - mu
        # Positive score means we've found a counterexample to the conjecture
        score = math.sqrt(self.N - 1) + 1 - lambda1 - mu
        
        return score

    def _step_to_edge(self, step):
        """Convert step number to graph edge indices."""
        # This uses the same edge ordering as in the original code
        count = 0
        for k in range(self.N):
            for j in range(k + 1, self.N):
                if count == step:
                    return k, j
                count += 1
        
        # Should never reach here
        raise ValueError(f"Invalid step: {step}")
