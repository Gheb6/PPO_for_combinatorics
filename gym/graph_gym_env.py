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
    
    When reverse_mode=True:
    - Starts with a complete graph
    - Action 0 removes an edge, Action 1 keeps it
    - Moves that would disconnect the graph automatically use Action 1
    
    When reverse_mode=False:
    - Starts with an empty graph
    - Action 0 doesn't add an edge, Action 1 adds it
    """

    def __init__(self, n_vertices=19, reverse_mode=True):
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
        
        # Very large negative number for invalid states
        self.INF = 1000000
        
        # Flag for reverse game mode (configurable)
        self.reverse_mode = reverse_mode

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        # Set seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Initialize graph based on mode
        if self.reverse_mode:
            # Complete graph for reverse mode
            self.G = nx.complete_graph(self.N)
        else:
            # Empty graph for normal mode
            self.G = nx.Graph()
            self.G.add_nodes_from(list(range(self.N)))

        # Reset step counter
        self.current_step = 0

        # Initialize state
        self.state = np.zeros(2 * self.num_edges, dtype=np.int32)
        
        # In reverse mode, all edges exist initially
        if self.reverse_mode:
            for i in range(self.num_edges):
                self.state[i] = 1
                
        # Mark first position
        self.state[self.num_edges] = 1

        
        return self.state, {}  # Return initial state and empty info dict per Gymnasium API

    def step(self, action):
        """
        Take a step in the environment by deciding whether to add7remove an edge.
        
        Args:
            action: 0 (don't add / remove edge) or 1 (add / keep edge)
            
        Returns:
            observation: The new state
            reward: The reward for the action
            terminated: Whether the episode is complete
            truncated: Whether the episode was truncated (unused here)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Get the edge indices for the current step
        i, j = self._step_to_edge(self.current_step)
        
        # Handle action based on mode
        if self.reverse_mode:
            # In reverse mode: action 0 = remove edge, action 1 = keep edge
            if action == 0:
                # Check if removing the edge would disconnect the graph
                self.G.remove_edge(i, j)
                is_connected = nx.is_connected(self.G)
                
                if not is_connected:
                    # Illegal move - restore edge and treat as action 1
                    self.G.add_edge(i, j)
                    action = 1  # Override to action 1
                    
                    # No need to add edge as it's already present
                    
                # Update state with (potentially modified) action
                self.state[self.current_step] = action
            else:
                # Action 1 (keep edge) - edge already exists
                self.state[self.current_step] = 1
        else:
            # In normal mode: action 0 = don't add edge, action 1 = add edge
            self.state[self.current_step] = action
            
            # Add edge if action is 1
            if action == 1:
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
            return -10 * (len(components) - 1)  # Still negative but much less severe
        
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
