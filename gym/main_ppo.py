import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

# Import Gym environment
import gymnasium as gym
from graph_gym_env import GraphGymEnv
import datetime 

# Set random seed for reproducibility
seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)

# Define constants
N_VERTICES = 19  # Number of vertices in the graph
LOG_DIR = "ppo_graph_gym_logs"        # Directory for TensorBoard logs
MODELS_DIR = "ppo_graph_gym_models"    # Directory for saving models
RESULTS_DIR = "ppo_graph_gym_results"  # Directory for saving training results

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy using a specific MLP architecture (128 -> 64) with ReLU activations
    and Xavier (Glorot) initialization.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],              # Disable default net_arch
            activation_fn=nn.Identity, # No activation function by default
            *args,
            **kwargs
        )
        # Define the MLP extractor: shared feature extractor for actor and critic
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[128, 64],
            activation_fn=nn.ReLU
        )
        # Apply Xavier initialization to all linear layers
        for m in self.mlp_extractor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Finalize the model construction
        self._build(lr_schedule)

# Define a callback for tracking rewards during training
class RewardTrackingCallback(BaseCallback):
    """
    Callback that tracks rewards after each rollout and saves the best model found.
    """
    def __init__(self, eval_env, verbose=1):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf
        self.best_reward = -np.inf
        self.max_score = -np.inf  # Track the maximum score of any graph (even partial)
        self.best_graph = None  # Store the best graph
        self.best_model_path = os.path.join(MODELS_DIR, "best_model")
        self.rewards_history = {
            "timesteps": [],
            "mean_rewards": [],
            "best_rewards": [],
            "episode_rewards": [],
            "max_score": []
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _on_step(self):
        """
        Required method by BaseCallback, not used because _on_rollout_end handles evaluation.
        """
        return True

    def _on_rollout_end(self):
        """
        After each rollout, evaluate the agent and log rewards.
        Save the model if a new best reward is found.
        """
        deterministic_reward = self._run_evaluation_episode(deterministic=True)
        stochastic_rewards = [self._run_evaluation_episode(deterministic=False) for _ in range(4)]
        episode_rewards = [deterministic_reward] + stochastic_rewards
        mean_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)

        if max_reward > self.best_reward:
            self.best_reward = max_reward
            self.model.save(self.best_model_path)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        # Save reward statistics
        self.rewards_history["timesteps"].append(self.num_timesteps)
        self.rewards_history["mean_rewards"].append(float(mean_reward))
        self.rewards_history["best_rewards"].append(float(self.best_reward))
        self.rewards_history["episode_rewards"].append([float(r) for r in episode_rewards])

        with open(os.path.join(RESULTS_DIR, "rewards_history.json"), "w") as f:
            json.dump(self.rewards_history, f, indent=4)

        if self.verbose > 0:
            print(f"Timestep {self.num_timesteps}: Mean reward: {mean_reward:.2f}, Best reward: {self.best_reward:.2f}")

        return True

    def _run_evaluation_episode(self, deterministic=True):
        """
        Run a full episode using the current model and return the total reward.
        """
        obs, _ = self.eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        return episode_reward

class CounterexampleCheckingCallback(BaseCallback):
    """
    Callback that checks for counterexamples after each step and logs them.
    """
    def __init__(self, verbose=0):
        super(CounterexampleCheckingCallback, self).__init__(verbose)
        self.counterexamples_found = 0
        self.counterexample_timesteps = []
        self.max_score = -np.inf
        self.best_graph = None
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs("saved_graphs", exist_ok=True)
        
    def _on_step(self):
        """
        Check the latest info dict for counterexample flags and track highest score.
        """
        # Check the latest info dict from each environment
        for env_idx in range(len(self.training_env.envs)):
            info = self.locals.get('infos', [{}])[env_idx]
            
            # Track max score for any graph (complete or partial)
            if 'current_score' in info and info['current_score'] > self.max_score:
                self.max_score = info['current_score']
                self.best_graph = info.get('graph_copy', None)
                
                # Log the new best score
                if self.verbose > 0:
                    print(f"New max score: {self.max_score:.4f} at timestep {self.num_timesteps}")
                
                # Save the best graph
                if self.best_graph is not None:
                    self._save_best_graph()
            
            # Handle counterexamples (when score > 0)
            if info.get('counterexample', False):
                self.counterexamples_found += 1
                self.counterexample_timesteps.append(self.num_timesteps)
                
                # Log the discovery
                if self.verbose > 0:
                    print(f"Found counterexample #{self.counterexamples_found} at timestep {self.num_timesteps}")
                
                # Save the timesteps to a file
                with open(os.path.join(RESULTS_DIR, "counterexample_timesteps.json"), "w") as f:
                    json.dump({
                        "count": self.counterexamples_found,
                        "timesteps": self.counterexample_timesteps,
                        "max_score": float(self.max_score)
                    }, f, indent=4)
                    
        return True
    
    def _save_best_graph(self):
        """Save the graph with the highest score."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = "saved_graphs"
        
        # Save graph image
        plt.figure(figsize=(8, 8))
        nx.draw(self.best_graph, with_labels=True, node_color="lightblue", edge_color="gray")
        plt.title(f"Best Graph (Score: {self.max_score:.4f})")
        plt.savefig(os.path.join(directory, f"best_graph_score_{self.max_score:.4f}_{timestamp}.pdf"))
        plt.close()
        
        # Save adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.best_graph).todense()
        np.savetxt(os.path.join(directory, f"best_adj_matrix_{self.max_score:.4f}_{timestamp}.txt"), adj_matrix, fmt="%d")

# Main training function
def main():
    """
    Main function to train a PPO agent on GraphGymEnv using the custom policy.
    Tracks and saves best model and reward history.
    """
    # Create necessary directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Log random seed used
    start_time = time.time()
    print(f"Using random seed: {seed}")
    with open(os.path.join(RESULTS_DIR, "seed.txt"), "w") as f:
        f.write(f"Seed: {seed}")

    # Create environment instances
    print("Creating environment...")
    env = GraphGymEnv(n_vertices=N_VERTICES)
    eval_env = GraphGymEnv(n_vertices=N_VERTICES)
    env = DummyVecEnv([lambda: env])

    # Create the PPO model with custom actor-critic policy
    print("Creating PPO model with custom policy...")
    model = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=0.0001,
        n_steps=171 * 100,
        batch_size=100,
        n_epochs=10,
        gamma=1,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=seed
    )

    # Create callbacks
    # reward_callback = RewardTrackingCallback(eval_env=eval_env)
    counterexample_callback = CounterexampleCheckingCallback(verbose=1)
    
    # Combine callbacks
    # callback = CallbackList([reward_callback, counterexample_callback])

    # Train the model
    print("Training model...")
    total_timesteps = 171 * 100 * 1000
    model.learn(total_timesteps=total_timesteps, callback=counterexample_callback)

    # Save the final model
    final_model_path = os.path.join(MODELS_DIR, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    #print(f"Best model saved to {reward_callback.best_model_path}")
    print(f"Found {counterexample_callback.counterexamples_found} counterexamples")

    # Log total training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

# Entry point
if __name__ == "__main__":
    main()
