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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

# Import Gym environment
import gymnasium as gym
from graph_gym_env import GraphGymEnv

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
        self.best_model_path = os.path.join(MODELS_DIR, "best_model")
        self.rewards_history = {
            "timesteps": [],
            "mean_rewards": [],
            "best_rewards": [],
            "episode_rewards": []
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

    # Create and assign the callback
    callback = RewardTrackingCallback(eval_env=eval_env)

    # Train the model
    print("Training model...")
    total_timesteps = 171 * 100 * 1000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    final_model_path = os.path.join(MODELS_DIR, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {callback.best_model_path}")

    # Log total training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

# Entry point
if __name__ == "__main__":
    main()
