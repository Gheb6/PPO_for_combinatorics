import os
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

# Import our custom environment
from graph_gym_env import GraphGymEnv

# Set random seeds using current time for randomness
seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)

# Define constants
N_VERTICES = 19
LOG_DIR = "ppo_graph_gym_logs"
MODELS_DIR = "ppo_graph_gym_models"
RESULTS_DIR = "ppo_graph_gym_results"

class RewardTrackingCallback(BaseCallback):
    """
    Custom callback for tracking rewards during training.
    Uses _on_rollout_end for evaluation while implementing required _on_step.
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

        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def _on_step(self):
        """
        This method is required by BaseCallback, but since we're using _on_rollout_end,
        we just return True here to satisfy the abstract method requirement.
        """
        return True

    def _on_rollout_end(self):
        # Run 1 deterministic evaluation episode
        deterministic_reward = self._run_evaluation_episode(deterministic=True)
        
        # Run 4 stochastic evaluation episodes
        stochastic_rewards = []
        for _ in range(4):
            stochastic_rewards.append(self._run_evaluation_episode(deterministic=False))
        
        # Combine all episode rewards
        episode_rewards = [deterministic_reward] + stochastic_rewards
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)

        # Update best reward if we have a new best
        if max_reward > self.best_reward:
            self.best_reward = max_reward
            # Save the best model
            self.model.save(self.best_model_path)

        # Update best mean reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        # Store the rewards history
        self.rewards_history["timesteps"].append(self.num_timesteps)
        self.rewards_history["mean_rewards"].append(float(mean_reward))
        self.rewards_history["best_rewards"].append(float(self.best_reward))
        self.rewards_history["episode_rewards"].append([float(r) for r in episode_rewards])

        # Save rewards history to file
        with open(os.path.join(RESULTS_DIR, "rewards_history.json"), "w") as f:
            json.dump(self.rewards_history, f, indent=4)

        if self.verbose > 0:
            print(f"Timestep {self.num_timesteps}: Mean reward: {mean_reward:.2f}, Best reward: {self.best_reward:.2f}")
            print(f"Deterministic: {deterministic_reward:.2f}, Stochastic: {np.mean(stochastic_rewards):.2f}")

        return True
        
    def _run_evaluation_episode(self, deterministic=True):
        """Run a single evaluation episode and return the total reward."""
        obs, _ = self.eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward

        return episode_reward

def main():
    """
    Main function to train a PPO agent on the GraphGymEnv environment.
    """
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Start timing
    start_time = time.time()
    
    # Log the seed used
    print(f"Using random seed: {seed}")
    with open(os.path.join(RESULTS_DIR, "seed.txt"), "w") as f:
        f.write(f"Seed: {seed}")

    print("Creating environment...")
    # Create environment
    env = GraphGymEnv(n_vertices=N_VERTICES)

    # Create evaluation environment (not wrapped in DummyVecEnv)
    eval_env = GraphGymEnv(n_vertices=N_VERTICES)

    # Create vector environment for training
    env = DummyVecEnv([lambda: env])

    print("Creating PPO model with MlpPolicy...")
    # Create the PPO model with MlpPolicy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=171 * 100, 
        batch_size=100,
        n_epochs=10,
        #n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed
    )

    # Create callback for tracking rewards
    callback = RewardTrackingCallback(eval_env=eval_env)

    print("Training model...")
    # Train the model
    total_timesteps = 171 * 100 * 1000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    # Save the final trained model
    final_model_path = os.path.join(MODELS_DIR, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {callback.best_model_path}")

    # Calculate and print total training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

    # Evaluate the best model
    print("Evaluating best model...")
    best_model = PPO.load(callback.best_model_path)

    # Generate statistics for test episodes
    test_episodes = 10
    scores = []
    lambda1_values = []
    mu_values = []

    for episode in range(test_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward

        # Get final score and info
        final_score = eval_env.calcScore()
        scores.append(final_score)

        # Store eigenvalue and matching number if available
        if hasattr(eval_env, 'G'):
            evals = np.linalg.eigvalsh(nx.adjacency_matrix(eval_env.G).todense())
            lambda1 = max(abs(evals))
            lambda1_values.append(lambda1)

            max_match = nx.max_weight_matching(eval_env.G)
            mu = len(max_match)
            mu_values.append(mu)

        print(f"Episode {episode+1} - Total reward: {total_reward}, Final score: {final_score}")

        # Visualize the final graph for the first episode
        if episode == 0:
            G = eval_env.G.copy()
            plt.figure(figsize=(10, 10))
            nx.draw(G, with_labels=True, node_color='lightblue',
                    node_size=500, edge_color='gray')
            plt.title(f"Final Graph - Score: {final_score}")
            plt.savefig(os.path.join(RESULTS_DIR, "final_graph.png"))
            plt.close()

    # Calculate and report statistics
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)

    print(f"Average score: {avg_score}")
    print(f"Best score: {best_score}")

    # Plot training progress
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(callback.rewards_history["timesteps"], callback.rewards_history["mean_rewards"], label='Mean Reward')
        plt.plot(callback.rewards_history["timesteps"], callback.rewards_history["best_rewards"], label='Best Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "training_progress.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting training progress: {e}")

    # Save comprehensive results to file
    results = {
        "seed": seed,
        "training_time_seconds": training_time,
        "training_time_hours": training_time/3600,
        "average_score": float(avg_score),
        "best_score": float(best_score),
        "all_scores": [float(s) for s in scores],
        "final_training_stats": {
            "best_mean_reward": float(callback.best_mean_reward),
            "best_reward": float(callback.best_reward)
        }
    }

    # Add eigenvalue and matching number stats if available
    if lambda1_values and mu_values:
        results["lambda1"] = {
            "mean": float(np.mean(lambda1_values)),
            "values": [float(v) for v in lambda1_values]
        }
        results["mu"] = {
            "mean": float(np.mean(mu_values)),
            "values": [float(v) for v in mu_values]
        }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"All results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
