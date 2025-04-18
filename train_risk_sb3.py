import os
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from SimpleRiskAI.gym_env import RiskEnv
from SimpleRiskAI.risk_sb3_wrapper import RiskSB3Wrapper
from FinalProject.risk_policy_network import RiskFeaturesExtractor, RiskMaskedPolicy
from SimpleRiskAI.Game.config import *


# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

# Create log directory
def create_log_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs/risk_sb3_{timestamp}"
    model_dir = f"{log_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir

# Create vectorized environment
def make_risk_env(n_envs=4, discrete=True, reward_shaping=True, seed=0):
    def _init():
        env = RiskEnv(max_steps=500)
        env = RiskSB3Wrapper(env, discrete_actions=discrete, reward_shaping=reward_shaping)
        env = Monitor(env)
        return env
    
    vec_env = make_vec_env(_init, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    return vec_env

# Define model hyperparameters
def get_model_hyperparams():
    return {
        "PPO": {
            "policy_kwargs": dict(
                features_extractor_class=RiskFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[128, 128], vf=[128, 128])
            ),
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "A2C": {
            "policy_kwargs": dict(
                features_extractor_class=RiskFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[128, 128], vf=[128, 128])
            ),
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
        "DQN": {
            "policy_kwargs": dict(
                features_extractor_class=RiskFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[128, 128]
            ),
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "SAC": {
            "policy_kwargs": dict(
                features_extractor_class=RiskFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[128, 128], qf=[128, 128])
            ),
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
        }
    }

# Create and train a model
def train_model(algo_name, env, total_timesteps, log_dir, model_dir, hyperparams):
    # Create the algorithm class
    if algo_name == "PPO":
        algo_class = PPO
        discrete = True
    elif algo_name == "A2C":
        algo_class = A2C
        discrete = True
    elif algo_name == "DQN":
        algo_class = DQN
        discrete = True
    elif algo_name == "SAC":
        algo_class = SAC
        discrete = False
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Create a new environment with the correct action space if needed
    if env.discrete_actions != discrete:
        env.close()
        env = make_risk_env(n_envs=4, discrete=discrete)
    
    # Create evaluation environment
    eval_env = RiskEnv(max_steps=500)
    eval_env = RiskSB3Wrapper(eval_env, discrete_actions=discrete)
    eval_env = Monitor(eval_env)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"{algo_name}"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Create and train the model
    model = algo_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        **hyperparams
    )
    
    print(f"\nTraining {algo_name} for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=algo_name
    )
    
    # Save the final model
    model.save(f"{model_dir}/{algo_name}_final")
    
    # Close environments
    eval_env.close()
    
    return model

# Evaluate model against random agent
def evaluate_against_random(model, env, n_episodes=10):
    victories = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            
            # Check if model's player won
            if done and info.get("Winner") == info.get("Current Player"):
                victories += 1
        
        episode_lengths.append(steps)
    
    win_rate = victories / n_episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)
    
    return win_rate, avg_length

# Main training function
def main():
    set_seed(42)
    
    # Create log directory
    log_dir, model_dir = create_log_dir()
    print(f"Logs will be saved to {log_dir}")
    
    # Create environment
    env = make_risk_env()
    
    # Get hyperparameters
    hyperparams = get_model_hyperparams()
    
    # Define algorithms to train
    algorithms = ["PPO", "A2C", "DQN", "SAC"]
    
    # Total timesteps for training
    total_timesteps = 250000
    
    # Train models
    results = {}
    for algo_name in algorithms:
        model = train_model(
            algo_name,
            env,
            total_timesteps,
            log_dir,
            model_dir,
            hyperparams[algo_name]
        )
        
        # Evaluate model
        eval_env = RiskEnv(max_steps=500)
        eval_env = RiskSB3Wrapper(
            eval_env, 
            discrete_actions=(algo_name in ["PPO", "A2C", "DQN"])
        )
        
        win_rate, avg_length = evaluate_against_random(model, eval_env, n_episodes=20)
        results[algo_name] = {
            "win_rate": win_rate,
            "avg_episode_length": avg_length
        }
        
        eval_env.close()
    
    # Save results
    results_df = pd.DataFrame(results).transpose()
    results_df.to_csv(f"{log_dir}/evaluation_results.csv")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), [r["win_rate"] for r in results.values()])
    plt.title("Win Rate Against Random Agent")
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), [r["avg_episode_length"] for r in results.values()])
    plt.title("Average Episode Length")
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/evaluation_results.png")
    plt.close()
    
    # Close environment
    env.close()
    
    print(f"\nTraining complete! Results saved to {log_dir}")
    print("Evaluation Results:")
    print(results_df)

if __name__ == "__main__":
    main()