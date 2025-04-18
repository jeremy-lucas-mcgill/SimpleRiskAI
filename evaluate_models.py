import os
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import json
import argparse

from SimpleRiskAI.gym_env import RiskEnv
from SimpleRiskAI.risk_sb3_wrapper import RiskSB3Wrapper
from SimpleRiskAI.Game.config import *


# Tournament settings
N_GAMES = 10
MAX_STEPS = 500

def load_models(model_dir):
    """Load all trained models from the specified directory"""
    models = {}
    
    # Find all model files
    for file in os.listdir(model_dir):
        if file.endswith("_final.zip"):
            algo_name = file.split("_final.zip")[0]
            model_path = os.path.join(model_dir, file)
            
            # Determine which algorithm class to use
            if algo_name == "PPO":
                model = PPO.load(model_path)
                discrete = True
            elif algo_name == "A2C":
                model = A2C.load(model_path)
                discrete = True
            elif algo_name == "DQN":
                model = DQN.load(model_path)
                discrete = True
            elif algo_name == "SAC":
                model = SAC.load(model_path)
                discrete = False
            else:
                continue  # Skip unknown models
            
            models[algo_name] = {
                "model": model,
                "discrete": discrete
            }
    
    # Try to load AlphaZero model if available
    try:
        az_model_path = "alphazero_model_500.pth"
        if os.path.exists(az_model_path):
            alphazero_model = th.load(az_model_path)
            models["AlphaZero"] = {
                "model": alphazero_model,
                "discrete": False
            }
    except Exception as e:
        print(f"Failed to load AlphaZero model: {e}")
    
    return models

def create_random_agent():
    """Create a random agent for comparison"""
    class RandomAgent:
        def predict(self, observation, deterministic=False):
            action = np.random.randint(42)  # 42 = TERRITORIES + 1
            return action, None
    
    return {
        "model": RandomAgent(),
        "discrete": True
    }

def run_tournament(models, output_dir):
    """Run a tournament between all models and a random agent"""
    model_names = list(models.keys()) + ["Random"]
    results = {name: {opponent: [] for opponent in model_names} for name in model_names}
    
    # Add random agent
    models["Random"] = create_random_agent()
    
    # Run games between all pairs of models
    for model1_name in tqdm(model_names, desc="Tournament Progress"):
        for model2_name in model_names:
            if model1_name == model2_name:
                continue  # Skip self-play
            
            # Create environment for this match
            env = RiskEnv(max_steps=MAX_STEPS)
            
            # Record outcomes of multiple games
            outcomes = []
            for game in range(N_GAMES):
                outcome = run_game(
                    env, 
                    models[model1_name], 
                    models[model2_name], 
                    model1_name, 
                    model2_name
                )
                outcomes.append(outcome)
                results[model1_name][model2_name].append(outcome)
            
            env.close()
    
    # Compute win rates
    win_rates = {name: {opponent: 0 for opponent in model_names} for name in model_names}
    for model1_name in model_names:
        for model2_name in model_names:
            if model1_name == model2_name:
                continue
            
            # Calculate win rate
            wins = results[model1_name][model2_name].count(1)
            losses = results[model1_name][model2_name].count(-1)
            draws = results[model1_name][model2_name].count(0)
            
            total_games = len(results[model1_name][model2_name])
            win_rate = wins / total_games if total_games > 0 else 0
            
            win_rates[model1_name][model2_name] = win_rate
    
    # Save results
    results_df = pd.DataFrame(win_rates)
    results_df.to_csv(os.path.join(output_dir, "tournament_results.csv"))
    
    # Save detailed results
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f)
    
    # Create heatmap visualization
    create_heatmap(win_rates, model_names, output_dir)
    
    return win_rates, results

def run_game(env, model1_info, model2_info, model1_name, model2_name):
    """Run a single game between two models"""
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    
    # Track which model controls which player
    player_to_model = {
        0: model1_info,
        1: model2_info
    }
    
    player_to_name = {
        0: model1_name,
        1: model2_name
    }
    
    while not (done or truncated):
        current_player = env.game.currentPlayer
        
        # Skip if player is not one of our models
        if current_player > 1:
            # Use random action for other players
            action = env.action_space.sample()
        else:
            # Get the appropriate model for this player
            model_info = player_to_model[current_player]
            model = model_info["model"]
            
            # Convert observation for AlphaZero model
            if player_to_name[current_player] == "AlphaZero":
                # Sample action from AlphaZero model
                value, probabilities = model.sample_action(th.tensor(obs, dtype=th.float32))
                action = probabilities.detach().numpy()
            else:
                # Use Stable-Baselines3 model
                action, _ = model.predict(obs, deterministic=True)
                
                # Convert discrete action to one-hot if needed
                if model_info["discrete"]:
                    action_vec = np.zeros(env.action_space.shape[0], dtype=np.float32)
                    action_vec[action] = 1.0
                    action = action_vec
        
        # Take step in environment
        obs, _, done, truncated, info = env.step(action)
        steps += 1
        
        # Check if game ended
        if done or truncated:
            if done:
                winner = info.get("Winner")
                if winner == 0:
                    return 1  # Model 1 won
                elif winner == 1:
                    return -1  # Model 2 won
            
            return 0  # Draw or other players won
    
    return 0  # Should not reach here

def create_heatmap(win_rates, model_names, output_dir):
    """Create a heatmap visualization of tournament results"""
    # Convert win_rates to a matrix
    win_rate_matrix = np.zeros((len(model_names), len(model_names)))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if model1 != model2:
                win_rate_matrix[i, j] = win_rates[model1][model2]
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(win_rate_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add labels and title
    plt.colorbar(label='Win Rate')
    plt.title('Tournament Results: Win Rates')
    plt.xlabel('Opponent')
    plt.ylabel('Model')
    
    # Add ticks and labels
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.yticks(np.arange(len(model_names)), model_names)
    
    # Add win rate values in cells
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if i != j:
                plt.text(j, i, f'{win_rate_matrix[i, j]:.2f}', 
                         ha='center', va='center', 
                         color='black' if 0.3 <= win_rate_matrix[i, j] <= 0.7 else 'white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tournament_heatmap.png'))
    plt.close()

def evaluate_against_alphazero(models, output_dir):
    """Evaluate SB3 models against AlphaZero specifically"""
    if "AlphaZero" not in models:
        print("AlphaZero model not found, skipping comparison")
        return
    
    alphazero = models["AlphaZero"]
    sb3_models = {k: v for k, v in models.items() if k != "AlphaZero" and k != "Random"}
    
    results = {}
    for model_name, model_info in sb3_models.items():
        env = RiskEnv(max_steps=MAX_STEPS)
        
        # Run multiple games
        outcomes = []
        for game in range(N_GAMES):
            outcome = run_game(env, model_info, alphazero, model_name, "AlphaZero")
            outcomes.append(outcome)
        
        # Calculate win rate against AlphaZero
        wins = outcomes.count(1)
        losses = outcomes.count(-1)
        draws = outcomes.count(0)
        
        results[model_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(outcomes)
        }
        
        env.close()
    
    # Save results
    results_df = pd.DataFrame(results).transpose()
    results_df.to_csv(os.path.join(output_dir, "alphazero_comparison.csv"))
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    win_rates = [results[model]["win_rate"] for model in sb3_models.keys()]
    plt.bar(sb3_models.keys(), win_rates)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Equal Performance')
    plt.ylim(0, 1)
    plt.title('Performance Against AlphaZero')
    plt.xlabel('Model')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alphazero_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate RL models for Risk game')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = load_models(args.model_dir)
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Run tournament
    print(f"Running tournament with {N_GAMES} games per matchup...")
    win_rates, detailed_results = run_tournament(models, args.output_dir)
    
    # Evaluate against AlphaZero
    print("Evaluating against AlphaZero...")
    evaluate_against_alphazero(models, args.output_dir)
    
    print(f"Evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()