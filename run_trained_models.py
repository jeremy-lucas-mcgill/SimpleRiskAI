import os
import argparse
import torch as th
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC

from gym_env import RiskEnv
from risk_sb3_wrapper import RiskSB3Wrapper
from Game.config import *

def load_model(model_path):
    """
    Load a trained model from the specified path.
    Automatically determines the algorithm from the filename.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine algorithm type from filename
    filename = os.path.basename(model_path)
    
    if "PPO" in filename:
        return PPO.load(model_path), True
    elif "A2C" in filename:
        return A2C.load(model_path), True
    elif "DQN" in filename:
        return DQN.load(model_path), True
    elif "SAC" in filename:
        return SAC.load(model_path), True
    elif "alphazero" in filename.lower():
        # For AlphaZero model
        return th.load(model_path), False
    else:
        raise ValueError(f"Unknown model type in filename: {filename}")

def run_model(model, is_sb3_model, num_episodes=3, render_mode="Visual", max_steps=500):
    """
    Run the specified model in the Risk environment for visualization.
    """
    # Create environment
    env = RiskEnv(max_steps=max_steps)
    
    # Wrap environment if using SB3 model
    if is_sb3_model:
        env = RiskSB3Wrapper(env, discrete_actions=True)
    
    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        
        print(f"\nRunning episode {episode+1}/{num_episodes}")
        
        # Render initial state
        env.render(render_mode=render_mode)
        
        while not (done or truncated):
            # Get action based on model type
            if is_sb3_model:
                # SB3 model
                action, _ = model.predict(obs, deterministic=True)
            else:
                # AlphaZero model
                value, probabilities = model.sample_action(th.tensor(obs, dtype=th.float32))
                action = probabilities.detach().numpy()
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            # Render state
            env.render(render_mode=render_mode)
            
            # Print information
            current_player = info.get("Current Player")
            print(f"Step {steps}: Player {current_player} | Reward: {reward:.2f}")
        
        # Print episode summary
        winner = info.get("Winner")
        print(f"\nEpisode {episode+1} ended after {steps} steps")
        if winner is not None:
            print(f"Winner: Player {winner}")
        else:
            print("No winner (draw or max steps reached)")
    
    # Close environment
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Run a trained model in the Risk environment')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--render_mode', type=str, default="Visual", choices=["Visual", "Text"],
                        help='Rendering mode (Visual or Text)')
    
    args = parser.parse_args()
    
    # Load model
    try:
        print(f"Loading model from {args.model_path}...")
        model, is_sb3_model = load_model(args.model_path)
        
        # Run model
        print("Running model...")
        run_model(
            model, 
            is_sb3_model, 
            num_episodes=args.episodes, 
            render_mode=args.render_mode
        )
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()