import torch
import pandas as pd
import torch
import numpy as np
import random
import os
from AlphaZero.alphazero_model import AlphaZeroModel
from gym_env import RiskEnv
from AlphaZero.dataset_for_alphazero import RiskDataset
from Game.config import *
from AlphaZero.alpha_mcts import *
from tqdm import tqdm 
import  matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)          # Python's built-in random module
    np.random.seed(seed)       # NumPy
    torch.manual_seed(seed)    # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility

set_seed(0)  # Call this function at the start of your script
#initialize environment
env = RiskEnv(max_steps=500)

#set the model path
model_path = "alphazero_model.pth"

#load the model if it exists
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    model = AlphaZeroModel(env.observation_space.shape[0],128,env.action_space.shape[0])

#initialize the data
#the data will be in the form state, action, player index of turn
#then, once the game is finished, the player index of turn column 
#will be adjusted to -1, 0, or 1 according to depending on if they won or not
dataset = RiskDataset(30000)

#training parameters
num_episodes = 100
num_episodes_per_update = 10
num_episodes_per_save = 100
episode_lengths = []

#self play
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    #check if it is time to save the model
    if episode % num_episodes_per_update == 0 and episode > 0:
        torch.save(model, f"{model_path}_{episode}")

    #check if it is time to update the model
    if episode % num_episodes_per_update == 0 and episode > 0:
        #train the model on the dataset
        states = torch.tensor(np.array(dataset.df['state'].to_list()),dtype=torch.float32)
        actions = torch.tensor(np.array(dataset.df['action'].to_list()),dtype=torch.float32)
        values = torch.tensor(dataset.df['value'].values,dtype=torch.float32)
        model.train_model(states,values,actions)
        #set new update frequency
        if episode < 100:
            num_episodes_per_update = 10
        else:
            num_episodes_per_update = 100
    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    
    #play the environment using the network
    while not done and not truncated:
        step += 1
        #initialize the tree search
        tree_search = AlphaMCTS(obs, model,num_simuations=10)

        #call search
        tree_search.search()

        #get probability distribution from Alpha MCTS
        action = tree_search.get_final_action_distribution()

        #take a step in the environment
        obs, _, done, truncated, info = env.step(action)
        
        #assemble data for this step
        data_step = (obs, action, info["Current Player"])

        #add to buffer
        dataset.push(data_step)

    #add the game outcome to the current data being collected
    if truncated:
        dataset.add_game_outcome(-1)
    else:
        dataset.add_game_outcome(info["Winner"])
    
    episode_lengths.append(step)

#save model at the end
torch.save(model, model_path)

#close the environment
env.close()

#print game length over training
plt.figure(figsize=(10, 6))
plt.plot(episode_lengths, label='Steps per Episode', color='royalblue', linewidth=2)

plt.title('Steps per Episode Over Games', fontsize=16)
plt.xlabel('Game Number', fontsize=14)
plt.ylabel('Steps per Episode', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
