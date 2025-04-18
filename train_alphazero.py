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
import time 

set_seed(0)
#initialize environment
env = RiskEnv(max_steps=2000)
#initialize the connections dictionary
territories = list(env.game.board.board_dict.values())
print(territories)

adjacency_dict = {}

for i, territory in enumerate(territories):
    print(territory,territory.adjecency_list)
    adjacency_indexes = [territories.index(adj) for adj in territory.adjecency_list]
    adjacency_dict[i] = adjacency_indexes

#initialize the continent dictionary
name_to_index = {name: idx for idx, name in enumerate(env.game.board.board_dict.keys())}
    
continent_dict = {
    continent_name: ([name_to_index[territory] for territory in data[2] if territory in name_to_index], data[3])
    for continent_name, data in env.game.board.continent_dict.items()
    if data[0]
}
print(continent_dict)

#set the model path
model_path = "model_30.pth"

#load the model if it exists
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    model = AlphaZeroModel(env.observation_space.shape[0],128,env.action_space.shape[0])

#initialize the data
#the data will be in the form state, action, player index of turn
#then, once the game is finished, the player index of turn column 
#will be adjusted to -1, 0, or 1 according to depending on if they won or not
dataset = RiskDataset(50000)

#training parameters
num_episodes = 1000
num_episodes_per_update = 1
episode_lengths = []

#self play
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    #check if it is time to update the model
    if episode % num_episodes_per_update == 0 and episode > 0:
        print("Training Model.")
        #train the model on the dataset
        states = torch.tensor(np.array(dataset.df['state'].to_list()),dtype=torch.float32)
        actions = torch.tensor(np.array(dataset.df['action'].to_list()),dtype=torch.float32)
        values = torch.tensor(dataset.df['value'].values,dtype=torch.float32)
        model.train_model(states,values,actions)
        if episode > 10:
            num_episodes_per_update = 10
        if episode > 200:
            num_episodes_per_update = 50

    #check if it is time to save the model
    if episode % 100 == 0 and episode > 0:
        torch.save(model, f"{episode}_{model_path}")
    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    
    #play the environment using the network
    while not done and not truncated:
        step += 1
        #initialize the tree search
        tree_search = AlphaMCTS(obs, model, adjacency_dict,continent_dict, num_simuations=30)
        #call search
        start_search_time = time.time()
        tree_search.search()
        end_search_time = time.time() 

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
torch.save(model, "new_"+model_path)

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
plt.savefig('steps_per_episode.png', dpi=300)
