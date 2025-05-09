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

###PARAMETERS###
NUM_EPISODES = 1001
UPDATE_FREQUENCY = 100
SAVE_FREQUENCY = 100
MODEL_PATH = "model_path.pth"
###############

#initialize environment
set_seed(0)
env = RiskEnv(max_steps=500)
#initialize the connections dictionary
territories = list(env.game.board.board_dict.values())

adjacency_dict = {}

for i, territory in enumerate(territories):
    adjacency_indexes = [territories.index(adj) for adj in territory.adjecency_list]
    adjacency_dict[i] = adjacency_indexes
#adjacency matrix
adjacency_matrix = build_adjacency_matrix(adjacency_dict)
#initialize the continent dictionary
name_to_index = {name: idx for idx, name in enumerate(env.game.board.board_dict.keys())}
    
continent_dict = {
    continent_name: ([name_to_index[territory] for territory in data[2] if territory in name_to_index], data[3])
    for continent_name, data in env.game.board.continent_dict.items()
    if data[0]
}

#load the model if it exists
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
else:
    model = AlphaZeroModel(TERRITORIES*(3+PLAYERS+PHASES)+TERRITORIES*TERRITORIES,256,env.action_space.n)
#set model to eval for inference
model.eval()

#initialize the data
#the data will be in the form state, action, player index of turn
#then, once the game is finished, the player index of turn column 
#will be adjusted to -1, 0, or 1 according to depending on if they won or not
dataset = RiskDataset(env.max_steps*125,adjacency_dict)

#training parameters
num_episodes = NUM_EPISODES
num_episodes_per_update = UPDATE_FREQUENCY
episode_lengths = []
#tree search depth
def tree_search_depth(episode):
       return 10

#self play
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    #check if it is time to update the model
    if episode % num_episodes_per_update == 0 and episode > 0:
        dataset.action_distributions_log(dataset.df)
        #train start time
        train_start_time = time.time()
        #Set model to train
        model.train()
        #train the model on the dataset
        states = torch.tensor(np.array(dataset.df['state'].to_list()),dtype=torch.float32)
        actions = torch.tensor(np.array(dataset.df['action'].to_list()),dtype=torch.float32)
        values = torch.tensor(dataset.df['value'].values,dtype=torch.float32)
        model.train_model(states,values,actions)
        #set model back to eval
        model.eval()
        #train end time
        train_end_time = time.time()
        print(f"Training Time: {train_end_time - train_start_time}")

    #check if it is time to save the model
    if episode % SAVE_FREQUENCY == 0 and episode > 0:
        torch.save(model, f"{episode}_"+MODEL_PATH)
    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    #get number of simulations
    num_simulations = tree_search_depth(episode)
    
    #play the environment using the network
    while not done and not truncated:
        #get obs inof
        player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(obs)
        step += 1
        #initialize the tree search
        tree_search = AlphaMCTS(obs, model, adjacency_dict,adjacency_matrix,continent_dict, num_simulations=num_simulations)
        #call search
        start_search_time = time.time()
        tree_search.search()
        end_search_time = time.time() 

        #get probability distribution from Alpha MCTS
        action = tree_search.get_final_action_distribution()
        #sample the action
        env_action = np.random.choice(len(action), p=action)
        
        #assemble data for this step
        data_step = (obs, action, current_player_index)

        #add to buffer
        dataset.push(data_step)

        #take a step in the environment
        obs, _, done, truncated, info = env.step(env_action)
        
    #add the game outcome to the current data being collected
    #if it was a draw clear the buffer
    if truncated:
         dataset.buffer.clear()
    else:
        dataset.add_game_outcome(info["Winner"],adjacency_matrix,True)

    episode_lengths.append(step)

dataset.action_distributions_log(dataset.df,True)
#save model at the end
torch.save(model, "new_"+MODEL_PATH)

#save dataset
dataset.save(f"{MODEL_PATH}_dataset.csv")

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
