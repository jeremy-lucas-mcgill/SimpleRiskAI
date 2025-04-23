import torch
import numpy as np
import random
import os
from gym_env import RiskEnv
from AlphaZero.alpha_mcts import getStateInfo,getValidActions,enrich_features,build_adjacency_matrix,softmax
from Game.config import *
import torch.nn.functional as F

###PARAMETERS###
MODEL_PATH = "Models\\NA SA EUR AFR Models\\300_NA_SA_EUR_AFR.pth"
RENDER = False
NUM_EPISODES = 200
################

set_seed(0)
#initialize environment
maxsteps = 500
env = RiskEnv(max_steps=maxsteps)

#initialize the connections dictionary
territories = list(env.game.board.board_dict.values())

adjacency_dict = {}

for i, territory in enumerate(territories):
    adjacency_indexes = [territories.index(adj) for adj in territory.adjecency_list]
    adjacency_dict[i] = adjacency_indexes

adjacency_matrix = build_adjacency_matrix(adjacency_dict)

#load the model if it exists
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
    model.eval()
    print(model)
else:
    raise ValueError

#evaluation parameters
legal_moves_percentages = []
average_steps = []
average_legal_moves_phases = []
average_phase_count = []

#self play
for episode in range(NUM_EPISODES):

    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    legal_moves_count = 0
    legal_moves_phases = [0 for _ in range(PHASES)]
    phase_counts = [0 for _ in range(PHASES)]
    RENDER and env.render(render_mode="Visual")
    
    #play the environment using the network
    while not done and not truncated:
        #get state info
        player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(obs)
        obs = enrich_features(obs,adjacency_matrix)
        #get action from model
        v, action = model.sample_action(torch.tensor(obs,dtype=torch.float32))
        #turn action back to numpy
        action = action.detach().numpy()
        #get valid actions
        valid_actions = getValidActions(current_phase,player_territories[current_player_index],current_last_selected_index,adjacency_dict)
        #take argmax from remaining probabilities
        env_action = np.random.choice(len(action),p=action)
        #compare if it is valid
        if env_action in valid_actions:
            legal_moves_count += 1
            legal_moves_phases[current_phase-1] += 1
        #increment phase count
        phase_counts[current_phase - 1] += 1
        
        #take a step in the environment
        obs, reward, done, truncated, info = env.step(env_action)
        #render
        RENDER and env.render(render_mode="Visual")

    legal_moves_percentages.append(legal_moves_count / env.total_steps)
    average_steps.append(env.total_steps)
    average_phase_count.append(phase_counts)
    #compute average legal move per phase and append
    with np.errstate(divide='ignore', invalid='ignore'):
        average_per_phase = np.true_divide(legal_moves_phases, phase_counts)
        average_per_phase[~np.isfinite(average_per_phase)] = 0
        average_legal_moves_phases.append(average_per_phase)

print(f"Average Legal Move Percentage: {np.mean(legal_moves_percentages)}")
print(f"Legal Moves Phases: {np.mean(np.array(average_legal_moves_phases), axis=0)}")
print(f"Average Phase Counts: {np.mean(np.array(average_phase_count), axis=0)}")
print(f"Average Total Steps: {np.mean(average_steps)}")

#close the environment
env.close()
