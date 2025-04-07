import torch
import numpy as np
from Game.config import *
from gym_env import RiskEnv

set_seed(0)
#initialize environment
env = RiskEnv(max_steps=10,player_debug_mode=False)

#parameters
num_episodes = 1

#play the environment
for episode in range(num_episodes):

    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    action = None
    perspective_player_index = env.perspective_player_index
    print(f"Perspective Player: {perspective_player_index}")
    print(f"Episode: {episode}")
    env.render()
    
    #play the environment 
    while not done and not truncated:
        #sample random action
        action = env.action_space.sample()
        #take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        
        #render
        env.render()

    print(f"Total Steps: {env.total_steps}")

#close the environment
env.close()
