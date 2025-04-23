import numpy as np
from Game.config import *
from gym_env import RiskEnv

set_seed(1)
#initialize environment
env = RiskEnv(max_steps=500,player_debug_mode=False)

#parameters
num_episodes = 100
episode_lengths = []

#play the environment
for episode in range(num_episodes):

    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    action = None
    print(f"Episode: {episode}")
    env.render(render_mode='Visual')
    
    #play the environment 
    while not done and not truncated:
        #sample random action
        action = env.action_space.sample()
        #take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        
        #render
        env.render(render_mode='Visual')

    print(f"Total Steps: {env.total_steps}")
    episode_lengths.append(env.total_steps)
print(np.mean(episode_lengths))
#close the environment
env.close()
