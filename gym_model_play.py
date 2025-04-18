import torch
import numpy as np
import random
import os
from gym_env import RiskEnv
from AlphaZero.alpha_mcts import getStateInfo
from Game.config import *

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
env = RiskEnv(max_steps=2000)

#set the model path
model_path = "new_model_30.pth"

#load the model if it exists
if os.path.exists:
    model = torch.load(model_path)
    model.eval()
else:
    raise ValueError

#evaluation parameters
num_episodes = 3

#self play
for episode in range(num_episodes):

    #reset the environment
    obs, _ = env.reset()
    done = False
    truncated = False
    env.render(render_mode="Visual")
    
    #play the environment using the network
    while not done and not truncated:
        #get action from model
        v, action = model.sample_action(torch.tensor(obs,dtype=torch.float32))
        #turn action back to numpy
        action = action.detach().numpy()
        #take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        #render
        env.render(render_mode="Visual")

    print(env.total_steps)

#close the environment
env.close()
