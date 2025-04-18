import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from gym_env import RiskEnv
from AlphaZero.alpha_mcts import getStateInfo
from Game.config import *

# Set seed for reproducibility
set_seed(0)

# Constants
NUM_EPISODES = 50
MAX_STEPS = 3000

# Initialize environment
env = RiskEnv(max_steps=MAX_STEPS)

# Define model paths and how many players each should control
model_configs = {
    "100_model_30.pth": 2,
    "500_model_30.pth": 2,
    "new_model_30.pth": 2,

}

# Determine if random players are needed
total_assigned = sum(model_configs.values())
random_players_present = total_assigned < PLAYERS

# Load models
loaded_models = {}
for path in model_configs:
    if os.path.exists(path):
        model = torch.load(path)
        model.eval()
        loaded_models[path] = model
    else:
        raise ValueError(f"Model path '{path}' does not exist.")

# Track statistics
model_wins = {path: 0 for path in model_configs}
model_wins["random"] = 0
ties = 0

# Run self-play episodes
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    truncated = False

    # Assign players
    assignments = []
    for path, count in model_configs.items():
        assignments.extend([path] * count)

    num_randoms = PLAYERS - len(assignments)
    assignments.extend([None] * num_randoms)
    random.shuffle(assignments)

    model_path_dict = {i: assignments[i] for i in range(PLAYERS)}
    model_player_dict = {
        i: loaded_models.get(assignments[i], None) for i in range(PLAYERS)
    }

    # Play the game
    while not done and not truncated:
        _, current_player_index, _, _ = getStateInfo(obs)
        model = model_player_dict[current_player_index]

        if model is None:
            action = env.action_space.sample()
        else:
            v, action = model.sample_action(torch.tensor(obs, dtype=torch.float32))
            action = action.detach().numpy()

        obs, reward, done, truncated, info = env.step(action)

    print(f"Episode {episode + 1} finished in {env.total_steps} steps.")

    # Track winner
    winner = info.get("Winner")
    if winner is not None:
        winner_path = model_path_dict[winner]
        if winner_path is not None:
            model_wins[winner_path] += 1
            print(f"Winner: Player {winner} using model '{winner_path}'")
        else:
            model_wins["random"] += 1
            print(f"Winner: Player {winner} using random strategy")
    else:
        ties += 1
        print("Game ended in a tie.")

# Close environment
env.close()

# === Results Summary ===
print("\n=== Win Statistics ===")
for path, wins in model_wins.items():
    if path == "random" and not random_players_present:
        continue
    label = "Random" if path == "random" else path
    print(f"{label}: {wins} wins")
print(f"Ties: {ties}")

# === Plot Pie Chart ===
labels = []
values = []

for path, wins in model_wins.items():
    if path == "random" and not random_players_present:
        continue
    label = "Random" if path == "random" else path
    labels.append(label)
    values.append(wins)

if ties > 0:
    labels.append("Ties")
    values.append(ties)

plt.figure(figsize=(8, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Win & Tie Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()
