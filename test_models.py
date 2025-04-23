import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from gym_env import RiskEnv
from AlphaZero.alpha_mcts import getStateInfo, enrich_features, build_adjacency_matrix
from Game.config import *

#config
set_seed(0)
NUM_EPISODES = 100
MAX_STEPS = 500
RENDER = False
ARGMAX = False

#environment setup
env = RiskEnv(max_steps=MAX_STEPS)

territories = list(env.game.board.board_dict.values())
adjacency_dict = {i: [territories.index(adj) for adj in t.adjecency_list] for i, t in enumerate(territories)}
adjacency_matrix = build_adjacency_matrix(adjacency_dict)

#model setup 
model_configs = {
    "Models\\NA SA EUR AFR Models\\300_NA_SA_EUR_AFR.pth": 1,
}
PLAYERS_TOTAL = PLAYERS
total_assigned = sum(model_configs.values())
random_players_present = total_assigned < PLAYERS_TOTAL

loaded_models = {}
for path in model_configs:
    if os.path.exists(path):
        model = torch.load(path)
        model.eval()
        loaded_models[path] = model
    else:
        raise FileNotFoundError(f"Model '{path}' not found.")

#tracking setup
model_wins = {path: 0 for path in model_configs}
model_wins["random"] = 0
ties = 0

#game loop
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    truncated = False
    RENDER and env.render(render_mode="Visual")

    #get first Player
    _, current_player_index, _, _ = getStateInfo(obs)
    #assign players to models/random
    assignments = []
    for path, count in model_configs.items():
        assignments.extend([path] * count)
    assignments.extend([None] * (PLAYERS_TOTAL - len(assignments)))
    random.shuffle(assignments)

    #track assignment by index
    player_path_dict = {i: assignments[i] for i in range(PLAYERS_TOTAL)}
    player_model_dict = {i: loaded_models.get(assignments[i], None) for i in range(PLAYERS_TOTAL)}
    player_index_order = [(current_player_index + i) % PLAYERS_TOTAL for i in range(PLAYERS_TOTAL)]

    #play one episode
    while not done and not truncated:
        _, current_player_index, _, _ = getStateInfo(obs)
        enriched_obs = enrich_features(obs, adjacency_matrix)
        model = player_model_dict[current_player_index]

        if model is None:
            action = env.action_space.sample()
        else:
            v, policy = model.sample_action(torch.tensor(enriched_obs, dtype=torch.float32))
            policy = policy.detach().numpy()
            action = np.argmax(policy) if ARGMAX else np.random.choice(len(policy), p=policy)

        obs, reward, done, truncated, info = env.step(action)
        RENDER and env.render(render_mode="Visual")

    print(f"Episode {episode + 1} finished in {env.total_steps} steps.")

    #get winner
    winner = info.get("Winner")
    if winner is not None:
        winner_path = player_path_dict[winner]

        #win counts
        if winner_path is not None:
            model_wins[winner_path] += 1
            print(f"Winner: Player {winner} using model '{winner_path}'")
        else:
            model_wins["random"] += 1
            print(f"Winner: Player {winner} using random strategy")
    else:
        ties += 1
        print("Game ended in a tie.")
        continue

#stats summary
env.close()

print("\n=== Win Statistics ===")
for path, wins in model_wins.items():
    if path == "random" and not random_players_present:
        continue
    label = "Random" if path == "random" else path
    print(f"{label}: {wins} wins")
print(f"Ties: {ties}")

#pie chart
labels = []
values = []
for path, wins in model_wins.items():
    if path == "random" and not random_players_present:
        continue
    labels.append("Random" if path == "random" else path)
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
