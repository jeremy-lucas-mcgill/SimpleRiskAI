from Game.game import Game
from Game.player import Player
from Game.config import *
import math
import torch
import numpy as np
import time
import copy

#class for AlphaZero Monte Carlo Tree Search
#Input the root node
#output action distribution
class AlphaMCTS:
    def __init__(self,root_state,model,adjacency_dict,adjacency_matrix,continent_dict,num_simulations=100,tau=1):
        #create the root node with the root state, no parent, and probability 1
        self.root = Node(root_state,None,1)
        self.model = model
        self.num_simulations = num_simulations
        self.tau = tau
        self.adjacency_dict = adjacency_dict
        self.adjacency_matrix = adjacency_matrix
        self.continent_dict = continent_dict

    def search(self):
        for _ in range(self.num_simulations):
            #search for best leaf node
            node = self.select(self.root)
            #expand the node if it is not terminal
            if not isTerminalState(node.state):
                #expand the node and get the value
                value = self.expand(node)
            else:
                #if it is a terminal node, value = 1, as the game is won
                value = 1
            #backpropogate
            self.backpropogate(node,value)        
            
    def select(self,node):
        #selecting a leaf node
        while node.children:
            #choose the next node using highest pucb score
            best_action = max(node.children,key=lambda u: node.children[u].calculate_PUCB(node.N))
            node = node.children[best_action]
        return node
    
    def expand(self,node):
        #get the state info
        player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(node.state)
        #valid actions, should be a list of ints corresponding to the valid actions
        valid_actions = getValidActions(current_phase,player_territories[current_player_index],current_last_selected_index,self.adjacency_dict)
        #next states
        next_states = [getNextState(player_territories,current_player_index,current_phase,current_last_selected_index,self.continent_dict,action) for action in valid_actions]
        #probabilties
        value, probabilities = self.model.sample_action(torch.tensor(enrich_features(node.state,self.adjacency_matrix),dtype=torch.float32))
        #renormalize
        valid = probabilities[valid_actions]
        valid = (valid / valid.sum()) if valid.sum() != 0 else torch.full_like(valid, 1.0 / valid.numel())
        valid_probabilities = valid.detach().cpu().numpy()
        #expand the node
        node.expand(valid_actions,next_states,valid_probabilities)
        return value.item()
    
    def backpropogate(self,node,value):
        #get the player index that the original value belongs to
        _, positive_player_index, _, _ = getStateInfo(node.state)
        while node:
            #get the player of the current node
            _, node_player_index, _, _ = getStateInfo(node.state)
            #update the node based on the value and if it is the winning player
            added_value = value if node_player_index == positive_player_index else -value
            node.update_from_value(added_value)
            #go up the tree
            node = node.parent
    def get_final_action_distribution(self):
        player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(self.root.state)
        #get the total visit count per child
        visits = np.array([child.N for child in self.root.children.values()])
        #get the list of all the action names
        actions = list(self.root.children.keys())
        #adjust the probabilities based on tau
        probs = visits**(1 / self.tau)
        #normalize probabilities
        probs = probs / np.sum(probs)
        #return action probabilities
        sample_action = np.zeros(self.model.action_size)
        for (action,probability) in zip(actions,probs):
            sample_action[int(action)] = probability
        return sample_action

#node class for tree search
class Node:
    def __init__(self,state,parent,prob):
        #what state the nodes represent
        self.state = state
        #parent
        self.parent = parent
        # Win percentage. Sum of value / total times selected
        self.Q = 0
        # Number of times selected
        self.N = 0
        # NN assigned probability
        self.P = prob
        # Hyper parameter for UCB
        self.C = 1
        #instantiate children 
        self.children = {}
    def calculate_PUCB(self,parent_N):
        #calculate pucb and return
        pucb = self.Q + self.C * self.P * math.sqrt(parent_N / (1 + self.N))
        return pucb
    def update_from_value(self,value):
        #update N and Q using incremental averaging
        self.N += 1
        self.Q += (value - self.Q) / self.N
    def expand(self,valid_actions,next_states,probabilities):
        for (action,next_state,probability) in zip(valid_actions,next_states,probabilities):
            if action not in self.children:
                self.children[action] = Node(next_state,self,probability)
            else:
                print("ERROR: Duplicate Action!")
        

# Get all valid actions depending on the phase of the game, setting all probabilities equal to 1
def getValidActions(phase,player_terr_list,last_territory_selected_index,adjacency_dict):
    #create sample action equal to the length of all the territories + 1 for doing nothing
    sample_action = np.ones(len(player_terr_list) + 1)
    valid_action_indices = None
    #create match for each phase
    match (phase):
        case 1:
            #find the valid territories to place troops on
            valid_indices = [index for index, troops in enumerate(player_terr_list) if troops > 0]
            #set invalid actions to 0 probability
            valid_actions = [a if index in valid_indices else 0 for index,a in enumerate(sample_action)]
            #turn valid actions into indices [0,1,1,0,0] into [1,2]
            valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
        case 2:
            #Create a mask for valid attacks. Territory should be owned by the player,
            #have at least more than one troop, and have adjacency with at least one territory that the player doesn't own
            #this preliminary check makes sure the territories are owned and more than one troop 
            possible_indices = [index for index, troops in enumerate(player_terr_list) if troops > 1]
            #iterate through the possible territories making sure each has adjacency
            #with at least one territory that the player doesn't own
            #iterate through each index, calling AdjTerr(index) and checking if that list contains a territory in our list that is zero (meaning we don't own it)
            valid_indices = [index for index in possible_indices if any(player_terr_list[adj_index] == 0 for adj_index in AdjacentIndices(index,adjacency_dict))]
            #set invalid actions to 0 probability
            valid_actions = [a if index in valid_indices or index==len(sample_action) - 1 else 0 for index,a in enumerate(sample_action)]
            #turn valid actions into indices 
            valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
        case 3:
            #indexes that are adjacent and we don't own to the last selected index
            valid_indices = [index for index in AdjacentIndices(last_territory_selected_index,adjacency_dict) if player_terr_list[index] == 0]
            #set invalid actions to 0 probability
            valid_actions = [a if index in valid_indices else 0 for index,a in enumerate(sample_action)]
            #turn valid actions into indices 
            valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
        case 4:
            #this preliminary check makes sure the territories are owned and more than one troop 
            possible_indices = [index for index, troops in enumerate(player_terr_list) if troops > 1]

            #iterate through the possible territories making sure each has adjacency
            #with at least one territory that the player owns
            valid_indices = [index for index in possible_indices if any(player_terr_list[adj_index] != 0 for adj_index in AdjacentIndices(index,adjacency_dict))]
            #set invalid actions to 0 probability
            valid_actions = [a if index in valid_indices or index==len(sample_action) - 1 else 0 for index,a in enumerate(sample_action)]
            #turn valid actions into indices 
            valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
        case 5:
            #indexes that are adjacent and we do own to the last selected index
            valid_indices = [index for index in AdjacentIndices(last_territory_selected_index,adjacency_dict) if player_terr_list[index] != 0]
            #set invalid actions to 0 probability
            valid_actions = [a if index in valid_indices else 0 for index, a in enumerate(sample_action)]
            #turn valid actions into indices 
            valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
        case _:
            print("Invalid Phase")
            return None
    return valid_action_indices

#Get next state should return a 1d array of the next state
#territory matrix
#turn
#phase
#last index
def getNextState(player_territories,current_player_index,current_phase,current_last_selected_index,continent_dict,action):
    player_territories = np.array(player_territories)
    match(current_phase):
        #place troops
        case 1:
            #calculate territory bonus
            available = max(TERRITORIES_PER_TROOP,np.count_nonzero(player_territories[current_player_index]) // TERRITORIES_PER_TROOP)
            #calculate the continent bonus
            bonus = 0
            for (indices, troops) in continent_dict.values():
                #check if player owns all territories in the continent
                if all(player_territories[current_player_index][i] > 0 for i in indices):
                    bonus += troops
            player_territories[current_player_index][action] += available + bonus
            current_phase = 2
            current_last_selected_index = -1
        #select attack from
        case 2:
            current_last_selected_index,current_phase = (action,3) if action != TERRITORIES else (-1,4)
        #select attack to
        case 3:
            defending_player = np.nonzero(player_territories[:,action])[0].item()
            troops_attacking = player_territories[current_player_index][current_last_selected_index] - 1
            troops_defending = player_territories[defending_player][action]

            attacking_troops_left,defending_troops_left = handleAttack(troops_attacking,troops_defending)
            if attacking_troops_left <= 0:
                player_territories[current_player_index][current_last_selected_index] = 1
                player_territories[defending_player][action] = defending_troops_left
            elif defending_troops_left <= 0:
                player_territories[current_player_index][current_last_selected_index] = 1
                player_territories[defending_player][action] = 0
                player_territories[current_player_index][action] = attacking_troops_left
            current_phase = 2
            current_last_selected_index = -1
        #select fortify from
        case 4:
            if action != TERRITORIES:
                current_last_selected_index = action
                current_phase = 5
            else:
                current_last_selected_index = -1
                current_phase = 1
                while (np.all(player_territories[(current_player_index + 1) % PLAYERS] == 0)):
                    current_player_index = (current_player_index) + 1 % PLAYERS
                current_player_index = (current_player_index + 1) % PLAYERS
        #select fortify to
        case 5:
            amount_to_fortify = player_territories[current_player_index][current_last_selected_index] - 1
            player_territories[current_player_index][current_last_selected_index] = 1
            player_territories[current_player_index][action] += amount_to_fortify
            current_phase = 1
            while (np.all(player_territories[(current_player_index + 1) % PLAYERS] == 0)):
                    current_player_index = (current_player_index + 1) % PLAYERS
            current_player_index = (current_player_index + 1) % PLAYERS
            current_last_selected_index = -1
        #return the new state
    new_state = turnStateInfoTo1DArray(player_territories,current_player_index,current_phase,current_last_selected_index)
    return new_state

def getStateInfo(state):
    #extract info from state
    num_players = PLAYERS
    num_phases = PHASES
    num_territories = TERRITORIES
    player_territories = [state[(p)*num_territories:(p+1)*num_territories] for p in range(num_players)]
    one_hot_turn = state[num_players*num_territories:num_players*num_territories+num_players]
    one_hot_phase = state[num_players*num_territories+num_players:num_players*num_territories+num_players+num_phases]
    last_index = state[num_players*num_territories+num_players+num_phases:]

    #get the current player turn and phase, and the last selected index
    current_player_index = list(one_hot_turn).index(1)
    current_phase = list(one_hot_phase).index(1) + 1
    current_last_selected_index = list(last_index).index(1) if 1 in last_index else -1

    return player_territories,current_player_index,current_phase,current_last_selected_index

def turnStateInfoTo1DArray(player_territories,current_player_index,current_phase,current_last_selected_index):
    flat_player_territories = np.array(player_territories).flatten()
    one_hot_current_player_index = np.eye(PLAYERS)[current_player_index]
    one_hot_current_phase = np.eye(PHASES)[current_phase - 1]
    one_hot_current_last_selected_index = np.eye(TERRITORIES)[current_last_selected_index] if current_last_selected_index != -1 else np.zeros(TERRITORIES)
    return np.concatenate([flat_player_territories,one_hot_current_player_index,one_hot_current_phase,one_hot_current_last_selected_index])

def isTerminalState(state):
    #extract info from state
    player_territories,_,_,_ = getStateInfo(state)

    #check if one player has all the territories
    player_won = any(np.all(np.array(territories) > 0) for territories in player_territories)

    return player_won

#return a list of indices according to the game
def AdjacentIndices(index,adjacency_dict):
    if index >= TERRITORIES:
        return None
    else:
        return adjacency_dict[index]

#enrich features
def enrich_features(state, adjacency_matrix):
        state = normalize_state(state)
        player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(state)
        new_state = []
        for i in range(TERRITORIES):
            troop_counts = np.array(player_territories)[:, i]
            owner = np.nonzero(troop_counts)[0][0]

            normalized_troop_count = troop_counts[owner]
            is_owner = 1 if current_player_index == owner else 0
            owner_one_hot = np.eye(PLAYERS)[owner]
            last_selected = 1 if i == current_last_selected_index else 0
            one_hot_phase = np.eye(PHASES)[current_phase-1]

            territory_features = [
            normalized_troop_count,
            is_owner,
            last_selected,
            *owner_one_hot,
            *one_hot_phase
            ]
            new_state.append(territory_features)

        new_state = np.array(new_state).flatten()

        # Flatten and append the adjacency matrix
        flattened_adjacency = adjacency_matrix.flatten()
        enriched = np.concatenate([new_state, flattened_adjacency])
        return enriched

def build_adjacency_matrix(adjacency_dict):
    T = len(adjacency_dict)
    adjacency_matrix = np.zeros((T, T), dtype=int)

    for i, neighbors in adjacency_dict.items():
        for j in neighbors:
            adjacency_matrix[i][j] = 1
    return adjacency_matrix

#normalize the troops
def normalize_state(state):
    n = TERRITORIES*PLAYERS
    max_value = max(state[:n]) if max(state[:n]) > 0 else 1
    return [0 if t == 0 else max(0.1, round(t / max_value, 1)) if i < n else t for i, t in enumerate(state)]

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def handleAttack(attacking_troops,defending_troops):
     # do rolls
        while True:
            if attacking_troops >= 3:
                diceA = 3
            elif attacking_troops == 2:
                diceA = 2
            elif attacking_troops == 1:
                diceA = 1
            else:
                #attacker lost
                return attacking_troops,defending_troops

            if defending_troops >= 2:
                diceB = 2
            elif defending_troops == 1:
                diceB = 1
            else:
                # defender lost, move attacker troops into territory
                return attacking_troops,defending_troops
            
            # roll for combat
            troopDiffA, troopDiffB = computeAttack(diceA, diceB)
            attacking_troops -= troopDiffA
            defending_troops -= troopDiffB

# Roll dice and figure out troop losses
def computeAttack(diceA, diceB):
    rollsA, rollsB = getRolls(diceA, diceB)
    return doRolls(rollsA, rollsB)

# Returns rolled dice in a sorted array
def getRolls(diceA, diceB):
    pArolls = []
    pBrolls = []

    for r in range(diceA):
        pArolls.append(random.randint(1,6))
    for r in range(diceB):
        pBrolls.append(random.randint(1,6))
    
    pArolls = list(reversed(sorted(pArolls)))
    pBrolls = list(reversed(sorted(pBrolls)))

    return pArolls, pBrolls

# Computes troop losses
def doRolls(rollsA, rollsB):
    troopDiffA = 0
    troopDiffB = 0

    for i,dice in enumerate(rollsB):
        if i <= len(rollsA)-1:
            if dice >= rollsA[i]:
                troopDiffA += 1
            else:
                troopDiffB += 1
    
    return troopDiffA, troopDiffB