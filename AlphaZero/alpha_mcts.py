from Game.game import Game
from Game.player import Player
from Game.config import *
import math
import torch
import numpy as np

#class for AlphaZero Monte Carlo Tree Search
#Input the root node
#output action distribution
class AlphaMCTS:
    def __init__(self,root_state,model,num_simuations=100,tau=1):
        #create the root node with the root state, no parent, and probability 1
        self.root = Node(root_state,None,1)
        self.model = model
        self.num_simulations = num_simuations
        self.tau = tau
    def search(self):
        for _ in range(self.num_simulations):
            #search for best leaf node
            node = self.select(self.root)
            #expand the node if it is not terminal
            if not isTerminalState(node.state):
                #expand the node
                self.expand(node)
                #get the vale from this node
                value = self.simulate(node)
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
        #valid actions, should be a list of ints corresponding to the valid actions
        valid_actions = getValidActionsFromState(node.state)
        
        #next states
        next_states = [getNextState(node.state,action) for action in valid_actions]
        #probabilties
        value, probabilities = self.model.sample_action(torch.tensor(node.state,dtype=torch.float32))
        probabilities = probabilities.detach().numpy()
        valid_probabilities = [probabilities[action] for action in valid_actions]
        #expand the node
        node.expand(valid_actions,next_states,valid_probabilities)
    def simulate(self,node):
        value, _ = self.model.sample_action(torch.tensor(node.state,dtype=torch.float32))
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
        #get the total visit count per child
        visits = np.array([child.N for child in self.root.children.values()])
        #get the list of all the action names
        actions = list(self.root.children.keys())
        #adjust the probabilities based on tau
        probs = visits**(1 / self.tau)
        #normalize probabilities
        probs = probs / np.sum(probs)
        #return action probabilities
        sample_action = np.zeros(self.model.output_action_size)
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
        
# Passing in a Game, State, and Action, return the next State
# The game should be of type game, the state should be a 1d array, and the action should be of type int
# Next State will also be a 1d array
def getNextState(state, action):
    #create an empty game
    game = Game()
    game.start()

    #extract info from state
    player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(state)

    #create the players
    game.player_list = [Player(list(game.board.board_dict.keys()),p) for p in range(game.num_players)]

    #fill board, call set troops on all territories, add the territories to the players
    for index,key in enumerate(game.board.board_dict.keys()):
        num_troops, player_index = [(p_t[index],p_i) for p_i,p_t in enumerate(player_territories) if p_t[index] != 0][0]
        game.board.setTroops(key, num_troops, game.player_list[player_index])
        game.player_list[player_index].gainATerritory(key)
    
    #set the game turn and phase
    game.currentPlayer = current_player_index
    game.currentPhase = current_phase

    #set the last index of the current player
    if current_last_selected_index != -1:
        game.player_list[current_player_index].from_terr_sel = game.player_list[current_player_index].terr_list[current_last_selected_index]
    
    #update the initial board state
    game.board.update_board_state(current_player_index,game.num_players,current_phase,game.total_num_phases, current_last_selected_index)

    #one hot action
    action = np.eye(len(game.board.board_dict.keys()) + 1)[action]
    #carry out action
    game.playersPlay(action)

    #return the state
    return game.board.board_state

def getValidActionsFromState(state):
    #create an empty game
    game = Game()
    game.start()

    #extract info from state
    player_territories,current_player_index,current_phase,current_last_selected_index = getStateInfo(state)

    #create the players
    game.player_list = [Player(list(game.board.board_dict.keys()),p) for p in range(game.num_players)]

    #fill board, call set troops on all territories, add the territories to the players
    for index,key in enumerate(game.board.board_dict.keys()):
        num_troops, player_index = [(p_t[index],p_i) for p_i,p_t in enumerate(player_territories) if p_t[index] != 0][0]
        game.board.setTroops(key, num_troops, game.player_list[player_index])
        game.player_list[player_index].gainATerritory(key)
    
    #set the game turn and phase
    game.currentPlayer = current_player_index
    game.currentPhase = current_phase

    #set the last index of the current player
    if current_last_selected_index != -1:
        game.player_list[current_player_index].from_terr_sel = game.player_list[current_player_index].terr_list[current_last_selected_index]
    
    #update the initial board state
    game.board.update_board_state(current_player_index,game.num_players,current_phase,game.total_num_phases, current_last_selected_index)
    # get list of all actions from current player
    valid_actions = game.player_list[current_player_index].getValidActions(game.board,game.currentPhase)
    return valid_actions

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

def isTerminalState(state):

    #extract info from state
    player_territories,_,_,_ = getStateInfo(state)

    #check if one player has all the territories
    player_won = any(np.all(np.array(territories) > 0) for territories in player_territories)

    return player_won
    
