import random
import numpy as np
# PLAYER CLASS
#  Default player class
#  Picks random territories for every action
class Player:
    def __init__(self, terr_list, index,debug_mode=False):

        self.terr_list = terr_list
        self.amountOfOwned = 0          # Amount of territories owned
        self.myOwnedTerritories = []    # Keys of all owned territories
        self.index = index              # Index of the player (for board matrix)
        self.game_state = None          #current state of the game

        self.from_terr_sel = None       #variable to keep track of from territory selected during attack/fortify
        self.to_terr_sel = None         #variable to keep track of from territory selected during attack/fortify
        self.debug_mode = debug_mode        #toggle print statements
    
    # When restarting a new game, clear the player
    def clearPlayer(self):
        self.amountOfOwned = 0
        self.myOwnedTerritories = []

    # Player gains a territory
    def gainATerritory(self, terrkey):
        self.myOwnedTerritories.append(terrkey)
        self.amountOfOwned += 1

    # Player loses a territory
    def loseATerritory(self, terrkey):
        self.myOwnedTerritories.remove(terrkey)
        self.amountOfOwned -= 1
    
    #all actions are of length of the number of territories + none action
    # GAME ACTION Phase 1
    # Asks the player where to place troops
    # Returns if the placing was successful and the reward the player should get
    def place_troops(self, board_obj,action):
        available = self.amountOfOwned
        #find the valid territories to place troops on
        valid_terr = [index for index,terr in enumerate(self.terr_list) if terr in self.myOwnedTerritories]
        #set invalid actions to 0 probability
        valid_actions = [a if index in valid_terr else 0 for index,a in enumerate(action)]
        #normalize the valid actions
        valid_actions = valid_actions / np.sum(valid_actions)
        #take sample action
        sampled_action = np.random.choice(len(valid_actions), p=valid_actions)
        terrkey = self.terr_list[sampled_action]
        #place troops
        if terrkey in self.myOwnedTerritories:
            board_obj.setTroopsAvailable(terrkey,available)
            self.debug_mode and print(f'Phase 1: Player {self.index} Placed {available} troops at {terrkey}')
            return True,available
        return False,0
    
    def add_troops(self,board_obj):
        for key in board_obj.board_dict.keys():
            board_obj.addAvailableTroops(key)
    
    # GAME ACTION Phase 2
    #  Asks the player where to attack from, None represents no attack. 
    def attack_from(self,board_obj,action):
        #Create a mask for valid attacks. Territory should be owned by the player,
        #have at least more than one troop, and have adjacency with at least one territory that the player doesn't own
        #this preliminary check makes sure the territories are owned and more than one troop 
        possible_terr = [terr for terr in self.terr_list if terr in self.myOwnedTerritories and (board_obj.board_dict[terr].troops) > 1]
        #iterate through the possible territories making sure each has adjacency
        #with at least one territory that the player doesn't own
        valid_terr = [self.terr_list.index(terr) for terr in possible_terr if any(board_obj.adjacencyIsValid(terr,t) for t in self.terr_list if t not in self.myOwnedTerritories)]

        #set invalid actions to 0 probability
        valid_actions = [a if index in valid_terr or index==len(action) - 1 else 0 for index,a in enumerate(action)]
        
        #normalize the valid actions
        valid_actions = valid_actions / np.sum(valid_actions)

        #take sample action
        sampled_action = np.random.choice(len(valid_actions), p=valid_actions)
        self.from_terr_sel = self.terr_list[sampled_action] if sampled_action < len(self.terr_list) else None

        if self.from_terr_sel != None:
            self.debug_mode and print(f"Phase 2: Player {self.index} attacks from {self.from_terr_sel}")
            return True, self.terr_list.index(self.from_terr_sel)
        else:
            self.debug_mode and print(f"Phase 2: Player {self.index} Player does not Attack.")
            return False, -1
    
    # GAME ACTION Phase 3
    #  Asks the player where to attack to.
    def attack_to(self,board_obj,action):

        valid_terr = [self.terr_list.index(terr) for terr in self.terr_list if board_obj.adjacencyIsValid(self.from_terr_sel,terr) and terr not in self.myOwnedTerritories]

        #set invalid actions to 0 probability
        valid_actions = [a if index in valid_terr else 0 for index,a in enumerate(action)]

        #normalize the valid actions
        valid_actions = valid_actions / np.sum(valid_actions)
        
        #take sample action
        sampled_action = np.random.choice(len(valid_actions), p=valid_actions)
        self.to_terr_sel = self.terr_list[sampled_action]

        self.debug_mode and print(f"Phase 3: Player {self.index} attacks to {self.to_terr_sel}")

    # GAME ACTION Phase 4
    #  Asks the player where to fortify from
    def fortify_from(self,board_obj,action):
        #this preliminary check makes sure the territories are owned and more than one troop 
        possible_terr = [terr for terr in self.terr_list if terr in self.myOwnedTerritories and board_obj.board_dict[terr].troops > 1]
        
        #iterate through the possible territories making sure each has adjacency
        #with at least one territory that the player owns
        valid_terr = [self.terr_list.index(terr) for terr in possible_terr if any(board_obj.adjacencyIsValid(terr,t) for t in self.terr_list if t in self.myOwnedTerritories)]

        #set invalid actions to 0 probability
        valid_actions = [a if index in valid_terr or index==len(action) - 1 else 0 for index,a in enumerate(action)]
        #normalize the valid actions
        valid_actions = valid_actions / np.sum(valid_actions)
        #take sample action
        sampled_action = np.random.choice(len(valid_actions), p=valid_actions)
        self.from_terr_sel = self.terr_list[sampled_action] if sampled_action < len(self.terr_list) else None
        #Select actions randomly plus the action to not fortify
        if self.from_terr_sel != None:
            self.debug_mode and print(f"Phase 4: Player {self.index} fortifies from {self.from_terr_sel}")
            return True, self.terr_list.index(self.from_terr_sel)
        else:
            self.debug_mode and print(f"Phase 4: Player {self.index} does not Fortify.")
            return False, -1
    
    # GAME ACTION Phase 5
    #  Asks the player where to fortify to
    def fortify_to(self,board_obj,action):
        valid_terr = [self.terr_list.index(terr) for terr in self.terr_list if board_obj.adjacencyIsValid(self.from_terr_sel,terr) and terr in self.myOwnedTerritories]
        #set invalid actions to 0 probability
        valid_actions = [a if index in valid_terr or index==len(action - 1) else 0 for index,a in enumerate(action)]
        #normalize the valid actions
        valid_actions = valid_actions / np.sum(valid_actions)
        #take sample action
        sampled_action = np.random.choice(len(valid_actions), p=valid_actions)
        self.to_terr_sel = self.terr_list[sampled_action]
        self.debug_mode and print(f"Phase 5: Player {self.index} fortifies to {self.to_terr_sel}")
    
    # Get all valid actions depending on the phase of the game, setting all probabilities equal to 1
    def getValidActions(self,board_obj,phase):
        #create sample action equal to the length of all the territories + 1 for doing nothing
        sample_action = np.ones(len(self.terr_list) + 1)
        #create match for each phase
        match (phase):
            case 1:
                #find the valid territories to place troops on
                valid_terr = [index for index,terr in enumerate(self.terr_list) if terr in self.myOwnedTerritories]
                #set invalid actions to 0 probability
                valid_actions = [a if index in valid_terr else 0 for index,a in enumerate(sample_action)]
                #turn valid actions into indices [0,1,1,0,0] into [1,2]
                valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
                return valid_action_indices
            case 2:
                #Create a mask for valid attacks. Territory should be owned by the player,
                #have at least more than one troop, and have adjacency with at least one territory that the player doesn't own
                #this preliminary check makes sure the territories are owned and more than one troop 
                possible_terr = [terr for terr in self.terr_list if terr in self.myOwnedTerritories and board_obj.board_dict[terr].troops > 1]
                #iterate through the possible territories making sure each has adjacency
                #with at least one territory that the player doesn't own
                valid_terr = [self.terr_list.index(terr) for terr in possible_terr if any(board_obj.adjacencyIsValid(terr,t) for t in self.terr_list if t not in self.myOwnedTerritories)]
                #set invalid actions to 0 probability
                valid_actions = [a if index in valid_terr or index==len(sample_action) - 1 else 0 for index,a in enumerate(sample_action)]
                #turn valid actions into indices 
                valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
                return valid_action_indices
            case 3:
                valid_terr = [self.terr_list.index(terr) for terr in self.terr_list if board_obj.adjacencyIsValid(self.from_terr_sel,terr) and terr not in self.myOwnedTerritories]
                #set invalid actions to 0 probability
                valid_actions = [a if index in valid_terr else 0 for index,a in enumerate(sample_action)]
                #turn valid actions into indices 
                valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
                return valid_action_indices
            case 4:
                #this preliminary check makes sure the territories are owned and more than one troop 
                possible_terr = [terr for terr in self.terr_list if terr in self.myOwnedTerritories and board_obj.board_dict[terr].troops > 1]
                
                #iterate through the possible territories making sure each has adjacency
                #with at least one territory that the player owns
                valid_terr = [self.terr_list.index(terr) for terr in possible_terr if any(board_obj.adjacencyIsValid(terr,t) for t in self.terr_list if t in self.myOwnedTerritories)]

                #set invalid actions to 0 probability
                valid_actions = [a if index in valid_terr or index==len(sample_action) - 1 else 0 for index,a in enumerate(sample_action)]
                #turn valid actions into indices 
                valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
                return valid_action_indices
            case 5:
                valid_terr = [self.terr_list.index(terr) for terr in self.terr_list if board_obj.adjacencyIsValid(self.from_terr_sel,terr) and terr in self.myOwnedTerritories]
                #set invalid actions to 0 probability
                valid_actions = [a if index in valid_terr or index==len(sample_action - 1) else 0 for index,a in enumerate(sample_action)]
                #turn valid actions into indices 
                valid_action_indices = [index for index,action in enumerate(valid_actions) if action == 1]
                return valid_action_indices
            case _:
                print("Invalid Phase")
                return None
    #attack
    def attack(self):
        result = self.to_terr_sel,self.from_terr_sel

        #reset tracking variables
        self.from_terr_sel = None
        self.to_terr_sel = None
        return result
    
    #fortify
    def fortify(self, board_obj):
        terrIn = self.to_terr_sel
        terrOut = self.from_terr_sel
        valid_fortification = board_obj.fortificationIsValid(terrIn, terrOut, self.index)
        if valid_fortification:
            theTerr, tindex = board_obj.getTerritory(terrOut)
            troops = theTerr.troops - 1
            if troops > 0:
                board_obj.addTroops(terrIn, troops, self)
                board_obj.removeTroops(terrOut, troops, self)

        #reset tracking variables
        self.from_terr_sel = None
        self.to_terr_sel = None