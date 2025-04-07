from Game.territory import Territory
from Game.player import Player
from Game.config import *
import numpy as np

class Board:
    def __init__(self,debug_mode=False):
        self.board_dict = {}        # Contains all territories
                                    # { <key>: <Territory()>, ...}
                                            
        self.board_state = []   # Observation space
                                    # Rows correspond to players
                                    # Rows contain all owned troops on territories
                                    #     t0 t1 t2 ...
                                    # p1 [ x, x, x, ...]
                                    # p2 [...]
        self.debug_mode = debug_mode
        self.maxTroopsOnTerr = MAXTROOPS

        ###Define the Connections###
        # NORTH AMERICA
        self.board_dict['alaska'] = Territory('Alaska', [1,1])
        self.board_dict['nwt'] = Territory('North West Territory', [1,5])
        self.board_dict['alberta'] = Territory('Alberta', [4,3])
        self.board_dict['ontario'] = Territory('Ontario', [4,6])
        self.board_dict['quebec'] = Territory('Quebec', [4,8])
        self.board_dict['wus'] = Territory('Western United States', [7,4])
        self.board_dict['eus'] = Territory('Eastern United States', [7,7])
        self.board_dict['greenland'] = Territory('Greenland', [1,10])
        self.board_dict['ca'] = Territory('Central America', [9,4])
        self.connections('alaska', 'nwt')
        self.connections('alaska', 'alberta')
        self.connections('nwt', 'greenland')
        self.connections('nwt', 'alberta')
        self.connections('nwt', 'ontario')
        self.connections('alberta', 'ontario')
        self.connections('alberta', 'wus')
        self.connections('ontario', 'quebec')
        self.connections('ontario', 'greenland')
        self.connections('quebec', 'greenland')
        self.connections('quebec', 'eus')
        self.connections('wus', 'eus')
        self.connections('wus', 'ca')
        self.connections('eus', 'ca')

    #Initialize the board state variable
    def initialize_board_state(self,num_players,num_phases):
        territory_matrix = np.array([[0 for _ in range(len(self.board_dict.keys()))] for p in range(num_players)]).reshape(-1)
        one_hot_players = np.array([0 for _ in range(num_players)])
        one_hot_phase = np.array([0 for _ in range(num_phases)])
        last_territory_selected = np.array([0 for _ in range(len(self.board_dict.keys()))])

        self.board_state = np.concatenate([territory_matrix,one_hot_players, one_hot_phase, last_territory_selected])
    
    #Update the board state variable. Normalize troop num by dividing by the largest one,phase is (1,2,3,4,5)
    def update_board_state(self,player_index,num_players,phase,num_phases,last_territory_selected_index):
        max_troops_board = np.max(np.array([self.board_dict[k].troops for k in self.board_dict.keys()]))
        territory_matrix = np.array([[self.board_dict[k].troops if self.board_dict[k].player_index == p else 0 for k in self.board_dict.keys()] for p in range(num_players)]).reshape(-1)
        one_hot_players = np.array([1 if i == player_index else 0 for i in range(num_players)])
        one_hot_phase = np.array([1 if i == (phase-1) else 0 for i in range(num_phases)])
        last_territory_selected = np.array([1 if i == last_territory_selected_index else 0 for i in range(len(self.board_dict.keys()))])
        self.board_state = np.concatenate([territory_matrix,one_hot_players, one_hot_phase, last_territory_selected])
    
    #Get the board state variable with placed troops. Normalize troop num by dividing by the largest one,phase is (1,2,3,4,5)
    def get_board_state_render(self,player_index,num_players,phase,num_phases,last_territory_selected_index):
        territory_matrix = np.array([[self.board_dict[k].troops+self.board_dict[k].troops_to_add if self.board_dict[k].player_index == p else 0 for k in self.board_dict.keys()] for p in range(num_players)]).reshape(-1)
        one_hot_players = np.array([1 if i == player_index else 0 for i in range(num_players)])
        one_hot_phase = np.array([1 if i == (phase-1) else 0 for i in range(num_phases)])
        last_territory_selected = np.array([1 if i == last_territory_selected_index else 0 for i in range(len(self.board_dict.keys()))])
        return np.concatenate([territory_matrix,one_hot_players, one_hot_phase, last_territory_selected])
  
    # Adding an amount of troops to a territory
    def addTroops(self, terrkey, num, player: Player):
        if num != 0:
            self.debug_mode and print(f'Adding {num} troops at {terrkey}')
            terr, tindex = self.getTerritory(terrkey)
            terr.troops += num
            terr.player_index = player.index
            if terr.troops > self.maxTroopsOnTerr:
                self.maxTroopsOnTerr = terr.troops

    # Set the amount of troops to be added at the end of the turn
    def setTroopsAvailable(self,terrkey,num):
        terr, tindex = self.getTerritory(terrkey)
        terr.troops_to_add += num

    #Add the available troops at the end of the turn
    def addAvailableTroops(self,terrkey):
        terr, tindex = self.getTerritory(terrkey)
        terr.troops += terr.troops_to_add
        if terr.troops > self.maxTroopsOnTerr:
            self.maxTroopsOnTerr = terr.troops
        terr.troops_to_add = 0
        
        
    # Setting the amount of troops on a territory to a fixed number
    def setTroops(self, terrkey, num, player: Player):
        self.debug_mode and print(f'Setting {num} troops at {terrkey}')
        terr, tindex = self.getTerritory(terrkey)
        terr.troops = num
        terr.player_index = player.index
    
    # Removing an amount of troops from a territory
    def removeTroops(self, terrkey, num, player: Player):
        if num != 0:
            self.debug_mode and print(f'Removing {num} troops from {terrkey}')
            terr, tindex = self.getTerritory(terrkey)
            terr.troops -= num

    # Check if fortification
    #  Includes real territories
    #  Is owned by the same player
    #  Are adjecent territories
    def fortificationIsValid(self, terrkeyIn, terrkeyOut, player_index):
        terrIn, tindex = self.getTerritory(terrkeyIn)
        terrOut, tindex = self.getTerritory(terrkeyOut)

        if terrIn.name == '???' or terrOut.name == '???':
            print('ERROR: Fortify failed, territories are not real')
            return False
        
        # make sure player owns both territories
        if terrIn.player_index != player_index or terrOut.player_index != player_index:
            print(' Invalid fortify: owner relationship invalid')
            return False

        return self.adjacencyIsValid(terrkeyIn, terrkeyOut)

    # Returns if 2 territories are adjecent
    def adjacencyIsValid(self, terrkeyA, terrkeyB):
        terrA, tindex = self.getTerritory(terrkeyA)
        terrB, tindex = self.getTerritory(terrkeyB)

        if terrA.name == '???' or terrB.name == '???':
            print('ERROR: Attack failed, territories are not real')
            return False
        
        if terrB in terrA.adjecency_list:
            return True
        
        self.debug_mode and print('territories are not adjecent')
        return False

    # Checks if attack
    #  Is between real territories
    #  Player does not own the attacked territory
    #  Player owns the territory they attack from
    #  Player has more than 1 troop to attack with
    def attackIsValid(self, terrkeyAttack, terrkeyFrom, player_index):        
        terrAttack, tindex = self.getTerritory(terrkeyAttack)
        terrFrom, tindex = self.getTerritory(terrkeyFrom)

        if terrAttack.name == '???' or terrFrom.name == '???':
            print('ERROR: Attack failed, territories are not real')
            return False
        
        # invalid if the player does not own the owned territory
        # or if the attacking territory is owned by that player
        if terrFrom.player_index != player_index or terrAttack.player_index == player_index:
            print(' Invalid attack, owner relationship invalid')
            return False
        
        # must have enough troops
        if terrFrom.troops <= 1:
            print(' Invalid attack, player does not have enough troops')
            return False

        return self.adjacencyIsValid(terrkeyAttack, terrkeyFrom)

    # Pass a key and get the corresponding territory and the index in the dictionary
    #  The index should correspond to the observation matrix index
    def getTerritory(self, terrkey):
        if terrkey in self.board_dict:
            keys = list(self.board_dict.keys())
            return self.board_dict[terrkey], keys.index(terrkey)
        else:
            print(f'ERROR: Wrong key -> {terrkey}')
            return Territory('???', [0,0]), -1

    # Adds a connection the both territories
    def connections(self, terra, terrb):
        terrA, tindex = self.getTerritory(terra)
        terrB, tindex = self.getTerritory(terrb)

        if not terrA.isConnected(terrB):
            terrA.connectto(terrB)
        
        if not terrB.isConnected(terrA):
            terrB.connectto(terrA)

    def reset(self):
        for name,t in self.board_dict.items():
            t.reset()
    def __repr__(self):
        rep = ""
        for k in self.board_dict.keys():
            rep += str(self.board_dict[k]) + " "
        return rep