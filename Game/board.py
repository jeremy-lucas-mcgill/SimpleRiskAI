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

        #DEFINE CONTINENT LISTS
        north_america = ['alaska','nwt','alberta','ontario','quebec','wus','eus','greenland','ca']
        south_america = ['venezuela','peru','brazil','argentina']
        europe = ['iceland','scand','ukraine','gb','neur','weur','seur']
        africa = ['nafr','egypt','eafr','congo','safr','mad']
        asia = ['ural','siberia','yakutsk','kam','irkutsk','mong','japan','china','afgh','me','india','siam']
        australia = ['indo','ng','waus','eaus']

        #Continent Dictionary (ACTIVE,NUMBER OF TERRITORIES, LIST, TROOPS TO BE GAINED)
        self.continent_dict = {
            'NORTH AMERICA':(NORTH_AMERICA,9,north_america,10),
            'SOUTH AMERICA':(SOUTH_AMERICA,4,south_america,8),
            'EUROPE':(EUROPE,7,europe,10),
            'AFRICA':(AFRICA,6,africa,6),
            'ASIA':(ASIA,12,asia,14),
            'AUSTRALIA':(AUSTRALIA,4,australia,4)
        }

        ###Define the Connections###
        # NORTH AMERICA
        if NORTH_AMERICA:
            self.board_dict['alaska'] = Territory('Alaska', [45,95])
            self.board_dict['nwt'] = Territory('North West Territory', [110,75])
            self.board_dict['alberta'] = Territory('Alberta', [105,130])
            self.board_dict['ontario'] = Territory('Ontario', [160,130])
            self.board_dict['quebec'] = Territory('Quebec', [225,125])
            self.board_dict['wus'] = Territory('Western United States', [105,190])
            self.board_dict['eus'] = Territory('Eastern United States', [160,195])
            self.board_dict['greenland'] = Territory('Greenland', [215,45])
            self.board_dict['ca'] = Territory('Central America', [120,265])
        
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
            self.connections('eus','ontario')

        #SOUTH AMERICA
        if SOUTH_AMERICA:
            self.board_dict['venezuela'] = Territory('Venezuela', [165,300])
            self.board_dict['peru'] = Territory('Peru', [180,370])
            self.board_dict['brazil'] = Territory('Brazil', [225,330])
            self.board_dict['argentina'] = Territory('Argentina', [185,460])

            self.connections('venezuela','peru')
            self.connections('venezuela','brazil')
            self.connections('peru','brazil')
            self.connections('argentina','peru')
            self.connections('argentina','brazil')

        #NORTH AMERICA TO SOUTH AMERICA
        if NORTH_AMERICA and SOUTH_AMERICA:
            self.connections('ca','venezuela')

        #EUROPE
        if EUROPE:
            self.board_dict['iceland'] = Territory('Iceland', [330,75])
            self.board_dict['scand'] = Territory('Scandinavia', [395,70])
            self.board_dict['ukraine'] = Territory('Ukraine', [460,140])
            self.board_dict['gb'] = Territory('Great Britain', [330,135])
            self.board_dict['neur'] = Territory('Northern Europe', [390,150])
            self.board_dict['weur'] = Territory('Western Europe', [340,190])
            self.board_dict['seur'] = Territory('Southern Europe', [400,190])

            self.connections('iceland','scand')
            self.connections('iceland', 'gb')
            self.connections('scand','gb')
            self.connections('gb','weur')
            self.connections('weur','seur')
            self.connections('seur','neur')
            self.connections('weur','neur')
            self.connections('gb','neur')
            self.connections('scand','neur')
            self.connections('scand','ukraine')
            self.connections('neur','ukraine')
            self.connections('seur','ukraine')

        #NORTH AMERICA TO EUROPE
        if NORTH_AMERICA and EUROPE:
            self.connections('greenland','iceland')

        #AFRICA
        if AFRICA:
            self.board_dict['nafr'] = Territory('North Africa',[335,270])
            self.board_dict['egypt'] = Territory('Egypt',[400,250])
            self.board_dict['eafr'] = Territory('East Africa', [420,300])
            self.board_dict['congo'] = Territory('Congo', [370,335])
            self.board_dict['safr'] = Territory('South Africa', [390,430])
            self.board_dict['mad'] = Territory('Madagascar', [455,410])

            self.connections('nafr','egypt')
            self.connections('nafr','eafr')
            self.connections('nafr','congo')
            self.connections('egypt','eafr')
            self.connections('eafr','congo')
            self.connections('eafr','safr')
            self.connections('eafr','mad')
            self.connections('mad','safr')
            self.connections('safr','congo')

        #SOUTH AMERICA AND EUROPE TO AFRICA
        if AFRICA and SOUTH_AMERICA:
            self.connections('brazil','nafr')
        if AFRICA and EUROPE:
            self.connections('seur','nafr')
            self.connections('seur','egypt')

        #ASIA
        if ASIA:
            self.board_dict['ural'] = Territory('Ural', [510, 155])
            self.board_dict['siberia'] = Territory('Siberia', [555, 95])
            self.board_dict['yakutsk'] = Territory('Yakutsk', [620, 75])
            self.board_dict['kam'] = Territory('Kamchatka', [700, 105])
            self.board_dict['irkutsk'] = Territory('Irkutsk', [620, 135])
            self.board_dict['mong'] = Territory('Mongolia', [600, 175])
            self.board_dict['japan'] = Territory('Japan', [680, 185])
            self.board_dict['china'] = Territory('China', [570, 225])
            self.board_dict['afgh'] = Territory('Afghanistan', [490, 190])
            self.board_dict['me'] = Territory('Middle East', [460, 245])
            self.board_dict['india'] = Territory('India', [535, 260])
            self.board_dict['siam'] = Territory('Siam', [580, 275])

            self.connections('me','afgh')
            self.connections('me', 'india')
            self.connections('afgh','india')
            self.connections('afgh','ural')
            self.connections('afgh','china')
            self.connections('india','siam')
            self.connections('india','china')
            self.connections('siam','china')
            self.connections('china','ural')
            self.connections('china','siberia')
            self.connections('china','mong')
            self.connections('ural','siberia')
            self.connections('siberia','yakutsk')
            self.connections('siberia','irkutsk')
            self.connections('siberia','mong')
            self.connections('mong','irkutsk')
            self.connections('mong','japan')
            self.connections('mong', 'kam')
            self.connections('irkutsk','yakutsk')
            self.connections('irkutsk','kam')
            self.connections('yakutsk','kam')
            self.connections('japan','kam')

        # AFRICA EUROPE AND NORTH AMERICA TO ASIA
        if ASIA and AFRICA:
            self.connections('egypt','me')
            self.connections('eafr','me')
        if ASIA and EUROPE:
            self.connections('seur','me')
            self.connections('ukraine','me')
            self.connections('ukraine','afgh')
            self.connections('ukraine','ural')
        if ASIA and NORTH_AMERICA:
            self.connections('alaska','kam')

        # AUSTRALIA
        if AUSTRALIA:
            self.board_dict['indo'] = Territory('Indonesia', [570, 340])
            self.board_dict['ng'] = Territory('New Guinea', [650, 350])
            self.board_dict['waus'] = Territory('Western Australia', [605, 430])
            self.board_dict['eaus'] = Territory('Eastern Australia', [665, 430])

            self.connections('indo','ng')
            self.connections('indo','waus')
            self.connections('ng','waus')
            self.connections('ng','eaus')
            self.connections('waus','eaus')

        # ASIA TO AUSTRALIA
        if ASIA and AUSTRALIA:
            self.connections('indo', 'siam')

    #Initialize the board state variable
    def initialize_board_state(self,num_players,num_phases):
        territory_matrix = np.array([[0 for _ in range(len(self.board_dict.keys()))] for p in range(num_players)]).reshape(-1)
        one_hot_players = np.array([0 for _ in range(num_players)])
        one_hot_phase = np.array([0 for _ in range(num_phases)])
        last_territory_selected = np.array([0 for _ in range(len(self.board_dict.keys()))])

        self.board_state = np.concatenate([territory_matrix,one_hot_players, one_hot_phase, last_territory_selected])
    
    #Update the board state variable. Normalize troop num by dividing by the largest one,phase is (1,2,3,4,5)
    def update_board_state(self,player_index,num_players,phase,num_phases,last_territory_selected_index):
        territory_matrix = np.array([[self.board_dict[k].troops if self.board_dict[k].player_index == p else 0 for k in self.board_dict.keys()] for p in range(num_players)]).reshape(-1)
        one_hot_players = np.array([1 if i == player_index else 0 for i in range(num_players)])
        one_hot_phase = np.array([1 if i == (phase-1) else 0 for i in range(num_phases)])
        last_territory_selected = np.array([1 if i == last_territory_selected_index else 0 for i in range(len(self.board_dict.keys()))])
        self.board_state = np.concatenate([territory_matrix,one_hot_players, one_hot_phase, last_territory_selected])
  
    # Adding an amount of troops to a territory
    def addTroops(self, terrkey, num, player: Player):
        if num != 0:
            self.debug_mode and print(f'Adding {num} troops at {terrkey}')
            terr, tindex = self.getTerritory(terrkey)
            terr.troops += num
            terr.player_index = player.index
            if terr.troops > self.maxTroopsOnTerr:
                self.maxTroopsOnTerr = terr.troops
        
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