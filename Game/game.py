import random
import numpy as np
from Game.board import Board
from Game.player import Player
from Game.config import *

# GAME CLASS
#   Controls the entire game
class Game:
    def __init__(self,player_debug_mode=False,board_debug_mode=False,game_debug_mode=False):
        
        # GAME INFO
        self.currentPlayer = 0      # current player index
        self.currentPhase = 1       # current phase of a players turn
        self.total_num_phases = PHASES
        self.num_players = PLAYERS
        self.last_territory_selected_index = -1

        #PARAMETERS
        self.player_debug_mode = player_debug_mode
        self.board_debug_mode = board_debug_mode
        self.debug_mode = game_debug_mode

    # Entrance to the program
    def start(self):

        # BOARD
        self.board = Board(self.board_debug_mode)

        # PLAYER LIST
        self.player_list = [Player(list(self.board.board_dict.keys()),p,self.player_debug_mode) for p in range(self.num_players)]

        # GAME SETUP
        self.newGame()

    # Sets up a new game
    def newGame(self):
        # Reset board
        self.board.reset()
        
        # PLAYERS
        #  Clear previous data
        for p in self.player_list:
            p.clearPlayer()

        # PLACE TROOPS
        self.distributeLand()

        #reset random player turn and phase to 0
        self.currentPlayer = random.choice(range(len(self.player_list)))
        self.currentPhase = 1

        #Initialize Observation
        self.board.update_board_state(self.currentPlayer, len(self.player_list), self.currentPhase,self.total_num_phases,-1)

    def reset(self):
        # Reset board
        self.board.reset()

        # PLAYERS
        for p in self.player_list:
            p.clearPlayer()
        #Initialize Observation
        self.board.update_board_state(self.currentPlayer, len(self.player_list), self.currentPhase,self.total_num_phases,-1)

    # MAIN
    # Checks for win condition
    # triggers the end game
    def checkForWinner(self):
        active = [p for p in self.player_list if p.amountOfOwned > 0]
        if len(active) == 1:
            #print(f"Winner: {active[0].index}") 
            return True, active[0].index
        else:
            return False, None

    # Get a turn from a player, return new observation, reward, and if the game is over
    def playersPlay(self,action):
        self.debug_mode and print([self.board.board_dict[k] for k in self.board.board_dict.keys()])
        # get the current player
        currPlayer = self.player_list[self.currentPlayer]
        #reset the last territory selected index
        self.last_territory_selected_index = -1
        #reset reward
        reward = 0
        
        self.debug_mode and print(f'Curr Player {currPlayer.index} Phase {self.currentPhase}')

        # check if player can play
        if currPlayer.amountOfOwned > 0:

            # Phase 1. Place troops on the board
            if self.currentPhase == 1:
                self.debug_mode and print(f'-Player {self.player_list[self.currentPlayer].index} turn-')
                next_phase,reward = currPlayer.place_troops(self.board,action)
                if next_phase:
                    self.currentPhase = 2
            
            # Phase 2. Select Where Attack is coming from
            elif self.currentPhase == 2:
                next_phase,self.last_territory_selected_index = currPlayer.attack_from(self.board,action)
                #If next phase is False skip to foritification
                self.currentPhase = 3 if next_phase else 4
            
            # Phase 3. Select where attack is going to and perform the attack
            elif self.currentPhase == 3:
                currPlayer.attack_to(self.board,action)
                next_phase = self.handleAttack(currPlayer)
                # next phase is true on invalid attack. If the attack is valid, go back to phase 2. 
                self.currentPhase = 4 if next_phase else 2
            
            # Phase 4. Select where Fortify is coming from
            elif self.currentPhase == 4:
                next_phase,self.last_territory_selected_index = currPlayer.fortify_from(self.board,action)
                #If next phase is False skip to end turn
                self.currentPhase = 5 if next_phase else 1
                if not next_phase:
                    self.nextPlayer()
            # Phase 5. Select where Fortify is going to and perform the fortification. Whether the fortification is valid or not 
            # go to next phase
            elif self.currentPhase == 5:
                currPlayer.fortify_to(self.board,action)
                currPlayer.fortify(self.board)
                self.currentPhase = 1 
                self.nextPlayer()
        else:
            #add troops and go to next player
            self.nextPlayer()
        
        #update board state
        self.board.update_board_state(self.currentPlayer,len(self.player_list),self.currentPhase,self.total_num_phases,self.last_territory_selected_index)

        #check if there is a winner
        done, winner = self.checkForWinner()

        #return obs,reward, and done flag
        return self.board.board_state, reward, done, {"Current Player": currPlayer.index, "Winner": winner}

    # Increment whose turn it is
    def nextPlayer(self):
        #make sure the next player can play
        while self.player_list[(self.currentPlayer + 1) % len(self.player_list)].amountOfOwned == 0:
            self.currentPlayer = (self.currentPlayer + 1) % len(self.player_list)
        self.currentPlayer = (self.currentPlayer + 1) % len(self.player_list)

    # Carry out the attack of a player
    def handleAttack(self, currPlayer: Player):
        # find territory in and territory out
        terrkeyAttack, terrkeyFrom = currPlayer.attack()
        
        # validate attack, quit if invalid
        if not self.board.attackIsValid(terrkeyAttack, terrkeyFrom, currPlayer.index):
            return True
        
        # determine winners
        self.doAttack(terrkeyAttack, terrkeyFrom)
        return False

    # Game set up
    #  Set up the board matrix
    #  Players get initial random territories
    def distributeLand(self):
        # get all possible territory keys
        possible_terrs = list(self.board.board_dict.keys())
        player_ind = random.randint(0,PLAYERS - 1)
        # loop until all territories have troops
        while len(possible_terrs) > 0:
            # pick a random territory key
            terrind = random.randint(0, len(possible_terrs)-1)
            # add troops according to the current player index
            self.board.addTroops(possible_terrs[terrind], 1, self.player_list[player_ind])
            
            # give that player the territory
            self.player_list[player_ind].gainATerritory(possible_terrs[terrind])

            # remove territory from consideration
            del possible_terrs[terrind]

            player_ind += 1
            player_ind = player_ind % len(self.player_list)

    # Perform a players attack
    def doAttack(self, terrkeyAttack, terrkeyFrom):
        terrAttacker, tindex = self.board.getTerritory(terrkeyFrom)
        terrDefender, tindex = self.board.getTerritory(terrkeyAttack)

        playerA = self.player_list[terrAttacker.player_index]
        playerB = self.player_list[terrDefender.player_index]
        if playerA == None or playerB == None:
            print('ERROR: Failed to find player!')

        if terrAttacker.name == '???' or terrDefender.name == '???':
            print('ERROR: Attack failed, territories are not real')
            return None
        
        # do rolls
        while True:
            self.debug_mode and print(f'Combat throw: troops A {terrAttacker.troops} troops B {terrDefender.troops}')
            if terrAttacker.troops-1 >= 3:
                diceA = 3
            elif terrAttacker.troops-1 == 2:
                diceA = 2
            elif terrAttacker.troops-1 == 1:
                diceA = 1
            else:
                self.debug_mode and print('Result: Attacker loses')
                break
            
            if terrDefender.troops >= 2:
                diceB = 2
            elif terrDefender.troops == 1:
                diceB = 1
            else:
                # defender lost, move attacker troops into territory
                self.board.setTroops(terrkeyAttack, terrAttacker.troops - 1, playerA)
                self.board.setTroops(terrkeyFrom, 1, playerA)
                playerA.gainATerritory(terrkeyAttack)
                playerB.loseATerritory(terrkeyAttack)
                self.debug_mode and print('Result: Attacker wins')
                break

            # roll for combat
            troopDiffA, troopDiffB = self.computeAttack(diceA, diceB)
            self.board.removeTroops(terrkeyFrom, troopDiffA, playerA)
            self.board.removeTroops(terrkeyAttack, troopDiffB, playerB)
    
    # Roll dice and figure out troop losses
    def computeAttack(self, diceA, diceB):
        rollsA, rollsB = self.getRolls(diceA, diceB)
        return self.doRolls(rollsA, rollsB)
    
    # Returns rolled dice in a sorted array
    def getRolls(self, diceA, diceB):
        pArolls = []
        pBrolls = []

        for r in range(diceA):
            pArolls.append(random.randint(1,6))
        for r in range(diceB):
            pBrolls.append(random.randint(1,6))
        
        pArolls = list(reversed(sorted(pArolls)))
        pBrolls = list(reversed(sorted(pBrolls)))

        self.debug_mode and print(f' Rolls A: {pArolls}, Rolls B: {pBrolls}')
        return pArolls, pBrolls

    # Computes troop losses
    def doRolls(self, rollsA, rollsB):
        troopDiffA = 0
        troopDiffB = 0

        for i,dice in enumerate(rollsB):
            if i <= len(rollsA)-1:
                if dice >= rollsA[i]:
                    troopDiffA += 1
                else:
                    troopDiffB += 1
        
        return troopDiffA, troopDiffB