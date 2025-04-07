import gymnasium as gym
from gymnasium import spaces
from Game.game import Game
import numpy as np 
from Game.display import RiskDisplay
from Game.config import print_obs


class RiskEnv(gym.Env):
    def __init__(self,max_steps=1000,player_debug_mode=False,board_debug_mode=False,game_debug_mode=False):
        #call super
        super().__init__()

        #create the actual risk environment
        self.game = Game(player_debug_mode,board_debug_mode,game_debug_mode)
        self.game.start()

        #define action space: number of territories + 1
        action_length = len(self.game.board.board_dict.keys()) + 1
        self.action_space = spaces.Box(low=0,high=1,shape=(action_length,),dtype=np.float32)

        #define observation space: grid of pxn (number of players x number of territories) + one hot phase, + one hot player, + last
        #territory selected
        observation_length = len(self.game.player_list) * (action_length - 1) + self.game.total_num_phases + len(self.game.player_list) + (action_length - 1)
        self.observation_space = spaces.Box(low=0,high=1,shape=(observation_length,),dtype=np.float32)

        #create the display object
        self.display = RiskDisplay(500)

        #set max steps
        self.max_steps = max_steps

        #set the current observation for rendering
        self.current_obs_render = None

    def reset(self, seed=None, options=None):
        #set seed if not provided
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        #call reset
        self.game.newGame()

        #set the perspective player
        self.perspective_player_index = self.game.perspective_player_index

        #start step count
        self.total_steps = 0

        #return observation
        obs = self.game.board.board_state

        #set observation and action for render
        self.current_obs_render = self.game.board.get_board_state_render(self.game.currentPlayer,self.game.num_players,self.game.currentPhase,self.game.total_num_phases,self.game.last_territory_selected_index)
        self.current_action = None

        return obs, {}
    
    def step(self, action):
        #track truncation
        self.total_steps += 1

        #convert the action into movement
        obs, reward, done, info = self.game.playersPlay(action)

        #set observation for render
        self.current_obs_render = self.game.board.get_board_state_render(self.game.currentPlayer,self.game.num_players,self.game.currentPhase,self.game.total_num_phases,self.game.last_territory_selected_index)
        self.current_action = action

        #return all parameters
        return obs,reward,done,(self.total_steps >= self.max_steps), {"Current Player": info["Current Player"],"Winner": info["Winner"]}

    def render(self,render_mode='Text'):
        if render_mode == 'Visual':
            self.display.draw(self.game.board)
        else:
            print_obs(self.current_obs_render,self.current_action)
            

    def close(self):
       self.display.close()
