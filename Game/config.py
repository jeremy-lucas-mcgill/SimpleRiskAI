####################game parameters############################
PLAYERS = 4
PHASES = 5
MAXTROOPS = 1000
TERRITORIES = 9

####################display parameters############################
import pygame


WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
COLORS = {0:RED,1:BLUE,2:YELLOW,3:GREEN}
SCREEN_WIDTH,SCREEN_HEIGHT = 800,600
GRID_SCREEN_WIDTH,GRID_SCREEN_HEIGHT = 400,300
GRID_WIDTH,GRID_HEIGHT = 10,10
GRID_SCREEN_POS_X,GRID_SCREEN_POS_Y = 50,50
RECT_THICKNESS = 1
LINE_WIDTH = 1
FPS = 60
RADIUS = 1
FONT_SIZE = 30
TIME_DELAY = 1000

# Map grid coordinates to screen coordinates
def grid_pos_to_screen_pos(grid_pos):
    x, y = grid_pos
    screen_x = (x * GRID_SCREEN_WIDTH) // GRID_WIDTH + GRID_SCREEN_POS_X
    screen_y = (y * GRID_SCREEN_HEIGHT) // GRID_HEIGHT + GRID_SCREEN_POS_Y
    return screen_x, screen_y

# Scales grid coordinates to screen coordinates
def grid_to_screen_scale(grid_value):
    screen_x = (grid_value[0] * GRID_SCREEN_WIDTH) / GRID_WIDTH
    screen_y = (grid_value[1] * GRID_SCREEN_HEIGHT) / GRID_HEIGHT
    return screen_x, screen_y

#draw the rectangle around the visualization
def draw_background(screen):
    pygame.draw.rect(screen,BLACK,(GRID_SCREEN_POS_X,GRID_SCREEN_POS_Y,GRID_SCREEN_WIDTH,GRID_SCREEN_HEIGHT),RECT_THICKNESS)

#setting the seed
import random
import numpy as np
import torch 
def set_seed(seed=42):
    random.seed(seed)          
    np.random.seed(seed)       
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

#turn the observation into something readable
def print_obs(obs,action=None):
        string = "New State:\n"
        for index in range(PLAYERS):
            string += f"Player {index} Territories:{obs[index * TERRITORIES:(index+1)*TERRITORIES]}\n"
        string += f"Turn: {obs[TERRITORIES*PLAYERS:TERRITORIES*PLAYERS+PLAYERS]}, Phase: {obs[TERRITORIES*PLAYERS+PLAYERS:TERRITORIES*PLAYERS+PLAYERS+PHASES]}, Last Index: {obs[TERRITORIES*PLAYERS+PLAYERS+PHASES:]}\n"
        string += f"Taken Action: {action}" if not action is None else ""
        print(string)