####################game parameters############################
PLAYERS = 6
PHASES = 5
MAXTROOPS = 1000
###DEFINE WHICH CONTINENTS TO USE###
NORTH_AMERICA = False #9
SOUTH_AMERICA = False #4
EUROPE = True #7
AFRICA = True #6
ASIA = True #12
AUSTRALIA = True #4

###UPDATE TERRITORY COUNT###
TERRITORIES = sum([count if active else 0 for (active, count) in [(NORTH_AMERICA,9),(SOUTH_AMERICA,4),(EUROPE,7),(AFRICA,6),(ASIA,12),(AUSTRALIA,4)]])
print(TERRITORIES)

####################display parameters############################
import pygame

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
PURPLE = (160,32,240)
ORANGE = (255,165,0)
PINK = (255,192,203)

COLORS = {0:RED,1:BLUE,2:YELLOW,3:GREEN,4:PURPLE,5:ORANGE}
SCREEN_WIDTH,SCREEN_HEIGHT = 800,600
GRID_SCREEN_WIDTH,GRID_SCREEN_HEIGHT = 500,400
GRID_WIDTH,GRID_HEIGHT = 750,500
GRID_SCREEN_POS_X,GRID_SCREEN_POS_Y = 50,50
RECT_THICKNESS = 1
LINE_WIDTH = 1
FPS = 60
RADIUS = 20
FONT_SIZE = 30
TIME_DELAY = 100

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


"""
# NORTH AMERICA 

Alaska: (45, 95)

North West Territory: (110, 75)

Greenland: (215, 45)

Alberta: (105, 130)

Ontario: (160, 130)

Quebec: (225, 125)

Western United States: (105, 190)

Eastern United States: (160, 195)

Central America: (120, 265)

# SOUTH AMERICA

Venezuela: (165, 300)

Peru: (180, 370)

Brazil: (225, 330)

Argentina: (185, 460)

# EUROPE 

Iceland: (330, 75)

Scandinavia: (395, 70)

Ukraine: (460, 140)

Great Britain: (330, 135)

Northern Europe: (390, 150)

Western Europe: (340, 190)

Southern Europe: (400, 190)

# AFRICA 

North Africa: (335, 270)

Egypt: (400, 250)

East Africa: (420, 330)

Congo: (370, 335)

South Africa: (390, 430)

Madagascar: (455, 410)

# ASIA 

Ural: (510, 155)

Siberia: (585, 115)

Yakutsk: (650, 95)

Kamchatka: (700, 125)

Irkutsk: (620, 150)

Mongolia: (600, 175)

Japan: (680, 185)

China: (555, 210)

Afghanistan: (490, 190)

Middle East: (460, 245)

India: (535, 260)

Siam: (580, 275)

# AUSTRALIA

Indonesia: (570, 340)

New Guinea: (650, 350)

Western Australia: (605, 430)

Eastern Australia: (665, 430)

"""