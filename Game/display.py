import pygame 
from Game.config import *

class RiskDisplay():
    def __init__(self,time_delay):
        self.time_delay = time_delay

    def draw(self,board):
        #only initialize pygame on the first call of draw. 
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
            pygame.display.set_caption("RISK Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, FONT_SIZE)
            self.small_font = pygame.font.Font(None, FONT_SIZE//2)
        #fill the screen with white
        self.screen.fill(WHITE)

        #draw background
        draw_background(self.screen)

        #draw connection lines first
        for key in board.board_dict.keys():
            territory = board.board_dict[key]
            #draw a line for each connection
            for neighbor in territory.adjecency_list:
                pygame.draw.line(self.screen, BLACK, grid_pos_to_screen_pos(territory.pos), grid_pos_to_screen_pos(neighbor.pos), LINE_WIDTH)

        for index,key in enumerate(board.board_dict.keys()):
            territory = board.board_dict[key]
            circle_center = grid_pos_to_screen_pos(territory.pos) 
            screen_radius = grid_to_screen_scale((RADIUS, RADIUS))
            
            #create a rectangle for the ellipse
            circle_rect = pygame.Rect(0, 0, screen_radius[0] * 2, screen_radius[1] * 2)  
            circle_rect.center = circle_center 

            #draw the ellipse
            pygame.draw.ellipse(self.screen, COLORS[territory.player_index], circle_rect, 0)

            #draw text
            text = self.font.render(str(territory.troops) + (" " + str(territory.troops_to_add) + "*" if territory.troops_to_add > 0 else ""), True, BLACK)
            text_rect = text.get_rect(center=circle_center) 
            self.screen.blit(text, text_rect)

            #draw text
            text = self.small_font.render(str(index), True, BLACK)
            text_rect = text.get_rect()
            text_rect.left = circle_center[0] + RADIUS*50
            text_rect.centery = circle_center[1] 
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.time.delay(self.time_delay)

    def close(self):
        pygame.quit()