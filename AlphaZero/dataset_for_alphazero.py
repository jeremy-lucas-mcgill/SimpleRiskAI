from collections import deque
import pandas as pd
import random 

class RiskDataset:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
        self.df = None
        self.capacity = capacity

    def push(self,transition):
        self.buffer.append(transition)
    
    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)
    
    def add_game_outcome(self,winner_index):
        #turn the buffer into a pandas dataframe
        df = pd.DataFrame(self.buffer, columns=['state','action','player_index'])

        #if there was a winner
        if not winner_index == -1:
            #set the player index column to reflect the outcome of the game
            df['value'] = df['player_index'].apply(lambda x: 1 if x == winner_index else -1)
        else:
            #it was a draw as maxsteps was reached so set everything to 0
            df['value'] = 0

        #remove the player index column
        df = df[['state', 'action', 'value']]

        #normalize the state column
        df['state'] = df['state'].apply(self.normalize_state)

        #append this data to the current dataset
        if not hasattr(self, 'df'):
            self.df = df
        else:
            self.df = pd.concat([self.df, df], ignore_index=True)

            #limit the size of the dataset
            if len(self.df) > self.capacity:
                self.df = self.df.iloc[-self.capacity:]
        
        #clear the buffer
        self.buffer.clear()
    def normalize_state(self,state):
        max_value = max(state) if max(state) > 0 else 1
        return [max(0.1, round(t / max_value,1)) for t in state]