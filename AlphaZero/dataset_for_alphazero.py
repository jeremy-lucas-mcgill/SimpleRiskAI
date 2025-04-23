from collections import deque
import pandas as pd
import random 
from Game.config import *
from AlphaZero.alpha_mcts import enrich_features

class RiskDataset:
    def __init__(self,capacity,adjacency_dict):
        self.buffer = deque(maxlen=capacity)
        self.df = None
        self.capacity = capacity
        self.adjacency_dict = adjacency_dict

    def push(self,transition):
        self.buffer.append(transition)
    
    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)
    
    def add_game_outcome(self,winner_index,adjacency_matrix,remove_pass_move=False):
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
        df['state'] = df['state'].apply(lambda s: enrich_features(s, adjacency_matrix))

        df['action'] = df['action'].apply(self.suppress_pass)
        df = df[df['action'].notnull()]

        to_remove = self.action_distributions_log(df)
        #normalize actions
        df = self.normalize_action_distributions(df,to_remove)

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
    

    def suppress_pass(self,pi):
        threshold = 0.3
        pass_index = len(pi) - 1
        pi = pi.copy()

        if pi[pass_index] > threshold and (pi[:-1] > 0).any():
            pi[pass_index] = 0.0
            leftover = pi.sum()
            if leftover == 0:
                return None
            pi /= leftover

        return pi
    
    def action_distributions_log(self,df,debug=False):
        num_actions = len(df.iloc[0]['action'])
        nonzero_counts = np.zeros(num_actions, dtype=int)

        for idx, row in df.iterrows():
            action_array = np.array(row['action'])
            nonzero_mask = action_array != 0
            nonzero_counts += nonzero_mask.astype(int)
        debug and print(nonzero_counts,nonzero_counts[num_actions-1],np.mean(nonzero_counts[0:num_actions-1]),nonzero_counts[num_actions-1] - np.mean(nonzero_counts[0:num_actions-1]))
        to_remove = nonzero_counts[num_actions-1] - np.mean(nonzero_counts[0:num_actions-1])
        return int(to_remove) if to_remove > 0 else 0

    def normalize_action_distributions(self, df, num_to_remove):
        if num_to_remove > 0:
            # Define the index for the last action
            action_index = len(df.iloc[0]['action']) - 1 
            
            # Find rows where the last action is non-zero
            action_last_rows = df[df['action'].apply(lambda action: action[action_index] != 0)]
            
            # If there are more rows than the number to remove safely proceed
            if len(action_last_rows) > num_to_remove:
                # Randomly sample the rows to remove
                rows_to_remove = action_last_rows.sample(n=num_to_remove).index
                
                # Remove the selected rows from the dataframe
                df = df.drop(rows_to_remove)
            else:
                # If there are fewer rows than num_to_remove, remove all such rows
                df = df.drop(action_last_rows.index)
            
        return df

    def save(self, filepath):
        if self.df is not None:
            self.df.to_csv(filepath, index=False)