import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Game.config import *

class RiskSB3Wrapper(gym.Wrapper):
    """
    A wrapper for the Risk environment to make it compatible with Stable-Baselines3.
    This handles action space conversion and reward shaping.
    """
    def __init__(self, env, discrete_actions=True, reward_shaping=True):
        super().__init__(env)
        self.discrete_actions = discrete_actions
        self.reward_shaping = reward_shaping
        self.action_length = len(self.env.game.board.board_dict.keys()) + 1
        
        # Modify the action space based on chosen mode
        if discrete_actions:
            # Discrete action space for algorithms like PPO, A2C, DQN
            self.action_space = spaces.Discrete(self.action_length)
        else:
            # Continuous action space for algorithms like SAC, TD3
            self.action_space = spaces.Box(low=0, high=1, shape=(self.action_length,), dtype=np.float32)
        
        # Keep the original observation space
        self.observation_space = self.env.observation_space
        
        # Store previous observation for reward shaping
        self.previous_obs = None
        
    def reset(self, **kwargs):
        """Reset the environment and store initial observation"""
        obs, info = self.env.reset(**kwargs)
        self.previous_obs = obs
        return obs, info
    
    def step(self, action):
        """Convert action to the format expected by the wrapped env and shape rewards"""
        # Convert discrete action to probability vector if needed
        if self.discrete_actions:
            action_vector = np.zeros(self.action_length, dtype=np.float32)
            # Handle case when action is outside the valid range
            if action >= self.action_length:
                action = self.action_length - 1
            action_vector[action] = 1.0
        else:
            # For continuous actions, normalize to ensure they sum to 1
            action_vector = np.clip(action, 0, 1)
            if np.sum(action_vector) > 0:
                action_vector = action_vector / np.sum(action_vector)
            else:
                action_vector[-1] = 1.0  # Default to "do nothing" if all zeros
        
        # Take a step in the environment
        next_obs, reward, done, truncated, info = self.env.step(action_vector)
        
        # Apply reward shaping if enabled
        if self.reward_shaping:
            shaped_reward = self._shape_reward(self.previous_obs, next_obs, reward, done, info)
        else:
            shaped_reward = reward
        
        # Store current observation for next step
        self.previous_obs = next_obs
        
        return next_obs, shaped_reward, done, truncated, info
    
    def _shape_reward(self, prev_obs, curr_obs, original_reward, done, info):
        """Apply reward shaping to encourage strategic play"""
        reward = original_reward
        
        # Terminal reward (win/lose)
        if done:
            winner = info.get("Winner")
            current_player = info.get("Current Player")
            if winner is not None and winner == current_player:
                reward += 10.0  # Win bonus
        
        # Extract player territories and troops from observations
        player_territories_before, player_index, _, _ = self._extract_territories(prev_obs)
        player_territories_after, _, _, _ = self._extract_territories(curr_obs)
        
        # Reward for territory gains
        territory_diff = self._count_territories(player_territories_after[player_index]) - \
                         self._count_territories(player_territories_before[player_index])
        reward += territory_diff * 0.5
        
        # Continent control bonus
        reward += self._calculate_continent_bonus(player_territories_after[player_index]) * 0.25
        
        # Reward for relative strength compared to opponents
        reward += self._calculate_relative_strength(player_territories_after, player_index) * 0.1
        
        return reward
    
    def _extract_territories(self, obs):
        """Extract territory information from observation"""
        num_players = PLAYERS
        num_territories = TERRITORIES
        
        player_territories = [obs[p*num_territories:(p+1)*num_territories] for p in range(num_players)]
        one_hot_turn = obs[num_players*num_territories:num_players*num_territories+num_players]
        one_hot_phase = obs[num_players*num_territories+num_players:num_players*num_territories+num_players+PHASES]
        
        current_player_index = list(one_hot_turn).index(1) if 1 in one_hot_turn else 0
        
        return player_territories, current_player_index, one_hot_turn, one_hot_phase
    
    def _count_territories(self, territories):
        """Count number of territories owned"""
        return sum(1 for t in territories if t > 0)
    
    def _calculate_continent_bonus(self, territories):
        """Calculate continent control bonus"""
        bonus = 0
        territory_list = list(self.env.game.board.board_dict.keys())
        
        for continent_name, (active, _, continent_territories, continent_bonus) in self.env.game.board.continent_dict.items():
            if not active:
                continue
                
            continent_indices = [territory_list.index(t) for t in continent_territories]
            controlled = all(territories[i] > 0 for i in continent_indices)
            
            if controlled:
                bonus += continent_bonus
                
        return bonus
    
    def _calculate_relative_strength(self, player_territories, player_index):
        """Calculate relative strength compared to opponents"""
        player_troops = sum(player_territories[player_index])
        all_troops = sum(sum(territories) for territories in player_territories)
        
        if all_troops == 0:
            return 0
            
        return (player_troops / all_troops) - (1.0 / PLAYERS)  # Relative strength minus expected average
        
    def render(self, mode='Text'):
        """Pass through render call to the wrapped environment"""
        return self.env.render(mode)
        
    def close(self):
        """Pass through close call to the wrapped environment"""
        return self.env.close()