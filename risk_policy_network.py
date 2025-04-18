import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class RiskFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the Risk game.
    Extracts relevant features from the game state.
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        n_input = int(np.prod(observation_space.shape))
        
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.feature_net(observations)

class RiskMaskedPolicy(ActorCriticPolicy):
    """
    Custom policy that masks invalid actions based on the current game state.
    """
    def __init__(self, *args, **kwargs):
        # Extract the game instance if provided
        self.game = kwargs.pop("game", None)
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        """
        super()._build_mlp_extractor()
    
    def _get_action_mask(self, obs):
        """
        Get a mask of valid actions for the current observation.
        
        Args:
            obs: The current observation
            
        Returns:
            A boolean mask where True indicates a valid action
        """
        if self.game is None:
            # If no game instance is available, allow all actions
            return np.ones(self.action_space.n, dtype=bool)
        
        try:
            # Extract game state info from observation
            from SimpleRiskAI.Game.config import PLAYERS, TERRITORIES, PHASES
            from SimpleRiskAI.AlphaZero.alpha_mcts import getStateInfo

            
            player_territories, player_index, phase, _ = getStateInfo(obs)
            
            # Create a temporary game state
            self.game.reset()
            
            # Set the game state based on observation
            for p_idx, territories in enumerate(player_territories):
                for t_idx, troops in enumerate(territories):
                    if troops > 0:
                        terr_key = list(self.game.board.board_dict.keys())[t_idx]
                        self.game.board.setTroops(terr_key, troops, self.game.player_list[p_idx])
                        self.game.player_list[p_idx].gainATerritory(terr_key)
            
            self.game.currentPlayer = player_index
            self.game.currentPhase = phase
            
            # Get valid actions based on current state
            valid_actions = self.game.player_list[player_index].getValidActions(
                self.game.board, phase)
            
            # Create the mask
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[valid_actions] = True
            
            return mask
            
        except Exception as e:
            # If there's any error, allow all actions as fallback
            print(f"Error generating action mask: {e}")
            return np.ones(self.action_space.n, dtype=bool)
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass in the policy network with action masking.
        """
        # Regular forward pass from parent class
        dist, value = super().forward(obs, deterministic=deterministic)
        
        if isinstance(obs, dict):
            obs = obs["obs"]
        
        # If using a batch of observations
        if len(obs.shape) > 1:
            masks = []
            for single_obs in obs.cpu().numpy():
                masks.append(self._get_action_mask(single_obs))
            mask = np.stack(masks)
        else:
            # Single observation
            mask = self._get_action_mask(obs.cpu().numpy())
        
        # Convert mask to PyTorch tensor
        mask_tensor = th.tensor(mask, dtype=th.float32, device=self.device)
        
        # Apply the mask to the distribution's logits
        if hasattr(dist, "logits"):
            # For categorical distribution
            dist.logits = dist.logits + th.log(mask_tensor + 1e-10)
            
        return dist, value