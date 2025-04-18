# SimpleRiskAI
# Risk Game Reinforcement Learning with Stable-Baselines3

This repository implements and compares different reinforcement learning algorithms from Stable-Baselines3 for playing the board game Risk. The implementation allows training and evaluating several state-of-the-art RL algorithms and comparing them against the existing AlphaZero implementation.

## Project Structure

- `risk_sb3_wrapper.py`: Adapts the Risk environment for Stable-Baselines3 compatibility
- `risk_policy_network.py`: Custom neural network architectures and action masking for policy networks
- `train_risk_sb3.py`: Main training script for Stable-Baselines3 models
- `evaluate_models.py`: Evaluation script to compare models in a tournament
- `run_trained_models.py`: Script to visualize trained models playing Risk

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd risk-rl
```

2. Install required dependencies:
```bash
pip install stable-baselines3[extra] gymnasium pygame numpy torch pandas matplotlib tqdm
```

## Training Models

To train models using Stable-Baselines3:

```bash
python train_risk_sb3.py
```

This will train four different RL algorithms:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)

Each model will be trained for 250,000 timesteps by default. The script will:
1. Create a vectorized environment for parallel training
2. Set up custom policy networks with action masking
3. Apply reward shaping to improve learning
4. Save models and evaluation metrics
5. Generate comparison visualizations

## Evaluating Models

To evaluate trained models against each other and AlphaZero:

```bash
python evaluate_models.py --model_dir ./logs/risk_sb3_YYYY-MM-DD_HH-MM-SS/models --output_dir ./evaluation_results
```

This will:
1. Load all trained models from the specified directory
2. Run a tournament where each model plays against every other model
3. Generate win rate heatmaps and comparison charts
4. Specifically compare SB3 models against the AlphaZero implementation

## Visualizing Trained Models

To see a trained model play the game:

```bash
python run_trained_models.py --model_path ./logs/risk_sb3_YYYY-MM-DD_HH-MM-SS/models/PPO_final.zip --episodes 3 --render_mode Visual
```

This will run the specified model for 3 episodes with visual rendering.

## Implementation Details

### Environment Wrapper

The `RiskSB3Wrapper` class adapts the existing Risk environment for compatibility with Stable-Baselines3:
- Converts between discrete and continuous action spaces
- Implements reward shaping for better learning
- Handles action masking for valid moves

### Custom Neural Networks

The implementation includes:
- `RiskFeaturesExtractor`: Custom feature extraction from game state
- `RiskMaskedPolicy`: Policy network with action masking to prevent invalid moves

### Reward Shaping

The environment wrapper implements advanced reward shaping:
- Territory control rewards
- Continent bonus rewards
- Relative strength compared to opponents
- Terminal rewards for winning/losing

### Action Space Handling

The implementation supports both:
- Discrete action space (for PPO, A2C, and DQN)
- Continuous action space (for SAC)

## Comparing Algorithms

The different algorithms have distinct characteristics:

- **PPO**: Generally performs well with discrete action spaces and complex environments. Balances exploration and exploitation effectively.

- **A2C**: Simple and faster to train but may be less sample efficient than PPO.

- **DQN**: Works well for environments with discrete action spaces but can struggle with large action spaces like in Risk.

- **SAC**: Best suited for continuous control tasks, may need adaptation for the discrete nature of Risk.

## Tips for Further Improvement

1. **Hyperparameter Tuning**: The hyperparameters in `train_risk_sb3.py` are starting points and can be tuned for better performance.

2. **Longer Training**: For complex games like Risk, training for more timesteps may yield better results.

3. **Advanced Policy Networks**: Try implementing more complex neural network architectures in `risk_policy_network.py`.

4. **Multi-Agent Training**: Consider implementing a population-based training approach for better self-play.

5. **Curriculum Learning**: Start with simpler versions of the game and gradually increase complexity.

## Comparison with AlphaZero

AlphaZero's approach differs from SB3 algorithms in several ways:
- Uses Monte Carlo Tree Search (MCTS) for planning
- Combines planning with learned value and policy functions
- Learns entirely through self-play
- Doesn't use traditional reinforcement learning techniques

The evaluation script helps quantify how the SB3 models compare to the AlphaZero implementation in terms of win rate and playing style.
