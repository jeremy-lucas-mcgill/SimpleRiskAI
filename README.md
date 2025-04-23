# SimpleRiskAI

This repository contains all code used to train an AlphaZero-based agent in a simplified Risk environment.

---

## 📁 Project Structure

- `AlphaZero/` – Contains all scripts used to implement the AlphaZero agent.
- `Game/` – Contains all scripts used to implement the simplified Risk environment.
- `Models/` – Contains trained models, organized by environment configuration.

---

## ⚙️ Configuring the Environment

Before running any scripts, adjust the environment settings in:

- **`Game/config.py`**  
  Here, you can configure:
  - The number of players
  - The continents included in the environment

> ⚠️ The current configuration includes North America, South America, Europe, and Africa with 2 players.  
> `gym_model_play.py` and `test_models.py` are pre-set to load the best model for this configuration.

---

## 🔧 Key Files

- **`gym_env.py`**  
  Implements the environment using the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API.

- **`gym_play.py`**  
  Runs a random agent with visualization. Use this to inspect the environment setup.

- **`gym_model_play.py`**  
  Runs a trained model in the environment.  
  **Parameters:**
  - `model_path`: Path to the saved model
  - `render`: Toggle for visualization
  - `num_episodes`: Number of episodes to simulate  
  All players are controlled by the loaded model.

- **`test_models.py`**  
  Runs a match between multiple models.  
  **Parameters:**
  - `model_configs`: A dictionary of model paths to the number of players
  - `num_episodes`: Number of episodes to run
  - `render`: Toggle for visualization  
  If fewer players are provided than players in the environment, the remaining players are assigned random agents.

- **`train_alphazero.py`**  
  Used to train models with AlphaZero.  
  **Parameters:**
  - `num_episodes`: Number of self-play episodes to run
  - `update_frequency`: Number of training steps per update
  - `save_frequency`: Frequency at which the model is saved

---

## 📎 Repository URL

https://github.com/jeremy-lucas-mcgill/SimpleRiskAI
