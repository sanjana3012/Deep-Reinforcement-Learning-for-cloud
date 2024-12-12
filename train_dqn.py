import gym
import os
import time
import json
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# Define the environment factory function
def env_fn():
    """
    Environment factory function for dqn.
    Creates a custom environment for training.

    Returns:
        gym.Env: Custom CloudEnv instance.
    """
    from custom_gym_cloud_env import CloudEnv
    num_task = 1000
    num_server = 4
    return CloudEnv(
        scale='small',
        fname='output_5000.txt',
        num_task=num_task,
        num_server=num_server,
        file_path=f"logs/cloud_env/dqn/final_training_data_with_{num_task}_tasks_and_{num_server}_servers_{time.time()}.csv",
    )

def load_best_hyperparameters(file_path):
    """
    Loads the best hyperparameters from a JSON file.

    Args:
        file_path (str): Path to the JSON file with hyperparameters.

    Returns:
        dict: Best hyperparameters.
    """
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Path to the file containing the best hyperparameters
    best_hyperparams_file = "optuna_best_hyperparameters_dqn_1733887864.txt"

    # Load the best hyperparameters
    if not os.path.exists(best_hyperparams_file):
        raise FileNotFoundError(f"{best_hyperparams_file} not found. Ensure the tuning script has generated this file.")
    
    with open(best_hyperparams_file, "r") as f:
        best_hyperparams = eval(f.read())

    # Create log directory for final training
    final_log_dir = os.path.join("final_training_logs", f"dqn_{time.time()}")
    os.makedirs(final_log_dir, exist_ok=True)

    # Configure Stable Baselines logger
    final_logger = configure(final_log_dir, ["stdout", "csv", "tensorboard"])

    # Create the environment
    env = make_vec_env(env_fn, n_envs=1, seed=42)

    # Initialize the dqn model with the best hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=final_log_dir,
        **best_hyperparams
    )
    
    model.set_logger(final_logger)

    # Define training timesteps for final training
    final_training_timesteps = 150*1000

    # Train the model
    print("Starting final training...")
    model.learn(total_timesteps=final_training_timesteps)
    
    # Save the trained model
    model_save_path = os.path.join(final_log_dir, "dqn_final_model.zip")
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")

    # Clean up
    env.close()
    print("Final training complete.")