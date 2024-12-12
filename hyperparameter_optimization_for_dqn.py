import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

def env_fn():
    """
    Environment factory function for DQN.
    Creates a custom environment for training.
    
    Returns:
        gym.Env: Custom CloudEnv instance.
    """
    from custom_gym_cloud_env import CloudEnv  # Ensure this is correctly imported
    
    num_task = 1000
    num_server = 4
    return CloudEnv(
        scale='small',  
        fname='output_5000.txt',
        num_task=num_task,  # Number of tasks per episode
        num_server=num_server,
        file_path=f"logs/cloud_env/dqn/dqn_training_data_with_{num_task}_tasks_and_{num_server}_servers_{int(time.time())}.csv",
    )

def objective(trial: Trial):
    """
    Objective function for Optuna hyperparameter tuning for DQN.
    
    Args:
        trial (Trial): Optuna trial object.
    
    Returns:
        float: Mean evaluation reward.
    """
    # Define the hyperparameter search space
    hyperparams = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
        'learning_starts': trial.suggest_categorical('learning_starts', [1000, 5000, 10000]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'gamma': trial.suggest_float('gamma', 0.90, 0.999, log=True),
        'target_update_interval': trial.suggest_categorical('target_update_interval', [1000, 5000, 10000]),
        'train_freq': trial.suggest_categorical('train_freq', [4, 8, 16]),
        'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
        'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
    }
    
    # Create log directory for this trial
    trial_log_dir = os.path.join("optuna_logs", "dqn", f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    
    # Configure Stable Baselines logger
    new_logger = configure(trial_log_dir, ["stdout", "csv", "tensorboard"])
    
    # Create the environment
    env = make_vec_env(env_fn, n_envs=1, seed=42)
    
    # Initialize the DQN model with hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=trial_log_dir,
        **hyperparams
    )
    
    model.set_logger(new_logger)
    
    # Define training timesteps for tuning (use fewer steps for faster optimization)
    tuning_timesteps = 10000  # Adjust as needed
    
    # Train the model
    model.learn(total_timesteps=tuning_timesteps)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=5,  # Number of evaluation episodes
        deterministic=True
    )
    
    # Clean up
    del model
    env.close()
    
    # Optuna tries to maximize the objective, so return mean_reward
    return mean_reward

if __name__ == '__main__':
    # Define the Optuna sampler and study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='dqn_hyperparam_tuning')
    
    # Number of trials for hyperparameter tuning
    n_trials = 5  # Adjust based on computational resources
    
    # Optimize the study
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    # Display the best trial
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    
    print("Best hyperparameters:")
    with open(f"optuna_best_hyperparameters_dqn_{int(time.time())}.txt", "w") as f:
        f.write(str(study.best_params))
    print(study.best_params)
    
    # Retrain with best hyperparameters
    # Extract the best hyperparameters
    