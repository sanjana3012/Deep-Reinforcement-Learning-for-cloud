import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import json
import optuna
from optuna import Trial
from optuna.samplers import TPESampler

def env_fn():
    """
    Environment factory function for A2C.
    Creates a custom environment for training.
    
    Returns:
        gym.Env: Custom CloudEnv instance.
    """
    from custom_gym_cloud_env import CloudEnv  # Ensure this is correctly imported
    
    num_task = 1000
    num_server = 4
    return CloudEnv(
        scale='small',  # Adjust scale if needed (small, medium, large)
        fname='output_5000.txt',
        num_task=num_task,  # Number of tasks per episode
        num_server=num_server,
        file_path=f"logs/cloud_env/a2c/a2c_training_data_with_{num_task}_tasks_and_{num_server}_servers_{int(time.time())}.csv",
    )

def objective(trial: Trial):
    """
    Objective function for Optuna hyperparameter tuning for A2C.
    
    Args:
        trial (Trial): Optuna trial object.
    
    Returns:
        float: Mean evaluation reward.
    """
    # Define the hyperparameter search space
    hyperparams = {
        'n_steps': trial.suggest_categorical('n_steps', [128, 256, 512]),
        'gamma': trial.suggest_float('gamma', 0.90, 0.999, log=True),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-5, 1e-1),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 0.9),
    }
    
    # Create log directory for this trial
    trial_log_dir = os.path.join("optuna_logs", "a2c", f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    
    # Configure Stable Baselines logger
    new_logger = configure(trial_log_dir, ["stdout", "csv", "tensorboard"])
    
    # Create the environment
    env = make_vec_env(env_fn, n_envs=1, seed=42)
    
    # Initialize the A2C model with hyperparameters
    model = A2C(
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
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='a2c_hyperparam_tuning')
    
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
    with open(f"optuna_best_hyperparameters_a2c_{int(time.time())}.txt", "w") as f:
        f.write(json.dumps(study.best_params, indent=4))
    print(study.best_params)
