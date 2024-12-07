# train_ppo.py
import os
import sys
import gym
import numpy as np
import torch
from torch.optim import Adam
import time
import datetime  # Import datetime for timestamp
import random
current_dir = os.getcwd()
# print(f"Current Directory: {current_dir}")

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# print(f"Parent Directory: {parent_dir}")

sys.path.append(parent_dir)
import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.run_utils import setup_logger_kwargs

# Import your custom CloudEnv
from gym_cloud_env import CloudEnv  # Ensure the filename is correct

def env_fn():
    """
    Environment factory function for PPO.
    Creates a large-scale CloudEnv for comprehensive training.
    
    Parameters:
        scale (str): Determines the number of farms based on server count.
        fname (str): Filename containing task data.
        num_task (int): Number of tasks per episode.
        num_server (int): Number of servers in the environment.
    
    Returns:
        CloudEnv: An instance of the custom CloudEnv.
    """
    # Create a large-scale environment
    return CloudEnv(
        scale='small',         # Adjust scale if needed (small, medium, large)
        fname='output_5000.txt',
        num_task=1000,          # Increased from 50 to 5000
        num_server=150          # Increased from 20 to 300
    )

if __name__ == '__main__':
    # Generate a unique run name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"training_run_{timestamp}"
    
    # Setup logger with the unique run name and a fixed seed
    logger_kwargs = setup_logger_kwargs(run_name, seed=42)

    # Define PPO hyperparameters tailored for a larger environment
    ppo(
        env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[128, 128, 64]),  # Increased network size for better capacity
        steps_per_epoch=2000,    # Increased from 1000 to 10000
        epochs=20,                # Increased from 3 to 100 for thorough training
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=50,       # Increased from 20 to 50
        train_v_iters=50,        # Increased from 20 to 50
        lam=0.95,
        max_ep_len=1200,         # Increased from 2000 to 5000, adjust based on task durations
        target_kl=0.01,
        logger_kwargs=logger_kwargs,  # Use the uniquely named logger_kwargs
        save_freq=10              # Increased save frequency to every 10 epochs
    )
