import gym
import numpy as np
import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def env_fn(agent_name):
    """
    Environment factory function.
    Creates a custom environment for training.

    Args:
        agent_name (str): Name of the agent ('ppo', 'dqn', or 'a2c').

    Returns:
        gym.Env: Custom CloudEnv instance.
    """
    from custom_gym_cloud_env import CloudEnv
    return CloudEnv(
        scale='small',
        fname='output_5000.txt',
        num_task=1000,
        num_server=4,
        file_path=f'logs/{agent_name}/testing_results.csv',
    )

def evaluate_agent(agent_name, model_path, num_episodes=10):
    """
    Evaluate a trained agent on the CloudEnv environment.

    Args:
        agent_name (str): Name of the agent ('ppo', 'dqn', 'a2c').
        model_path (str): Path to the trained model.
        num_episodes (int): Number of episodes to evaluate.
    """
    assert agent_name in ['ppo', 'dqn', 'a2c'], "agent_name must be 'ppo', 'dqn', or 'a2c'"
    
    # Initialize the environment
    env = env_fn(agent_name)

    # Load the trained model
    model = None
    if agent_name == 'ppo':
        model = PPO.load(model_path, env=env)
    elif agent_name == 'dqn':
        model = DQN.load(model_path, env=env)
    elif agent_name == 'a2c':
        model = A2C.load(model_path, env=env)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes, 
        render=False, 
        deterministic=True
    )

    # Print evaluation results
    print(f"Evaluation results for {agent_name.upper()}:")
    print(f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

def main():
    """
    Main function to train and evaluate PPO, DQN, and A2C agents on CloudEnv.
    """
    log_dir = 'final_training_logs/'

    # Define model paths
    ppo_model_path = os.path.join(log_dir, 'ppo_1733884148.357394', 'ppo_final_model.zip')
    dqn_model_path = os.path.join(log_dir, 'dqn_1733926347.484513', 'dqn_final_model.zip')
    a2c_model_path = os.path.join(log_dir, 'a2c_1733929133', 'a2c_final_model.zip')

    # Evaluate PPO agent
    if os.path.exists(ppo_model_path):
        print("Evaluating PPO agent...")
        evaluate_agent(agent_name='ppo', model_path=ppo_model_path, num_episodes=10)
    else:
        print(f"PPO model not found at {ppo_model_path}")

    # Evaluate DQN agent
    if os.path.exists(dqn_model_path):
        print("Evaluating DQN agent...")
        evaluate_agent(agent_name='dqn', model_path=dqn_model_path, num_episodes=10)
    else:
        print(f"DQN model not found at {dqn_model_path}")

    # Evaluate A2C agent
    if os.path.exists(a2c_model_path):
        print("Evaluating A2C agent...")
        evaluate_agent(agent_name='a2c', model_path=a2c_model_path, num_episodes=10)
    else:
        print(f"A2C model not found at {a2c_model_path}")

if __name__ == "__main__":
    main()
