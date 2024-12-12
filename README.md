Welcome! This is my final project for the class CS-5891-02 Special Topics (Reinforcement Learning).

## Abstract
Energy efficiency is a critical challenge in cloud computing, with data centers consuming 460 TWh in
2022, a figure projected to exceed 1,000 TWh by 2026 under a worst-case scenario (IEA). This project
addresses this challenge by modeling a custom gym-compatible cloud environment and designing a reward
structure to prioritize energy efficiency and reduce task rejection rates. Furthermore, Stable Baslines3’s
on-policy Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C), and off-policy Deep
Q-Learning (DQN) were trained on a small-scale cloud environment. Experimental results demonstrated
that while A2C and PPO gave comparable results during training, PPO broke the tie during testing.
DQN, on the other hand, was not able to come up with a good policy leading to high energy consumption
and task rejection rates.

## Steps to train and test
To run my project follow these steps:
1. Clone the repo.
``` bash
git clone git@github.com:sanjana3012/Deep-Reinforcement-Learning-on-Custom-Cloud-Env.git
```
2. Create a virtual environment
``` bash
python3.9.18 -m venv venv
source venv/bin/activate
```
3. If you do not have python 3.9.18, pyenv is a great tool for installing python versions on the fly. After installing python 3.9.18, then create a virtual environment!
 ``` bash
brew update
brew install pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
source ~/.zshrc  # Or ~/.bashrc, ~/.bash_profile
```
3. Download appropriate libraries.
``` bash
pip install --upgrade pip
pip install stable-baselines3[extra]
pip install gym numpy matplotlib optuna
```
4. Perform hyperparameter optimization on agent!
``` bash
hyperparameter_train_ppo.py
```
5. Train the agent with best parameters!
``` bash
python train_ppo.py
```
6. Test the agent.
``` bash
 python test_ppo.py
 ```

## Descriptions of different scripts
1. The different hyperparameter_optimization_* scripts use optuna library and tune parameters for different agents.

2. The train scripts train those agents based on the best hyperparameters.

3. The best hyperparameters can be found in optuna_best_hyperparameters_* files.

4. The saved models can be found under final_training_logs/agent_name/agent_name_final_model.zip

5. The testing script is test_agents.py

6. The raw input file is called "input.txt" and the transformed file that is input to the enviroment is called "output_5000.txt". The number 5000 has nothing to do with the number of tasks in the file (sorry for the confusion, will change the name and update all of the other files to have that changed name).

