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

Descriptions of different scripts:

