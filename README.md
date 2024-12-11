To run my project follow these steps:
1. Clone the repo.
``` bash
git clone
```
2. Create a virtual environment
``` bash
python3 -m venv venv
source venv/bin/activate
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
