from stable_baselines3.common.env_util import make_vec_env
from gym_cloud_env import CloudEnv
def env_fn():
    """
    Environment factory function for creating instances of the custom CloudEnv.
    """
    return CloudEnv(
        scale='small',  # Adjust scale (small, medium, large)
        fname='output_5000.txt',
        num_task=500,
        num_server=6,
        reward_type="simple",
    )
env = make_vec_env(env_fn, n_envs=1, seed=42)
print(type(env))
