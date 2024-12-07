from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import os
import datetime

class MetricLogger(BaseCallback):
    """
    Custom callback to log metrics from the `info` dictionary at the end of each episode.
    Logs metrics to the terminal and CSV file.
    """

    def __init__(self, log_frequency=100, verbose=0):
        super(MetricLogger, self).__init__(verbose)
        self.log_frequency = log_frequency  # Frequency of logging

    def _on_training_start(self) -> None:
        # Debug to confirm logger output formats
        print("[DEBUG] Logger Output Formats in Callback:", self.model.logger.output_formats)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            print("[DEBUG] Info Dictionary:", info)
            if "EpElecCost" in info:
                metrics = {
                    "EpElecCost": info["EpElecCost"],
                    "EpPowerUsed": info["EpPowerUsed"],
                    "EpEnergyConsumed": info["EpEnergyConsumed"],
                    "EpTasksRejected": info["EpTasksRejected"],
                    "AvgCPUUsagePerTask": info["AvgCPUUsagePerTask"],
                    "AvgRAMUsagePerTask": info["AvgRAMUsagePerTask"],
                }

                # Log metrics to terminal
                if self.verbose > 0:
                    print("--------------------------------")
                    for key, value in metrics.items():
                        print(f"{key}: {value}")
                    print("--------------------------------")

                # Use the model's logger to record metrics
                print("[DEBUG] Logging Metrics to CSV:", metrics)
                for key, value in metrics.items():
                    self.model.logger.record(key, value)

                # Dump metrics using the model's logger
                print("[DEBUG] Dumping logger data...")
                self.model.logger.dump(self.num_timesteps)

        return True



def env_fn():
    """
    Environment factory function for PPO.
    Creates a custom environment for training.
    
    Returns:
        gym.Env: Custom CloudEnv instance.
    """
    from gym_cloud_env import CloudEnv  # Ensure this is correctly imported
    return CloudEnv(
        scale='small',         # Adjust scale if needed (small, medium, large)
        fname='output_5000.txt',
        num_task=1000,         # Number of tasks per episode
        num_server=10,
        reward_type="simple",
    )


if __name__ == '__main__':
    from stable_baselines3.common.logger import configure

    # Configure Stable Baselines logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"ppo_training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    print("Logger Output Formats in main:", logger.output_formats)

    # Create the environment
    env = make_vec_env(env_fn, n_envs=1, seed=42)

    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        n_steps=1000,
        batch_size=50,
        learning_rate=3e-4,
        clip_range=0.2,
        vf_coef=1.0,
        seed=42,
        tensorboard_log=log_dir,
    )
    model.set_logger(logger)

    # Instantiate the custom MetricLogger
    custom_logging_callback = MetricLogger(verbose=1)

    # Train the model
    model.learn(
        total_timesteps=50*1000,
        callback=custom_logging_callback
    )
    

    # Save the final model
    model.save(os.path.join(log_dir, "ppo_final_model"))
    print(f"Training complete. Logs and models saved in: {log_dir}")