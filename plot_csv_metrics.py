import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a function to plot and save graphs
def plot_agent_data_with_rolling_average(file_paths, output_dir, window_size=10):
    """
    Plots rolling averages of data from CSV files for three agents and saves the plots in a directory.

    Args:
        file_paths (list): List of file paths for agent data.
        output_dir (str): Directory where plots will be saved.
        window_size (int): The window size for computing rolling averages.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Labels for the agents
    agent_labels = ['A2C', 'DQN', 'PPO']

    # Read data from each agent's file
    agents_data = [pd.read_csv(file_path) for file_path in file_paths if file_path != '']

    # Compute and plot rolling averages for Total Energy Consumed
    plt.figure()
    for i, data in enumerate(agents_data):
        rolling_avg = data['Total_Energy_Consumed'][:101].rolling(window=window_size).mean()
        plt.plot(data['Episode'][:101], rolling_avg, label=agent_labels[i])
    plt.title(f'Rolling Average of Total Energy Consumed (Window: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Total Energy Consumed')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'rolling_avg_total_energy_consumed.png'))
    plt.close()

    # Compute and plot rolling averages for Total Rejections
    plt.figure()
    for i, data in enumerate(agents_data):
        rolling_avg = data['Total_Rejections'][:101].rolling(window=window_size).mean()
        plt.plot(data['Episode'][:101], rolling_avg, label=agent_labels[i])
    plt.title(f'Rolling Average of Total Rejections (Window: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Total Rejections')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'rolling_avg_total_rejections.png'))
    plt.close()

    # Compute and plot rolling averages for Total Reward
    plt.figure()
    for i, data in enumerate(agents_data):
        rolling_avg = data['Total_Reward'][:101].rolling(window=window_size).mean()
        plt.plot(data['Episode'][:101], rolling_avg, label=agent_labels[i])
    plt.title(f'Rolling Average of Total Reward (Window: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'rolling_avg_total_reward.png'))
    plt.close()

    print(f"Plots with rolling averages saved in {output_dir}")


def plot_agent_data(file_paths, output_dir):
    """
    Plots data from CSV files for three agents and saves the plots in a directory.

    Args:
        file_paths (list): List of file paths for agent data.
        output_dir (str): Directory where plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Labels for the agents
    agent_labels = ['A2C', 'DQN', 'PPO']

    # Read data from each agent's file
    agents_data = [pd.read_csv(file_path) for file_path in file_paths if file_path != '']

    # Plot Energy Consumed
    plt.figure()
    for i, data in enumerate(agents_data):
        plt.plot(data['Episode'][:101], data['Total_Energy_Consumed'][:101], label=agent_labels[i])
    plt.title('Total Energy Consumed per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Energy Consumed')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'total_energy_consumed.png'))
    plt.close()

    # Plot Total Rejections
    plt.figure()
    for i, data in enumerate(agents_data):
        plt.plot(data['Episode'][:101], data['Total_Rejections'][:101], label=agent_labels[i])
    plt.title('Total Rejections per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Rejections')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'total_rejections.png'))
    plt.close()

    # Plot Total Reward
    plt.figure()
    for i, data in enumerate(agents_data):
        plt.plot(data['Episode'][:101], data['Total_Reward'][:101], label=agent_labels[i])
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'total_reward.png'))
    plt.close()

    print(f"Plots saved in {output_dir}")

# Example usage
if __name__ == '__main__':
    # Replace these with the actual file paths of your agents' data
    agent_files = [
        'logs/cloud_env/a2c/a2c_cleaned_data.csv',  # Data for A2C
        'logs/cloud_env/dqn/dqn_cleaned_data.csv',  # Data for PPO
        'logs/cloud_env/ppo/ppo_cleaned_data.csv',  # Data for DQN
    ]
    output_directory = "logs/cloud_env/plots"
    plot_agent_data(agent_files, output_directory)
    plot_agent_data_with_rolling_average(agent_files, output_directory)
