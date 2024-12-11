import pandas as pd
import os

def calculate_statistics(csv_file, output_cleaned_file):
    """
    Calculates average, minimum, and maximum for each column in the given CSV file
    after filtering out rows where Total Tasks > 1000 or Total Rejections > 1000.

    Args:
        csv_file (str): Path to the CSV file.
        output_cleaned_file (str): Path to save the cleaned CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the statistics for each column.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Filter rows where Total Tasks <= 1000 and Total Rejections <= 1000
    filtered_data = data[(data['Total Tasks'] <= 1000) & (data['Total_Rejections'] <= 1000)]
    
    # Save the cleaned data to a new file
    filtered_data.to_csv(output_cleaned_file, index=False)
    print(f"Cleaned data saved to: {output_cleaned_file}")
    
    # Calculate statistics
    statistics = pd.DataFrame({
        'Mean': filtered_data.mean(),
        'Min': filtered_data.min(),
        'Max': filtered_data.max()
    })
    
    return statistics

if __name__ == '__main__':
    # Replace this with the path to your CSV file
    csv_file = 'logs/cloud_env/ppo/ppo_training_data_with_1000_tasks_and_4_servers_1733872755.2762594.csv'
    output_cleaned_file = 'logs/cloud_env/ppo/ppo_cleaned_data.csv'

    # Calculate statistics
    stats = calculate_statistics(csv_file, output_cleaned_file)

    # Display the statistics
    print("Statistics for the filtered dataset:")
    print(stats)

    # Save the statistics to a CSV file
    stats_output_path = 'logs/cloud_env/stats/ppo_statistics_summary.csv'
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
    stats.to_csv(stats_output_path, index_label='Metric')
    print(f"Statistics saved to '{stats_output_path}'")
