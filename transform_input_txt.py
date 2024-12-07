import pandas as pd

# Step 1: Load input.txt
column_names = ['JobType', 'JobID', 'TaskID', 'Priority', 'CPU', 'RAM', 'Disk']
data = pd.read_csv("input.txt", delim_whitespace=True, header=None, names=column_names)
num_rows=data.shape[0]
# print(f"Loaded {num_rows} rows from input.txt!")

# # Step 2: Open output_5000.txt for writing
# with open("output_5000.txt", "w") as f:
#     current_job = None

#     for _, row in data.iterrows():
#         # Step 3: Check if it's a new job
#         if row['JobID'] != current_job:
#             current_job = row['JobID']
#             f.write(f"J {int(current_job)}\n")  # Write job header

#         # Step 4: Write task details
#         task_line = f"{int(row['JobID'])} {int(row['TaskID'])} {row['CPU']} {row['RAM']} {row['Disk']} 1\n"
#         f.write(task_line)

# print("Transformed input.txt to output_5000.txt!")


