import pandas as pd
# data = pd.read_csv("output_5000.txt", delim_whitespace=True, header=None)
# print(data.head(5))
count=0
with open("output_5000.txt", "r") as f:
    for line_num, line in enumerate(f, start=1):
        # Skip job headers (lines starting with 'J')
        if line.startswith("J"):
            continue
        count+=1
        # Check the number of fields
        fields = line.strip().split()
        if len(fields) != 6:
            print(f"Error on line {line_num}: {line.strip()}")

print(f"Total number of tasks: {count}")
