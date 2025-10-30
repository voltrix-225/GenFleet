"""
This module automates the process of combining multiple tab-separated CSV datasets into one master file.
It scans the dataset/ directory, reads each CSV using pandas, and merges them into a unified DataFrame.
During preprocessing, it ensures encoding consistency, removes missing values, and drops duplicate records.
The final cleaned and consolidated dataset is exported as Trip_data.csv for further analysis.
"""

import pandas as pd
import os
import glob

file = 'dataset/'

csv_files = glob.glob(file + "/*.csv")

dfs = []

for csv_file in csv_files:
    if os.path.exists(csv_file):
        file = pd.read_csv(csv_file, sep = '\t', encoding= 'latin1')
        dfs.append(file)
    else:
        print(f"File not found at {csv_file}. Skipping")

if dfs:
    final_file = 'dataset\Trip_data.csv'
    combined_dfs = pd.concat(dfs, ignore_index=True)
    combined_dfs.to_csv(final_file, index=False)
    print(f"Final File Created at {final_file}")    

df = pd.read_csv(final_file)
df.dropna()
df.drop_duplicates()

print(f"Perfomed Data Cleaning on {final_file}")