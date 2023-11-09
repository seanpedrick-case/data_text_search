# %%
import pandas as pd
import csv

# %%
# Define your file paths
file_dir = "../"
extracted_file_path = file_dir + "2022_08_case_notes.txt"
parquet_file_path = file_dir + "2022_08_case_notes.parquet"

# %%
# Read the TXT file using the csv module and convert to DataFrame
csv.field_size_limit(1000000)  # set to a higher value

data_list = []
with open(extracted_file_path, mode='r', encoding='iso-8859-1') as file:
    csv_reader = csv.reader(file, delimiter=',')  # Change the delimiter if needed
    for row in csv_reader:
        data_list.append(row)

# Filter rows that have the same number of columns as the header
header = data_list[0]
filtered_data = [row for row in data_list if len(row) == len(header)]

# Convert list of rows to DataFrame
casenotes = pd.DataFrame(filtered_data[1:], columns=header)  # Assuming first row is header

print(casenotes.head())  # Display the first few rows of the DataFrame

# %%
casenotes.to_parquet(parquet_file_path)


