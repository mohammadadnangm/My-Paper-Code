import pandas as pd
import random

# Read the data
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get a list of unique cell IDs
unique_cell_ids = vehicle_data['cell_id'].unique()

# Get the count of vehicles in each cell
vehicle_counts = vehicle_data.groupby('cell_id').size()

# Display the unique cell IDs and their vehicle counts
print("Unique Cell IDs and their vehicle counts:")
for cell_id in unique_cell_ids:
    print(f"{cell_id}: {vehicle_counts[cell_id]} vehicles")

# Select a random cell ID
selected_cell_id = random.choice(unique_cell_ids)

# Save the selected cell ID to a file
with open('selected_cell_id.txt', 'w') as file:
    file.write(str(selected_cell_id))

print(f"Selected Cell ID: {selected_cell_id} has been saved to 'selected_cell_id.txt'")