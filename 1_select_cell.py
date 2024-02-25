import pandas as pd

# Read the data
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get the count of vehicles in each cell
vehicle_counts = vehicle_data.groupby('cell_id').size()

# Find the cell ID with the maximum number of vehicles
selected_cell_id = vehicle_counts.idxmax()

# Save the selected cell ID to a file
with open('selected_cell_id.txt', 'w') as file:
    file.write(str(selected_cell_id))

# Print the selected cell ID and the number of vehicles in it
print(f"Cell ID: {selected_cell_id} having {vehicle_counts[selected_cell_id]} vehicles is selected.'")

# New code to filter the DataFrame and write it back to the CSV
selected_vehicle_data = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]
selected_vehicle_data.to_csv('selected_vehicle_data.csv', index=False)

# Print the information
print(f"Data for {vehicle_counts[selected_cell_id]} vehicles from cell ID {selected_cell_id} is filtered and saved into 'selected_vehicle_data.csv'.")