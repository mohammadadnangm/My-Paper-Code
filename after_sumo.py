import matplotlib.pyplot as plt
import pandas as pd
import ast

# Define the size of each cell (in meters)
cell_size = 500

# Load the updated DataFrame from the CSV file
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get the count of vehicles in each cell
cell_vehicle_counts = vehicle_data['cell_id'].value_counts()


def get_cell_data(cell_id):
    # Convert the cell ID back to a tuple
    cell_id = ast.literal_eval(cell_id)

    # Calculate the cell center
    cell_center_x = (float(cell_id[0]) * cell_size) + (cell_size / 2)
    cell_center_y = (float(cell_id[1]) * cell_size) + (cell_size / 2)
    cell_center = (cell_center_x, cell_center_y)

    # Calculate the cell radius
    cell_radius = cell_size / 2

    return cell_center, cell_radius
# Plot the graph
plt.figure(figsize=(10, 6))
plt.bar(cell_vehicle_counts.index, cell_vehicle_counts.values, color='blue')
plt.xlabel('Cell ID')
plt.ylabel('Number of Vehicles')
plt.title('Distribution of Vehicles in Cells')

# Set y-axis limits for better visibility
plt.ylim(0, max(cell_vehicle_counts.values) + 10)

# Get the group leader's cell ID
leader_cell_id = vehicle_data.loc[vehicle_data['group_leader'], 'cell_id'].values[0]

# Get the center of the group leader's cell
leader_cell_center, leader_cell_radius = get_cell_data(leader_cell_id)

# Plot a marker or annotation for the group leader
plt.scatter(leader_cell_center[0], leader_cell_center[1], color='red', marker='*', s=200, label='Group Leader')

# Get a subset of other vehicles for visualization (e.g., 20 vehicles)
other_vehicles = vehicle_data[~vehicle_data['group_leader']].sample(n=20)

# Increase marker size for better visibility
plt.scatter(other_vehicles['longitude'], other_vehicles['latitude'], color='green', marker='o', s=50, label='Other Vehicles')

# Add legend
plt.legend()

plt.show()