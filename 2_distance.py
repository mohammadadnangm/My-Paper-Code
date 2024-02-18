import pandas as pd
import numpy as np
import math
import ast

# Assuming the vehicle_data DataFrame is already loaded
vehicle_data = pd.read_csv('vehicle_data.csv')

# Read the selected cell ID from the file
with open('selected_cell_id.txt', 'r') as file:
    selected_cell_id = file.read().strip()

# Function to calculate distance from the cell center
def calculate_distance(vehicle_data, selected_cell_id, cell_size):
    # Get the cell center and radius for the selected cell ID
    cell_center, cell_radius = get_cell_data(selected_cell_id, cell_size)

    # Filter the DataFrame to only include vehicles in the selected cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]

    # Iterate over all vehicles in the cell
    for index, row in cell_vehicles.iterrows():
        # Calculate the Euclidean distance between the vehicle and the cell center
        raw_distance = math.sqrt((row['longitude'] - cell_center[0])**2 + (row['latitude'] - cell_center[1])**2)

        # Subtract the cell radius from the calculated distance
        distance = raw_distance - cell_radius

        # Save the calculated distance in the DataFrame
        vehicle_data.loc[index, 'distance'] = distance

    # Save the updated DataFrame to a CSV file
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    # Print the information
    print(f"Distances of all vehciles in cell ID {selected_cell_id} having {len(cell_vehicles)} vehicles calculated and saved into data.")

    return vehicle_data


# Function to calculate cell data
def get_cell_data(cell_id, cell_size):
    # Convert the cell ID back to a tuple
    cell_id = ast.literal_eval(cell_id)

    # Calculate the cell center
    cell_center_x = (float(cell_id[0]) * cell_size) + (cell_size / 2)
    cell_center_y = (float(cell_id[1]) * cell_size) + (cell_size / 2)
    cell_center = (cell_center_x, cell_center_y)

    # Calculate the cell radius
    cell_radius = cell_size / 2

    return cell_center, cell_radius

calculate_distance(vehicle_data, selected_cell_id, cell_size=500)