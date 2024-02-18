import pandas as pd
import numpy as np
import random
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

# Function to evaluate resources
def evaluate_resources(vehicle_data, selected_cell_id):
    # Get the rows for the vehicles in the selected cell from the DataFrame
    vehicle_rows = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]

    # Define the weight factors for each component of the resources evaluation
    processing_power_weight = 0.5
    available_storage_weight = 0.5

    # Iterate over each vehicle in the cell
    for index, vehicle_row in vehicle_rows.iterrows():
        # Get the processing power and available storage for the vehicle from the DataFrame
        processing_power = vehicle_row['processing_power']
        available_storage = vehicle_row['available_storage']

        # Calculate the resources evaluation
        resources = (processing_power_weight * processing_power +
                     available_storage_weight * available_storage)

        # Save the calculated resources in the DataFrame
        vehicle_data.loc[index, 'resources'] = resources

    return vehicle_data

# Function to select the group leader
def select_group_leader(vehicle_data, selected_cell_id):
    # Filter the DataFrame to only include vehicles in the selected cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]

    # Initialize the maximum total value and the group leader ID
    max_total_value = -1
    group_leader_id = None

    # Iterate over all vehicles in the cell
    for index, row in cell_vehicles.iterrows():
        # Calculate the centrality of the vehicle
        Centerness = 1 / row['distance'] if row['distance'] != 0 else float('inf')

        # Calculate the total value for the vehicle
        traffic_rules_obeyed = int(row['traffic_rules_obeyed'])  # Convert boolean to integer
        Resources = row['resources']
        total_value = traffic_rules_obeyed + Centerness + Resources

        # Update the maximum total value and the group leader ID
        if total_value > max_total_value:
            max_total_value = total_value
            group_leader_id = row['vehicle_id']

    # Initialize the 'group_leader' column with False for only the vehicles in the selected cell
    vehicle_data.loc[vehicle_data['cell_id'] == selected_cell_id, 'group_leader'] = False

    # Now you can assign boolean values without any warning
    vehicle_data.loc[vehicle_data['vehicle_id'] == group_leader_id, 'group_leader'] = True

    print("Group Leader Cell ID: ", selected_cell_id)
    print("Number of vehicles in cell: ", len(cell_vehicles))
    print("Group Leader ID: ", group_leader_id)

    # Save the modified DataFrame to a CSV file
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    # Save the leader information to a new file
    with open('leader_info.txt', 'w') as file:
        file.write(f"Cell ID: {selected_cell_id}\n")
        file.write(f"Number of vehicles in cell: {len(cell_vehicles)}\n")
        file.write(f"Leader ID: {group_leader_id}\n")

    return vehicle_data  # Return the modified DataFrame

vehicle_data = calculate_distance(vehicle_data, selected_cell_id, cell_size=500)
vehicle_data = evaluate_resources(vehicle_data, selected_cell_id)
vehicle_data = select_group_leader(vehicle_data, selected_cell_id)

# Save the DataFrame back to the CSV file after all functions have been called
vehicle_data.to_csv('vehicle_data.csv', index=False)