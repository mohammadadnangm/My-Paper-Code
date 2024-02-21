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

    # Initialize the 'group_leader' column with False for all vehicles
    if 'group_leader' not in vehicle_data.columns:
        vehicle_data['group_leader'] = False
    else:
        vehicle_data['group_leader'] = vehicle_data['group_leader'].astype(bool)

    # Now you can assign boolean values without any warning
    vehicle_data.loc[vehicle_data['vehicle_id'] == group_leader_id, 'group_leader'] = True

    print("Group Leader Cell ID: ", selected_cell_id)
    print("Number of vehicles in cell: ", len(cell_vehicles))
    print("Group Leader ID: ", group_leader_id)

    # Save the modified DataFrame to a CSV file
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    # Save the leader information to a new file
    with open('leader_info.txt', 'w') as file:
        file.write(f"Group Leader Cell ID: {selected_cell_id}\n")
        file.write(f"Number of vehicles in cell: {len(cell_vehicles)}\n")
        file.write(f"Group Leader ID: {group_leader_id}\n")

    return vehicle_data  # Return the modified DataFrame

# Call the function with the appropriate cell ID
select_group_leader(vehicle_data, selected_cell_id)