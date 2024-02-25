import pandas as pd
import numpy as np


# Load the selected vehicle data
vehicle_data = pd.read_csv('selected_vehicle_data.csv')

# Read the selected cell ID from the DataFrame
selected_cell_id = vehicle_data['cell_id'].unique()[0]

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

    # Save the updated DataFrame to a CSV file
    vehicle_data.to_csv('selected_vehicle_data.csv', index=False)

    # Print the information
    print(f"Resources of {len(vehicle_rows)} vehicles from cell ID {selected_cell_id} is calculated and saved into data.")

    return vehicle_data

evaluate_resources(vehicle_data, selected_cell_id)
