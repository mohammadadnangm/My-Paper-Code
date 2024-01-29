import pandas as pd
import numpy as np
import random
import math
import ast

# Assuming the vehicle_data DataFrame is already loaded
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get a list of unique cell IDs
unique_cell_ids = vehicle_data['cell_id'].unique()

# Select a random cell ID
random_cell_id = random.choice(unique_cell_ids)


# Function to evaluate resources
def evaluate_resources(vehicle_data, random_cell_id):
    # Get the rows for the vehicles in the random cell from the DataFrame
    vehicle_rows = vehicle_data[vehicle_data['cell_id'] == random_cell_id]

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
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    return vehicle_data

evaluate_resources(vehicle_data, random_cell_id)


