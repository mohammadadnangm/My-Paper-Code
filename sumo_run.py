import pandas as pd
import numpy as np
import traci
import pytz
import datetime
import math
import random
import ast
import matplotlib.pyplot as plt

# Function to get current date and time
def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Karachi"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

# Command to start SUMO GUI
sumoCmd = ["sumo-gui", "-c", "osm.sumocfg", "--start"]

try:
    traci.start(sumoCmd)
    traci.simulationStep()  # Start the simulation
except Exception as e:
    print(f"Failed to start SUMO GUI: {e}")
    exit(1)

data_accumulator = []  # List to accumulate data

# Define the size of each cell (in meters)
cell_size = 250  # 50m x 50m

# Create a data structure to hold the cells and the vehicles in each cell
cells = {}

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    # During the simulation, assign each vehicle to a cell based on its position
    for i in range(0, len(vehicles)):
        try:
            # Collect required vehicle properties
            timestep = getdatetime()
            speed = round(traci.vehicle.getSpeed(vehicles[i]) * 3.6, 2)  # Speed in km/h
            x, y = traci.vehicle.getPosition(vehicles[i])
            longitude, latitude = traci.simulation.convertGeo(x, y)  # Longitude and latitude
            laneid = traci.vehicle.getLaneID(vehicles[i])
            running_position = round(traci.vehicle.getDistance(vehicles[i]), 2)  # Running position from start of the lane

            # Calculate the cell id based on the position
            cell_id = (int(x // cell_size), int(y // cell_size))

            # Generate random values for the new attributes
            acting_frequency = random.randint(1, 20)  # Random integer between 1 and 20
            transmission_power = random.uniform(0.1, 5.0)  # Random float between 0.1 and 5.0
            traffic_rules_obeyed = random.choice([True, False])  # Random boolean value
            processing_power = random.uniform(1.0, 3.0)  # Random float between 1.0 and 3.0
            available_storage = random.randint(1, 100)  # Random integer between 1 and 100

            # Create a dictionary with vehicle data
            veh_data = {
                'datetime': timestep,
                'vehicle_id': vehicles[i],
                'speed': speed,
                'longitude': longitude,
                'latitude': latitude,
                'lane_id': laneid,
                'running_position': running_position,
                'cell_id': cell_id,
                'acting_frequency': acting_frequency,
                'transmission_power': transmission_power,
                'traffic_rules_obeyed': traffic_rules_obeyed,
                'processing_power': processing_power,
                'available_storage': available_storage
            }

            # Assign the vehicle to the cell
            if cell_id not in cells:
                cells[cell_id] = []
            cells[cell_id].append(veh_data)

            # Append vehicle data to the accumulator list
            data_accumulator.append(veh_data)

        except Exception as e:
            print(f"Failed to collect data for vehicle {vehicles[i]}: {e}")

# Convert the accumulated data to a DataFrame
vehicle_data = pd.DataFrame(data_accumulator)

# Save vehicle data to CSV file (outside the simulation loop)
vehicle_data.to_csv('vehicle_data.csv', index=False)

# When finished reading the data, stop the simulation
traci.close()  # Close the TraCI connection

# Assuming the vehicle_data DataFrame is already loaded
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get a list of unique cell IDs
unique_cell_ids = vehicle_data['cell_id'].unique()

# Select a random cell ID
random_cell_id = random.choice(unique_cell_ids)

# Function to evaluate resources
def evaluate_resources(random_cell_id):
    # Filter the DataFrame to only include vehicles in the given cell
    vehicles_in_cell = vehicle_data[vehicle_data['cell_id'] == str(random_cell_id)]

    # Iterate over all vehicles in the cell
    for index, row in vehicles_in_cell.iterrows():
        # Define the weight factors for each component of the resources evaluation
        processing_power_weight = 0.5
        available_storage_weight = 0.5

        # Get the processing power and available storage for the vehicle from the DataFrame
        processing_power = row['processing_power']
        available_storage = row['available_storage']

        # Calculate the resources evaluation
        resources = (processing_power_weight * processing_power +
                     available_storage_weight * available_storage)

        # Save the calculated resources in the DataFrame
        vehicle_data.loc[index, 'resources'] = resources

    # Save the updated DataFrame to the CSV file
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    return resources

# Function to calculate cell data
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

# Function to calculate distance from the cell center
def calculate_distance(random_cell_id):
    # Get the cell center and radius for the random cell ID
    cell_center, cell_radius = get_cell_data(random_cell_id)

    # Initialize the 'distance' column for the entire DataFrame
    vehicle_data['distance'] = np.nan

    # Filter vehicles that are in the random cell
    vehicles_in_cell = vehicle_data[vehicle_data['cell_id'] == str(random_cell_id)].copy()

    for index, row in vehicles_in_cell.iterrows():
        # Calculate the Euclidean distance between the vehicle and the cell center
        raw_distance = math.sqrt((row['longitude'] - cell_center[0])**2 + (row['latitude'] - cell_center[1])**2)

        # Subtract the cell radius from the calculated distance
        distance = raw_distance - cell_radius

        # Save the calculated distance in the copy of the DataFrame
        vehicle_data.loc[index, 'distance'] = distance

    # Return the DataFrame with the updated distances
    return vehicle_data[vehicle_data['cell_id'] == str(random_cell_id)]

# Function to select the group leader
def select_group_leader(random_cell_id):

    # Calculate resources for all vehicles in the given cell
    evaluate_resources(random_cell_id) 

    # Calculate distances for all vehicles in the randomly selected cell
    calculate_distance(random_cell_id)

    # Filter the DataFrame to only include vehicles in the randomly selected cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == str(random_cell_id)]

    # Initialize the maximum total value and the group leader ID
    max_total_value = -1
    group_leader_id = None

    # Iterate over all vehicles in the cell
    for index, row in cell_vehicles.iterrows():
        # Calculate the centrality of the vehicle
        Centerness = 1 / row['distance'] if row['distance'] != 0 else float('inf')

        # Get the total value for the vehicle
        Rules_obyed = row['traffic_rules_obeyed']
        Resources = row['resources']  # assuming resources is a column in your DataFrame
        total_value = Rules_obyed + Centerness + Resources

        # Update the maximum total value and the group leader ID
        if total_value > max_total_value:
            max_total_value = total_value
            group_leader_id = row['vehicle_id']

    # Set the group_leader column to False for all vehicles
    vehicle_data['group_leader'] = False

    # Set the group_leader column to True for the selected group leader
    vehicle_data.loc[vehicle_data['vehicle_id'] == group_leader_id, 'group_leader'] = True

    # Save the updated DataFrame to the CSV file
    vehicle_data.to_csv('vehicle_data.csv', index=False)

    return group_leader_id


# Select a group leader for the random cell
group_leader_id = select_group_leader(random_cell_id)

# Print the selected random cell ID and the group leader ID
print(f"Selected random cell ID: {random_cell_id}")
print(f"Selected group leader ID: {group_leader_id}")

