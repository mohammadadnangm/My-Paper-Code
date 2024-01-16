import pandas as pd
import numpy as np
import traci
import pytz
import datetime
import math
import random

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
traci.load([])  # Close the SUMO GUI
traci.close()  # Close the TraCI connection

# Assuming the vehicle_data DataFrame is already loaded
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get a list of unique cell IDs
unique_cell_ids = vehicle_data['cell_id'].unique()

# Select a random cell ID
random_cell_id = random.choice(unique_cell_ids)


# Function to calculate direct trust
def calculate_direct_trust(random_cell_id):
    # Filter the DataFrame to only include vehicles in the specified cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == random_cell_id]

    # Create a new column 'direct_trust' and initialize it with NaN
    cell_vehicles['direct_trust'] = np.nan

    for index, row in cell_vehicles.iterrows():
        try:
            # Extract relevant attributes
            acting_frequency = row['acting_frequency']
            transmission_power = row['transmission_power']
            traffic_rules_obeyed = row['traffic_rules_obeyed']

            # Define the weight factors for each component of the direct trust metric
            acting_frequency_weight = 0.3
            transmission_power_weight = 0.3
            traffic_rules_obeyed_weight = 0.4

            # Calculate the direct trust metric
            direct_trust = (acting_frequency_weight * acting_frequency +
                            transmission_power_weight * transmission_power +
                            traffic_rules_obeyed_weight * traffic_rules_obeyed)

            # Assign the calculated direct trust to the 'direct_trust' column of the current row
            cell_vehicles.at[index, 'direct_trust'] = direct_trust

        except KeyError:
            print(f"Required data for vehicle with ID {row['vehicle_id']} not found in the data.")

    return cell_vehicles

# Function to calculate indirect trust
def calculate_indirect_trust(random_cell_id):
    # Filter the DataFrame to only include vehicles in the specified cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == random_cell_id]

    # Create a new column 'indirect_trust' and initialize it with NaN
    cell_vehicles['indirect_trust'] = np.nan

    for index, row in cell_vehicles.iterrows():
        try:
            # Get the neighbors for the current vehicle based on the cell
            neighbors = get_neighbors_based_on_cell(row['vehicle_id'])  # Replace with your function to get neighbors

            # Extract relevant attributes for the neighbors
            neighbor_direct_trusts = vehicle_data[vehicle_data['vehicle_id'].isin(neighbors)]['direct_trust']

            # Calculate the indirect trust as the average of the neighbors' direct trusts
            indirect_trust = neighbor_direct_trusts.mean()

            # Assign the calculated indirect trust to the 'indirect_trust' column of the current row
            cell_vehicles.at[index, 'indirect_trust'] = indirect_trust

        except KeyError:
            print(f"Required data for vehicle with ID {row['vehicle_id']} not found in the data.")

    return cell_vehicles

# Function to get neighbors based on cell
def get_neighbors_based_on_cell(vehicle_id):
    # Get the cell ID for the vehicle
    cell_id = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id]['cell_id'].values[0]

    # Get the list of vehicles in the same cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == cell_id]['vehicle_id'].values

    # Remove the current vehicle from the list
    cell_vehicles = cell_vehicles[cell_vehicles != vehicle_id]

    return cell_vehicles


# Function to calculate total trust
def calculate_total_trust(random_cell_id, beta=0.7):
    # Ensure beta is within the specified range
    if not 0.5 < beta < 1:
        raise ValueError("Beta must be between 0.5 and 1")

    # Calculate the direct and indirect trust for the vehicles in the cell
    direct_trust_df = calculate_direct_trust(random_cell_id)
    indirect_trust_df = calculate_indirect_trust(random_cell_id)

    # Merge the direct and indirect trust DataFrames
    cell_vehicles = pd.merge(direct_trust_df, indirect_trust_df, on='vehicle_id')

    # Create a new column 'total_trust' and initialize it with NaN
    cell_vehicles['total_trust'] = np.nan

    for index, row in cell_vehicles.iterrows():
        # Get the direct and indirect trust for the vehicle
        direct_trust = row['direct_trust']
        indirect_trust = row['indirect_trust']

        # Calculate the total trust as the weighted sum of the direct and indirect trust
        total_trust = beta * direct_trust + (1 - beta) * indirect_trust

        # Assign the calculated total trust to the 'total_trust' column of the current row
        cell_vehicles.at[index, 'total_trust'] = total_trust

    return cell_vehicles



# Function to calculate distance from the cell center
def calculate_distance(vehicle, cell_center, cell_radius, random_cell_id):
    # Check if the vehicle's cell ID matches the random cell ID
    if vehicle.cell_id != random_cell_id:
        return None

    # Calculate the Euclidean distance between the vehicle and the cell center
    raw_distance = math.sqrt((vehicle.x - cell_center.x)**2 + (vehicle.y - cell_center.y)**2)

    # Subtract the cell radius from the calculated distance
    distance = raw_distance - cell_radius

    # Save the calculated distance in the DataFrame
    vehicle_data.loc[vehicle_data['vehicle_id'] == vehicle.vehicle_id, 'distance'] = distance

    return distance

# Function to evaluate resources
def evaluate_resources(vehicle_id, random_cell_id):
    # Get the cell ID for the vehicle from the DataFrame
    vehicle_row = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id]
    vehicle_cell_id = vehicle_row['cell_id'].values[0]

    # Check if the vehicle's cell ID matches the random cell ID
    if vehicle_cell_id != random_cell_id:
        return None

    # Define the weight factors for each component of the resources evaluation
    processing_power_weight = 0.5
    available_storage_weight = 0.5

    # Get the processing power and available storage for the vehicle from the DataFrame
    processing_power = vehicle_row['processing_power'].values[0]
    available_storage = vehicle_row['available_storage'].values[0]

    # Calculate the resources evaluation
    resources = (processing_power_weight * processing_power +
                 available_storage_weight * available_storage)

    return resources

# Function to select the group leader
def select_group_leader(random_cell_id):
    # Filter the DataFrame to only include vehicles in the randomly selected cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == random_cell_id]

    # Initialize the maximum total value and the group leader ID
    max_total_value = -1
    group_leader_id = None

    # Iterate over all vehicles in the cell
    for index, row in cell_vehicles.iterrows():
        # Calculate the centrality of the vehicle
        Centerness = 1 / row['distance'] if row['distance'] != 0 else float('inf')

        # Calculate the total value for the vehicle
        Total_trust = row['total_trust']
        Resources = evaluate_resources(row['vehicle_id'], random_cell_id)
        total_value = Total_trust + Centerness + Resources

        # Update the maximum total value and the group leader ID
        if total_value > max_total_value:
            max_total_value = total_value
            group_leader_id = row['vehicle_id']

    return group_leader_id
